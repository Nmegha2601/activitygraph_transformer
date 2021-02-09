"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .matcher import build_matcher
from .transformer import build_transformer
from .joiner import build_joiner

import numpy as np
from utils.misc import accuracy, get_world_size, get_rank,is_dist_avail_and_initialized
from utils import segment_utils as segment_utils

class AGT(nn.Module):
    """ This is the AGT module that performs temporal localization """
    def __init__(self, joiner, transformer, dim_feedforward, num_classes, num_queries, aux_loss = True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of action classes
            num_queries: number of action queries, ie detection slot. This is the maximal number of actions
                         that can be detected in a video.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.joiner = joiner
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.segments_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.input_proj = nn.Conv1d(2048, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss

    def forward(self, samples, mask):
        """ The forward expects two inputs:
               - samples.tensor: batched videos features, of shape [batch_size x 2048 x T]
               - samples.mask: a binary mask of shape [batch_size x T], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized boxes coordinates for all queries, represented as
                               (start_time, end_time). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        assert mask is not None
        src, pos = self.joiner(samples,mask)
        hs, _, edge = self.transformer(self.input_proj(src), (mask==1), self.query_embed.weight,pos)

        outputs_class = self.class_embed(hs)
        outputs_segments = F.relu(self.segments_embed(hs))

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_segments[-1],'edges': edge}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segments)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segments):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b} for a, b in zip(outputs_class[:-1], outputs_segments[:-1])]



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segments, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [num_segments, 2]
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)

        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_segments_scaled = torch.cat([t['segments'][i] * t['length'] for t, (_, i) in zip(targets, indices)], dim=0)

        src_segments_orig = outputs['pred_segments']
        src_segments = src_segments_orig[idx] 

        src_segments_scaled = src_segments_orig * torch.cat([t['length'] for t in targets],dim=0).unsqueeze(1).repeat(1,2)[:,None,:]
        src_segments_scaled = src_segments_scaled[idx] 

        if target_segments.dim() > 1:
            loss_segment = F.l1_loss(src_segments, target_segments, reduction='none')
            losses = {}
            losses['loss_segment'] = loss_segment.sum()/num_segments
            
            loss_siou = 1 - torch.diag(segment_utils.generalized_segment_iou(target_segments_scaled,src_segments_scaled))
            losses['loss_siou'] = loss_siou.sum()/num_segments

        else:
            losses = {}
            losses['loss_segment'] = 0
            losses['loss_siou'] = 0


        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_segments):
        """ Compute the cardinality error, ie the absolute error in the number of predicted no-action segments
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-action" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

          

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'segments': self.loss_segments,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        losses = {}

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessMatched(nn.Module):
    def __init__(self,matcher):
       super().__init__()
       self.matcher = matcher

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self,outputs,targets,target_lengths):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        if targets[0]['segments'].dim() > 1:
            out_logits = outputs['pred_logits']
            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)
    
            out_segments = outputs['pred_segments']
            scale_factor = target_lengths
            segments = out_segments * scale_factor[:,None,:]
 
            idx = self._get_src_permutation_idx(indices)
            out_scores = []
            out_labels = []
            out_segments = []

            for i in list(set(idx[0].tolist())):
                logit_idx = [logit for bidx, logit in zip(idx[0].tolist(),idx[1].tolist()) if bidx == i]
                segment_idx = [segment for bidx, segment in zip(idx[0].tolist(),idx[1].tolist()) if bidx == i]
                out_scores.append(scores[i][logit_idx])
                out_labels.append(labels[i][logit_idx])
                out_segments.append(segments[i][segment_idx,:])

            if 'edges' in outputs.keys():
              if type(outputs['edges']) != type(None):
                edges_orig = outputs['edges']
                output_edges = []
                for i in list(set(idx[0].tolist())):
                    edge_ids = [src_index for bidx, src_index in zip(idx[0].tolist(),idx[1].tolist()) if bidx == i]
                    output_edges.append(edges_orig[i][edge_ids,:][:,edge_ids])
  
                results = [{'score': torch.tensor(scr), 'label': torch.tensor(lbl), 'segment': torch.tensor(seg), 'edge': torch.tensor(e)} for scr, lbl, seg, e in zip(out_scores, out_labels, out_segments, output_edges)]

            else:
                results = [{'score': torch.empty(targets[0]['segments'].size()), 'label': torch.empty(targets[0]['segments'].size()), 'segment': torch.empty(targets[0]['segments'].size()), 'edge': torch.empty(targets[0]['segments'].size())}]

        else:
            results = [{'score': torch.empty(targets[0]['segments'].size()), 'label': torch.empty(targets[0]['segments'].size()), 'segment': torch.empty(targets[0]['segments'].size()), 'edge': torch.empty(targets[0]['segments'].size())}]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = args.num_classes
    
    joiner = build_joiner(args)
 
    transformer = build_transformer(args)

    if args.model == 'agt':
        model = AGT(
          joiner,
          transformer,
          dim_feedforward=args.dim_feedforward,        
          num_classes=args.num_classes,
          num_queries=args.num_queries,
          aux_loss=args.aux_loss,
        )
        device = torch.device(args.device)

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_segment': args.segment_loss_coef, 'loss_siou': args.siou_loss_coef}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'segments', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'segments': PostProcessMatched(matcher)}

    print(model)
    return model, criterion, postprocessors


