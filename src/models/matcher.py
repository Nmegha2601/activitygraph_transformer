import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import utils.segment_utils as segment_utils

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_action. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-actions).
    """

    def __init__(self, cost_class: float = 1, cost_segment: float = 1, cost_siou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_segment = cost_segment
        self.cost_siou = cost_siou
        assert cost_class != 0 or cost_segment != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           actions in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_segment = outputs["pred_segments"].flatten(0, 1)  # [batch_size * num_queries, 4]

        scale_factor = torch.stack([t["length"] for t in targets], dim=0)
        out_segment_scaled = out_segment * scale_factor.unsqueeze(1).repeat(1,num_queries,1).flatten(0,1).repeat(1,2)           

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_segment = torch.cat([v["segments"] for v in targets])
        tgt_segment_scaled = torch.cat([v["segments"] * v['length'] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        if tgt_segment.dim() > 1:
            # Compute the L1 cost between segments
            cost_segment = torch.cdist(out_segment, tgt_segment, p=1)
            # Compute the siou cost between segments 
            cost_siou = -segment_utils.generalized_segment_iou(tgt_segment_scaled, out_segment_scaled) 

        else:
            cost_segment = torch.zeros(cost_class.size()).to(cost_class.device)
            cost_siou = torch.zeros(cost_class.size()).to(cost_class.device)


        # Final cost matrix
        C = self.cost_segment * cost_segment + self.cost_class * cost_class + self.cost_siou * cost_siou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["segments"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [torch.as_tensor(i, dtype=torch.int64) for i in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_segment=args.set_cost_segment, cost_siou= args.set_cost_siou)



