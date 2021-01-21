import torch
import numpy as np

def segment_iou(target_segment,candidate_segments):
    tt1 = torch.max(target_segment[0], candidate_segments[:, 0])
    tt2 = torch.min(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clamp(min=0)

    # Segment union.
    segments_union = (candidate_segments[:,1] - candidate_segments[:,0])  + (target_segment[1] - target_segment[0]) - segments_intersection

    tIoU = segments_intersection / segments_union

    tIoU[torch.isnan(tIoU)] = 0
    tIoU[torch.isinf(tIoU)] = 0

    return tIoU

def generalized_segment_iou(target_segments,candidate_segments):
    if candidate_segments.ndim !=2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = torch.zeros(n, m)
    for i in range(m):
        tiou[:, i] = segment_iou(target_segments[i,:], candidate_segments)

    tiou[torch.isnan(tiou)] = 0
    tiou[torch.isinf(tiou)] = 0
    return torch.tensor(tiou,device=candidate_segments.device)

