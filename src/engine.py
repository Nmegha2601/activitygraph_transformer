import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os,sys
import copy
import numpy as np
import math
from typing import Iterable
import time

import utils.misc as utils
import datasets
from metrics.detection_metrics import ActionDetectionEvaluator

def train_one_epoch(epoch, max_norm, model, criterion, data_loader, optimizer, scheduler, device):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples.tensors, samples.mask)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    print("Train epoch:", epoch, "Averaged stats:", train_stats)
    return train_stats



@torch.no_grad()
def evaluate(epoch, model, criterion, postprocessors, data_loader, output_dir, dataset, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('mAP', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Test: [{}]'.format(epoch)
    print_every = 50

    predictions = {}
    groundtruth = {}

    for samples, targets in metric_logger.log_every(data_loader, print_every, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples.tensors, samples.mask)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(mAP=torch.tensor(-1))
        metric_logger.update(classification_mAP=torch.tensor(-1))

        scale_factor = torch.cat([t["length"].data for t in targets], dim=0)
        target_lengths = scale_factor.unsqueeze(1).repeat(1,2)
        results = postprocessors['segments'](outputs, targets, target_lengths)
            

        data_utils = getattr(datasets,dataset+'_utils')

        res = {data_utils.getVideoName(target['video_id'].tolist()): output for target, output in zip(targets, results)}
        gt = {data_utils.getVideoName(target['video_id'].tolist()): target for target in targets}

        predictions.update(res)
        groundtruth.update(gt)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    ######For mAP calculation need to gather all data###########
    all_predictions = utils.all_gather(predictions)
    all_groundtruth = utils.all_gather(groundtruth)

    stats = {}
    evaluator = ActionDetectionEvaluator(dataset, all_groundtruth, all_predictions) 
    detection_stats = evaluator.evaluate()

    stats = {'mAP': detection_stats}

    metric_logger.update(**stats)

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if 'mAP' not in k}
    test_stats.update({k: meter.value for k, meter in metric_logger.meters.items() if 'mAP' in k})

    print("Test epoch:", epoch, "Averaged test stats:",  test_stats)
    return test_stats
    



