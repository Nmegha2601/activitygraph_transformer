import json
import os,sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import torch
from metrics.metrics_utils import interpolated_prec_rec
from metrics.metrics_utils import segment_iou

class ActionDetectionEvaluator(object):
    def __init__(self, dataset, groundtruth, prediction, tiou_thresholds=np.linspace(0.1, 0.5, 5), subset='testing', verbose=True):
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
        self.activity_index_file = os.path.join('data',dataset,'action_mapping.txt')

        # Import ground truth and predictions.
        self.groundtruth, self.activity_index = self._import_ground_truth(os.path.join('data', dataset, dataset+'_action.json'),groundtruth)
        self.prediction = self._import_prediction(prediction)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.groundtruth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    
    def _import_ground_truth(self, ground_truth_filename, groundtruth):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        
        if isinstance(groundtruth,dict):
          groundtruth = [groundtruth]

        # Read ground truth data.
        activity_index, cidx = {}, 0
        activity_fobj = open(self.activity_index_file, 'r')
        activities = activity_fobj.read().split('\n')[:-1]
        activity_fobj.close() 
        for a in activities:
            activity_index[int(a.split()[0])] = a.split()[1]

        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []

        for g in groundtruth:
          for videoid, v in g.items():
            if len(v['segments']) == 0: continue
            for t_s, t_e, lbl in zip(v['segments'][:,0].tolist(),v['segments'][:,1].tolist(),v['labels'].tolist()):
                video_lst.append(videoid)
                t_start_lst.append(t_s * v['length'].cpu().data)
                t_end_lst.append(t_e * v['length'].cpu().data)
                label_lst.append(activity_index[int(lbl)])

        groundtruth = pd.DataFrame({'video-id': video_lst,'t-start': t_start_lst, 't-end': t_end_lst, 'label': label_lst})
        return groundtruth, activity_index

    @torch.no_grad()
    def _import_prediction(self, prediction):
        """Reads prediction dict, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        if isinstance(prediction,dict):
          prediction = [prediction]
      

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for p in prediction:
          for videoid, v in p.items():
            for t_s, t_e, lbl, scr in zip(v['segment'][:,0].tolist(),v['segment'][:,1].tolist(),v['label'].tolist(),v['score'].tolist()):
                video_lst.append(videoid)
                t_start_lst.append(t_s)
                t_end_lst.append(t_e)
                label_lst.append(self.activity_index[int(lbl)])
                score_lst.append(scr)
        prediction_df = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction_df

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label. 
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            return pd.DataFrame()

    def _get_ground_truth_with_label(self, ground_truth_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label. 
        """
        try:
            return ground_truth_by_label.get_group(cidx).reset_index(drop=True)
        except:
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(list(self.activity_index.items()))))

        for cidx, activity in self.activity_index.items():
            gt_idx = self.groundtruth['label'] == activity
            pred_idx = self.prediction['label'] == activity
            if gt_idx.empty or pred_idx.empty: continue
            ap[:,cidx] = compute_average_precision_detection(self.groundtruth.loc[gt_idx].reset_index(drop=True),self.prediction.loc[pred_idx].reset_index(drop=True), tiou_thresholds=self.tiou_thresholds)


        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()
        if self.verbose:
            print('[RESULTS] Performance on action detection task.', self.mAP)
            print('\tAverage-mAP: {}'.format(self.average_mAP))
        return self.average_mAP

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.1,0.5,5)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap
    if ground_truth.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1

    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)

        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    for tidx in range(len(tiou_thresholds)):
        this_tp = np.cumsum(tp[tidx,:]).astype(np.float)
        this_fp = np.cumsum(fp[tidx,:]).astype(np.float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)

    return ap
