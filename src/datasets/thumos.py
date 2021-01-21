import torch
from torch.utils.data import Dataset
import torchvision
import torch.nn as nn

import random
from pathlib import Path
import os,sys
import math
import numpy as np
import json
import datasets.thumos_utils as data_utils

class ThumosDetection(Dataset):
    """
       thumos dataset
    """
    def __init__(self, vid_data_file, num_classes, actions_dict, features_root, sample_rate,split,out_size):
        self.vid_data_file = vid_data_file
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.features_root = features_root
        self.sample_rate = sample_rate
        self.split = split
        self.out_size = out_size

        self.ids = self.read_data(self.vid_data_file,self.split)
        self.prepare = ThumosVideoData(self.sample_rate,self.actions_dict, self.split, self.out_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        vid = self.ids[idx]
        target = {}
        target = {'video_id': vid, 'annotations': target}
        
        target['annotations']['feature_path'] = self.features_root + "/" + vid + '.npy'
        with open(self.vid_data_file,'r') as f:
            data = json.load(f)     
        target['annotations']['actions'] = data[vid]['actions']
        target['annotations']['duration'] = data[vid]['duration']
        inp, target = self.prepare(target)
        return inp, target


    def read_data(self, vid_data_file,split):
        with open(vid_data_file,'r') as f:
            data = json.load(f)     

        list_of_examples = [] 
        for k,v in data.items():
            if v['subset'] != split: continue
            if not os.path.exists(os.path.join(self.features_root,k+'.npy')): continue
            if os.path.exists(os.path.join(self.features_root,k+'.npy')):
                k_shape = np.load(os.path.join(self.features_root,k+'.npy')).shape[0]

            list_of_examples.append((k,k_shape))

        list_of_examples = sorted(list_of_examples, key=lambda x:x[1]) 
        list_of_examples = [e[0] for e in list_of_examples]  
        
        return list_of_examples

class ThumosVideoData(object):
    def __init__(self,sample_rate,actions_dict,mode,out_size):
        self.sample_rate = sample_rate
        self.actions_dict = actions_dict
        self.mode = mode
        self.out_size = out_size
        pass

    def __call__(self, target):
        vid =  target['video_id']
        video_id = data_utils.getVideoId(vid)
        
        anno = target['annotations']

        feature = np.load(anno['feature_path'])
              
        triplets = anno['actions']
        duration = anno['duration']
        set_targets = {}
        set_targets['video_id'] = torch.tensor(video_id)
        set_targets['segments'] = []
        set_targets['labels'] = []
 
        for triplet in triplets:
            set_targets['segments'].append([triplet[1]/duration,triplet[2]/duration])
            set_targets['labels'].append(triplet[0])
       
        feature = torch.tensor(feature,dtype=torch.float32).squeeze_(1).squeeze_(1).permute(1,0)

        out_size = self.out_size
        offset = 32


        if feature.size(1) <= out_size:
            upsampled_size = feature.size(1) * 4
            initial_size = feature.size(1)
            tile_idx = np.zeros(initial_size)
            for i in range(initial_size):
                for j in range(math.floor(upsampled_size/initial_size)):
                    tile_idx[i] += 1
            for i in range(initial_size):
                tile_idx[i] += 1
                if np.sum(tile_idx) == upsampled_size: break
            feature = feature.repeat_interleave(torch.LongTensor(tile_idx),dim=1)

            if self.mode == 'training':
                fidx = sorted(random.sample(list(range(0,feature.size(1))),initial_size))
                feature = feature[:,torch.tensor(fidx)].squeeze(0)
        
            elif self.mode in ['testing', 'validation']:
                fidx = list(np.linspace(0,feature.size(1)-1,initial_size))
                fidx = [int(idx) for idx in fidx]
                feature = feature[:,torch.tensor(fidx)].squeeze(0)

        elif feature.size(1) > out_size: 
            if self.mode == 'training':
                if len(list(range(0+offset,feature.size(1)-offset))) < out_size: 
                    initial_size = len(list(range(0+offset,feature.size(1)-offset)))
                    tile_idx = np.zeros(initial_size)
                    for i in range(initial_size):
                        for j in range(math.floor(out_size/feature.size(1))):
                            tile_idx[i] += 1
    
                    for i in range(initial_size):
                        tile_idx[i] += 1
                        if np.sum(tile_idx) == out_size: break

                    sample_idx = torch.tensor(list(range(0+offset,feature.size(1)-offset))).repeat_interleave(torch.LongTensor(tile_idx))
                    fidx = sorted(sample_idx.tolist())
                else:
                    fidx = sorted(random.sample(list(range(0+offset,feature.size(1)-offset)),out_size))

                feature_sampled = []
                for idx in fidx:
                    feature_sampled.append(torch.mean(feature[:,torch.tensor(list(range(idx-offset,idx+offset)))],dim=1).unsqueeze(1))
                feature = torch.cat(feature_sampled,dim=1)
    
            elif self.mode in ['testing', 'validation']:
                fidx = list(np.linspace(0+offset,feature.size(1)-offset,out_size))
                fidx = [int(idx) for idx in fidx]
                feature_sampled = []
                for idx in fidx:
                    feature_sampled.append(torch.mean(feature[:,torch.tensor(list(range(idx-offset,idx+offset)))],dim=1).unsqueeze(1))
                feature = torch.cat(feature_sampled,dim=1)


        set_targets['num_feat'] = torch.tensor(feature.size(1),dtype = torch.float32)
        set_targets['segments'] = torch.tensor(set_targets['segments'],dtype=torch.float32)
        set_targets['labels'] = torch.tensor(set_targets['labels'],dtype=torch.int64)
        set_targets['length'] = torch.tensor([duration/self.sample_rate],dtype=torch.float32)
        set_targets['sample_rate'] = torch.tensor([self.sample_rate],dtype=torch.float32)
        set_targets['fps'] = torch.tensor([float(30)/self.sample_rate],dtype=torch.float32)
        set_targets['num_classes'] = torch.tensor([len(list(self.actions_dict.keys()))],dtype=torch.float32)
        return feature, set_targets 


def build_thumos_detection(mode, args):
    root = Path(args.data_root)
    assert root.exists()

    features_path = os.path.join(args.data_root, args.features)
    vid_data_file = os.path.join(args.data_root, args.dataset+ "_action.json")
    # action category mappings
    mapping_file = os.path.join(args.data_root, "action_mapping.txt")
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    
    num_classes = len(actions_dict)
    assert num_classes == args.num_classes   # check if the number of classes are determined correctly

    dataset = ThumosDetection(vid_data_file, args.num_classes, actions_dict, features_path, args.sample_rate, mode, args.num_inputs)

    return dataset 
        
     
