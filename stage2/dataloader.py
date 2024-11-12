#%%
from config import config
import torch, glob
import torch.utils.data as data
from fmutils import fmutils as fmu
import decord as de
import numpy as np
from pathlib import Path

# decord.bridge.set_bridge('torch')
#%%
class GEN_DATA_LISTS():

    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def get_data(self, num_fold=1):
        data_dir = self.data_dir
        train_path = f'{data_dir}fold_{num_fold}/train/'
        test_path = f'{data_dir}fold_{num_fold}/test/'
        print(train_path)
        train_paths = fmu.get_all_files(train_path)
        test_paths = fmu.get_all_files(test_path)

        # experiment type
        if config['exp_typ'] == 'gvp': # gtcs vs pnes
            train_paths = [path for path in train_paths if 'baseline' not in path]
            test_paths = [path for path in test_paths if 'baseline' not in path]
        elif config['exp_typ'] == 'bvg': # baseline vs gtcs
            train_paths = [path for path in train_paths if 'a_' in path or 'IMAGE' in path]
            test_paths = [path for path in test_paths if 'a_' in path or 'IMAGE' in path]
        elif config['exp_typ'] == 'bvp': # baseline vs pnes
            train_paths = [path for path in train_paths if 'b_' in path or 'PNES' in path]
            test_paths = [path for path in test_paths if 'b_' in path or 'PNES' in path]

        return train_paths, test_paths
        

#%%
class MME_Loader(data.Dataset):
    def __init__(self, data_path, config=config, val_feat='all',
                 validation=False, t_val=False, num_segments=6):
        # get labels paths
        self.data_paths = data_path
        self.config = config
        self.val_feat = val_feat
        self.validation = validation
        self.t_val = t_val 
        self.num_segments = num_segments

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_sample = {}
        # LODAING INPUT DATA
        filename = Path(self.data_paths[index]).stem
        
        # load whole data
        all_feats_lbls = np.load(self.data_paths[index])
        ecg_feats = all_feats_lbls['ecg'] # shape -> Samples x 256
        pose_feats = all_feats_lbls['pose']
        flow_feats = all_feats_lbls['flow']
        lbls = all_feats_lbls['lbls']# - 1 # shape -> Samples [1 or 2] -1 so that it becomes [0 or 1]
        # 0 is GTCS and 1 is PNES
        # print(f"before feats shape:, {ecg_feats.shape}, {len(ecg_feats)}")
        assert len(ecg_feats) == len(pose_feats) == len(flow_feats), f'Length Mismatch {filename}'
        if not self.validation:
            # randomly select 6 consequitive segment/smaples from the feats i.e., shape -> 6 x 256 
            if len(ecg_feats) > self.num_segments:
                start = np.random.randint(0, len(ecg_feats) - self.num_segments)
            else:
                start = 0  # esle start from 0
                
            ecg_feats_seg = ecg_feats[start:start+self.num_segments]
            pose_feats_seg = pose_feats[start:start+self.num_segments]
            flow_feats_seg = flow_feats[start:start+self.num_segments]
            lbls = lbls[start:start+6]
        if self.validation:
            # sequentially select 6 consequitive segment/smaples from the feats i.e., shape -> 6 x 256
            # cover full recording legth by stacking the 6x256 segments till end of the sample
            ecg_feats_seg = np.vstack([ecg_feats[i:i+self.num_segments] for i in range(0, len(ecg_feats), self.num_segments)])
            pose_feats_seg = np.vstack([pose_feats[i:i+self.num_segments] for i in range(0, len(pose_feats), self.num_segments)])
            flow_feats_seg = np.vstack([flow_feats[i:i+self.num_segments] for i in range(0, len(flow_feats), self.num_segments)])

        # print(f'filename: {filename}, ecg_feats_seg: {ecg_feats_seg.shape}, pose_feats_seg: {pose_feats_seg.shape}, flow_feats_seg: {flow_feats_seg.shape}, lbls: {lbls.shape}')

        # for each 6x256 sample the label will be either 0 or 1, so put a check on it
        assert len(np.unique(lbls)) == 1, f'Multiple classes in the same sample {filename}, {np.unique(lbls)}'

        if not self.validation:
            if ecg_feats_seg.shape[0] != self.num_segments:
                # print(f'Less than 6 samples, {filename}, {feats_seg.shape}')
                # print('-------------------')
                # fix the smaple by appending zeros
                ecg_feats_seg = np.vstack([ecg_feats_seg, np.zeros((self.num_segments - ecg_feats_seg.shape[0], 256))])
                pose_feats_seg = np.vstack([pose_feats_seg, np.zeros((self.num_segments - pose_feats_seg.shape[0], 256))])
                flow_feats_seg = np.vstack([flow_feats_seg, np.zeros((self.num_segments - flow_feats_seg.shape[0], 256))])
        # get one label
        lbl = lbls[0]
        #new
        if self.config['exp_typ'] == 'gvp': # gtcs vs pnes
            if filename[0] == 'a' or 'IMAGE' in filename:
                lbl = 0
            elif filename[0] == 'b' or 'PNES' in filename:
                lbl = 1
        if self.config['exp_typ'] == 'bvg': # baseline vs gtcs
            if 'baseline' in filename:
                lbl = 0
            elif filename[0] == 'a' or 'IMAGE' in filename:
                lbl = 1
        if self.config['exp_typ'] == 'bvp':
            if 'baseline' in filename:
                lbl = 0
            elif filename[0] == 'b' or 'PNES' in filename:
                lbl = 1

        if not self.validation:
            # create a 1x3 1-hot vecoter to randomly select one or two or three feats
            # e.g., [0,0,1] or [1,1,0] or [1,0,0] or [0,1,0] so on
            selector = np.random.choice([0, 1], 3)
            # check if all are zeros, then make one of them 1
            if np.sum(selector) == 0:
                selector[np.random.randint(0, 3)] = 1
        if self.t_val:
            if self.val_feat == 'all':
                selector = [1, 1, 1]
            elif self.val_feat == 'pose':
                selector = [0, 1, 0]
            elif self.val_feat == 'flow':
                selector = [0, 0, 1]
            elif self.val_feat == 'ecg':
                selector = [1, 0, 0]
        if self.validation:
            if self.val_feat == 'all':
                selector = [1, 1, 1]
            elif self.val_feat == 'pose':
                selector = [0, 1, 0]
            elif self.val_feat == 'flow':
                selector = [0, 0, 1]
            elif self.val_feat == 'ecg':
                selector = [1, 0, 0]
            elif self.val_feat == 'pose_flow': #'pose_flow', 'ecg_flow', 'ecg_pose'
                selector = [0, 1, 1]
            elif self.val_feat == 'ecg_pose':
                selector = [1, 1, 0]
            elif self.val_feat == 'ecg_flow':
                selector = [1, 0, 1]
        # print(f'filename: {filename}, selector: {selector}')
        data_sample['ecg_feats'] = ecg_feats_seg * selector[0]
        data_sample['pose_feats'] = pose_feats_seg * selector[1]
        data_sample['flow_feats'] = flow_feats_seg * selector[2]
        data_sample['lbl'] = lbl
        data_sample['filename'] = filename

        # print all info here compeltely even if it's printed above.
        # print(30*'+')
        # print(f'filename: {filename}')
        # print(f"before feats shape:, {ecg_feats.shape}, {len(ecg_feats)}")
        # print(f'start: {start} to {start+6}')
        # print(f'op feats shape:, {feats_seg.shape}, {len(feats_seg)}')
        # print(f'lbls shape:, {lbls.shape}, {len(lbls)}')
        # print(30*'+')
        return data_sample

#%%

class MME_Loader2(data.Dataset):
    def __init__(self, data_path, config=config, val_feat='all', validation=False, t_val=False, num_segments=6):
        # get labels paths
        self.data_paths = data_path
        self.config = config
        self.val_feat = val_feat
        self.validation = validation
        self.t_val = t_val 
        self.num_segments = num_segments

    def __len__(self):
        return len(self.data_paths)

    def _pad_or_trim(self, feats, num_segments, segment_length=256):
        if len(feats) >= num_segments:
            start = np.random.randint(0, len(feats) - num_segments + 1)
            return feats[start:start+num_segments]
        else:
            return np.pad(feats, ((0, num_segments - len(feats)), (0, 0)), 'constant')

    def _select_segments(self, feats, segment_length=256):
        return np.vstack([feats[i:i+self.num_segments] for i in range(0, len(feats), self.num_segments)])

    def __getitem__(self, index):
        data_sample = {}
        # Loading input data
        filename = Path(self.data_paths[index]).stem
        all_feats_lbls = np.load(self.data_paths[index])
        ecg_feats = all_feats_lbls['ecg']
        pose_feats = all_feats_lbls['pose']
        flow_feats = all_feats_lbls['flow']
        lbls = all_feats_lbls['lbls']
        
        assert len(ecg_feats) == len(pose_feats) == len(flow_feats), f'Length Mismatch {filename}'
        
        if not self.validation:
            ecg_feats_seg = self._pad_or_trim(ecg_feats, self.num_segments)
            pose_feats_seg = self._pad_or_trim(pose_feats, self.num_segments)
            flow_feats_seg = self._pad_or_trim(flow_feats, self.num_segments)
            lbls = lbls[:self.num_segments]
        else:
            ecg_feats_seg = self._select_segments(ecg_feats)
            pose_feats_seg = self._select_segments(pose_feats)
            flow_feats_seg = self._select_segments(flow_feats)

        assert len(np.unique(lbls)) == 1, f'Multiple classes in the same sample {filename}, {np.unique(lbls)}'
        
        lbl = lbls[0]
        if self.config['exp_typ'] == 'gvp': # gtcs vs pnes
            lbl = 0 if filename[0] == 'a' else 1
        elif self.config['exp_typ'] == 'bvg': # baseline vs gtcs
            lbl = 0 if 'baseline' in filename else 1
        elif self.config['exp_typ'] == 'bvp':
            lbl = 0 if 'baseline' in filename else 1

        if not self.validation:
            selector = np.random.choice([0, 1], 3)
            if np.sum(selector) == 0:
                selector[np.random.randint(0, 3)] = 1
        else:
            selector = self._get_selector(self.val_feat)

        data_sample['ecg_feats'] = ecg_feats_seg * selector[0]
        data_sample['pose_feats'] = pose_feats_seg * selector[1]
        data_sample['flow_feats'] = flow_feats_seg * selector[2]
        data_sample['lbl'] = lbl
        data_sample['filename'] = filename

        return data_sample

    def _get_selector(self, val_feat):
        if val_feat == 'all':
            return [1, 1, 1]
        elif val_feat == 'pose':
            return [0, 1, 0]
        elif val_feat == 'flow':
            return [0, 0, 1]
        elif val_feat == 'ecg':
            return [1, 0, 0]
        elif val_feat == 'pose_flow':
            return [0, 1, 1]
        elif val_feat == 'ecg_pose':
            return [1, 1, 0]
        elif val_feat == 'ecg_flow':
            return [1, 0, 1]
        else:
            raise ValueError(f"Unknown val_feat: {val_feat}")
        

#%%


def get_exteranl_data(exp_typ, base_dir):

    if exp_typ == 'gvp':
        ext_data = {
                'flow_paths': sorted(glob.glob(f'{base_dir}/flow/*_flow.mp4'), key=fmu.numericalSort),
                'ecg_paths': sorted(glob.glob(f'{base_dir}/ecg_arr/*.npy'), key=fmu.numericalSort),
                'pose_paths': sorted(glob.glob(f'{base_dir}/pose/*/body_coco.npy'), key=fmu.numericalSort),
                'flow_lbls': sorted(glob.glob(f'{base_dir}/labels/*_vid_lbl.npy'), key=fmu.numericalSort),
                'ecg_lbls': sorted(glob.glob(f'{base_dir}/labels/*_ecg_lbl.npy'), key=fmu.numericalSort),
            }
    else:
        if exp_typ == 'bvg':
            e_split = 'IMAGE' 

        if exp_typ == 'bvp':
            e_split = 'PNES'
            
        ext_data = {
                'flow_paths': sorted(glob.glob(f'{base_dir}/flow/{e_split}*_flow.mp4'), key=fmu.numericalSort),
                'ecg_paths': sorted(glob.glob(f'{base_dir}/ecg_arr/{e_split}*.npy'), key=fmu.numericalSort),
                'pose_paths': sorted(glob.glob(f'{base_dir}/pose/{e_split}*/body_coco.npy'), key=fmu.numericalSort),
                'flow_lbls': sorted(glob.glob(f'{base_dir}/labels/{e_split}*_vid_lbl.npy'), key=fmu.numericalSort),
                'ecg_lbls': sorted(glob.glob(f'{base_dir}/labels/{e_split}*_ecg_lbl.npy'), key=fmu.numericalSort),
            }

    return ext_data