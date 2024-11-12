#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:15:28 2024

@author: user01
"""

#%%
import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/mme/time_exps/')

from config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available")
print(f"Number of CPUs: {os.cpu_count()}")
memory = psutil.virtual_memory()
print(f"Total Memory: {memory.total / (1024**3):.2f} GB") 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from fmutils import fmutils as fmu

import cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from termcolor import cprint
from tqdm import tqdm

from dataloader import GEN_DATA_LISTS, MME_Loader, MME_Loader2
# from data.utils import collate, values_fromreport
from vit import VIT
from model import EWT
from lr_scheduler import LR_Scheduler
from tools import save_chkpt, values_fromreport

from fusion_modules.rag import RAG
from fusion_modules.mfi import MFI
from fusion_modules.cmft import CMFT
from fusion_modules.cat import CAT

from training import Trainer, Evaluator

from sklearn.metrics import confusion_matrix, classification_report

from tsaug.visualization import plot
from IPython.display import HTML
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tools import calculate_confusion_matrix_gtcs_pnes, plot_cm, plot_cm2, evaluate_gtcs_pnes
from global_viz import (save_evaluation_result, load_results, plot_heatmap,
                        plot_line, plot_box, calculate_averages,plot_linev2,
                        segment_to_seconds)

config['data_dir'] = '/home/user01/Data/mme/dataset/ext_gtcs_feats/'
#%%
exclude_preictal = False
SEGMENTS = [3,4,6,7,9,10,11,13,14,16,17]

# for segments in SEGMENTS:
#     # segments = 14 #
#     TOLERANCES = [1,2,3]
#     folds = [1,2,3,4,5]

    # update model weights name to be loaded
exp_typ = config['exp_typ']

    # for TOLERANCE in TOLERANCES:
    #     all_tp, all_tn, all_fp, all_fn = [], [], [], []
    #     all_sense, all_spec, all_prec, all_f1 = [], [], [], []
    #     for fold in folds:
    #         # fold = 1
fold = 1    
segments = config['num_segments']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = GEN_DATA_LISTS(config)

train_data, test_data = data.get_data(1) # fold

# cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/mfi1.pth'
cpth = f'/home/user01/Data/mme/time_exps/fusion_modules/chkpt/ext_bvg_rag.pth' # rag  gp_rag
# cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cmft5.pth'
# cpth = '/home/user01/Data/mme/gp_test/fusion_modules/chkpt/cat1.pth'

pretrained_chkpt = f'/home/user01/Data/mme/time_exps/chkpts/ext_rag_bvg_seg6.pth'
cprint(pretrained_chkpt, 'cyan')
fusion = RAG(chkpt=cpth, num_super_classes=1) # CAT MFI CMFT RAG ##  num_super_classes=1 <- for gvp exps.
fusion.to(DEVICE)

dim_l_hid_drp = [256, 3, 128, 0.2]

model = VIT(time_dim=config['num_segments'], 
            input_dim = 256, # 768 for MFI :: 
            embedding_dim = dim_l_hid_drp[0], 
            num_heads = 8, 
            num_layers = dim_l_hid_drp[1],
            hidden_dim = dim_l_hid_drp[2], 
            dropout_rate=0.5, 
            attn_dropout_rate=dim_l_hid_drp[3],
            class_dim=2,
            return_embedding=False)

model.load_state_dict(torch.load(pretrained_chkpt)['model_state_dict'])

model.to(DEVICE)
model.eval() # <-set mode important
#%
evaluator = Evaluator(fusion, model)

for feature in ['all']:#, 'all', 'ecg', 'flow', 'pose', 'pose_flow', 'ecg_flow', 'ecg_pose']:
    # print(30*'=')
    # print(f'Feature: {feature}')
    test_dataset = MME_Loader(test_data, config=config, validation=True,
                            val_feat=feature, num_segments=segments)
    test_loader = DataLoader(test_dataset,
                            batch_size=1, shuffle=False,
                            num_workers=config['num_workers'], drop_last=False,
                            collate_fn=None, pin_memory=config['pin_memory'],
                            prefetch_factor=1, persistent_workers=True,
                            # sampler=BiasedSampler(test_dataset)
                            )

    total_avg_acc, test_acc = [], []
    all_preds, all_lbls, files = [], [], []
    # for _ in range(10): 
    for step, test_batch in enumerate(test_loader):
        if exclude_preictal:
            # remove last 9 (60sec) segments from recording having baseline segment file.
            if 'baseline' in test_batch['filename'][0]:
                test_batch['ecg_feats'] = test_batch['ecg_feats'][:,:-9,:]
                test_batch['flow_feats'] = test_batch['flow_feats'][:,:-9,:]
                test_batch['pose_feats'] = test_batch['pose_feats'][:,:-9,:]

        for i in range(0, test_batch['ecg_feats'].shape[1], segments):
            
            ecg_feat_seg = test_batch['ecg_feats'][:,i:i+segments,:]
            flow_feat_seg = test_batch['flow_feats'][:,i:i+segments,:]
            pose_feat_seg = test_batch['pose_feats'][:,i:i+segments,:]
            # print(ecg_feat_seg.shape)
            if ecg_feat_seg.shape[1] != segments:
                # print('skipping')
                # continue
                # FIX the smaple by getting last 6 segments
                # ecg_feat_seg = test_batch['ecg_feats'][:, -segments:,:]
                # flow_feat_seg = test_batch['flow_feats'][:, -segments:,:]
                # pose_feat_seg = test_batch['pose_feats'][:, -segments:,:]

                # FIX the smaple by appending zeros
                ecg_feat_seg = ecg_feat_seg.squeeze(0)
                flow_feat_seg = flow_feat_seg.squeeze(0)
                pose_feat_seg = pose_feat_seg.squeeze(0)
                ecg_feat_seg = np.vstack([ecg_feat_seg, np.zeros((segments - ecg_feat_seg.shape[0], 256))])
                flow_feat_seg = np.vstack([flow_feat_seg, np.zeros((segments - flow_feat_seg.shape[0], 256))])
                pose_feat_seg = np.vstack([pose_feat_seg, np.zeros((segments - pose_feat_seg.shape[0], 256))])
                ecg_feat_seg = torch.from_numpy(ecg_feat_seg[np.newaxis, ...])
                flow_feat_seg = torch.from_numpy(flow_feat_seg[np.newaxis, ...])
                pose_feat_seg = torch.from_numpy(pose_feat_seg[np.newaxis, ...])

                # print('Padded Segments: ', ecg_feat_seg.shape)

            # create new test_batch using segments
            test_batch_seg = {'ecg_feats': ecg_feat_seg,
                            'flow_feats': flow_feat_seg,
                            'pose_feats': pose_feat_seg,
                            'lbl': test_batch['lbl']}

            acc, preds, lbl_batch = evaluator.eval_step(test_batch_seg)
            test_acc.append(acc)
            all_preds.append(preds)
            all_lbls.append(lbl_batch)
            files.append(test_batch['filename'][0])
            # break

    # print(f'=> Average Acc: {np.nanmean(test_acc):.4f}')
    total_avg_acc.append(np.nanmean(test_acc))
    current_acc = np.nanmax(total_avg_acc)

    all_preds = np.asarray(all_preds).reshape(-1, 2)
    all_lbls = np.asarray(all_lbls).reshape(-1,)

    matrix = confusion_matrix(all_lbls, np.argmax(all_preds, axis=1), normalize='true')
    report = classification_report(all_lbls, np.argmax(all_preds, axis=1),
                                    output_dict=True,
                                    zero_division=0)
    p, r, f1 = values_fromreport(report)
    # cprint(f'Class Accuracies:: {matrix.diagonal()/matrix.sum(axis=1)}', 'blue')
    # cprint(f'Average Acc: {np.mean(matrix.diagonal()/matrix.sum(axis=1)):.4f}', 'green')
    # cprint(f'Class-wise Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}', 'light_magenta')

TOLERANCE = 1
op = calculate_confusion_matrix_gtcs_pnes(all_lbls, np.argmax(all_preds, axis=1),
                                        files, tolerance=TOLERANCE)
# op = evaluate_gtcs_pnes(all_lbls, np.argmax(all_preds, axis=1), files,
#                         threshold=TOLERANCE, ratio_thresh=0.6)

TP = op['TP']
TN = op['TN']
FP = op['FP']
FN = op['FN']

# fix following for zerodivision error
# Recall
sense = TP / (TP + FN) if TP + FN != 0 else 0
spec = TN / (TN + FP) if TN + FP != 0 else 0
prec = TP / (TP + FP) if TP + FP != 0 else 0
f1 = 2 * (prec * sense) / (prec + sense) if prec + sense != 0 else 0

print(f'Sensitivity: {sense:.4f}, Specificity: {spec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}')

# append all values
# all_tp.append(TP)
# all_tn.append(TN)
# all_fp.append(FP)
# all_fn.append(FN)
# all_sense.append(sense)
# all_spec.append(spec)
# all_prec.append(prec)
# all_f1.append(f1)