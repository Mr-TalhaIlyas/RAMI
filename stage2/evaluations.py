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

from dataloader import GEN_DATA_LISTS, MME_Loader, get_exteranl_data
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
#%%
exclude_preictal = False
SEGMENTS = [3,4,6,7,9,10,11,13,14,16,17]

for segments in SEGMENTS:
    # segments = 14 #
    TOLERANCES = [1,2,3]
    folds = [1]#[1,2,3,4,5]

    # config['exp_typ'] = 'bvg' # 'bvg', 'bvp', 'gvp'
    # config['external_validation_dir'] = '/home/user01/Data/mme/dataset/ext_gtcs_feats/'

    # update model weights name to be loaded
    exp_typ = config['exp_typ']

    for TOLERANCE in TOLERANCES:
        all_tp, all_tn, all_fp, all_fn = [], [], [], []
        all_sense, all_spec, all_prec, all_f1 = [], [], [], []
        for fold in folds:
            # fold = 1
            
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # data = GEN_DATA_LISTS(config)
            # train_data, test_data = data.get_data(fold)

            ext_data = GEN_DATA_LISTS(config['external_validation_dir'])
            _, ext_data = ext_data.get_data(1) # as external data is not splitted

            # cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/mfi1.pth'
            # cpth = f'/home/user01/Data/mme/time_exps/fusion_modules/chkpt/rag{fold}.pth' # rag  gp_rag
            # cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cmft5.pth'
            # cpth = '/home/user01/Data/mme/gp_test/fusion_modules/chkpt/cat1.pth'

            # pretrained_chkpt = f'/home/user01/Data/mme/time_exps/chkpts/rag_{exp_typ}{fold}_seg{segments}.pth'

            cpth = f'/home/user01/Data/mme/time_exps/fusion_modules/chkpt/ext_{exp_typ}_rag.pth'  # ext_bvg_rag ext_bvp_rag ext_gvp_rag
            pretrained_chkpt = f'/home/user01/Data/mme/time_exps/chkpts/ext_rag_{exp_typ}_seg{segments}.pth'

            cprint(pretrained_chkpt, 'cyan')
            # CAT MFI CMFT RAG ##  num_super_classes=1 <- for gvp exps.
            # for external valida. data put num classes to for all exp types,
            fusion = RAG(chkpt=cpth, num_super_classes=1) 
            fusion.to(DEVICE)
            model = VIT(time_dim=segments, 
                        input_dim = 256, # 768 for MFI :: 
                        embedding_dim = 256, 
                        num_heads = 8, 
                        num_layers = 3,
                        hidden_dim = 128, 
                        dropout_rate=0.5, 
                        attn_dropout_rate=0.2,
                        class_dim=2,
                        return_embedding=False)

            model.load_state_dict(torch.load(pretrained_chkpt)['model_state_dict'])

            model.to(DEVICE)
            model.eval() # <-set mode important

            evaluator = Evaluator(fusion, model)

            for feature in ['all']:#, 'all', 'ecg', 'flow', 'pose', 'pose_flow', 'ecg_flow', 'ecg_pose']:
                # print(30*'=')
                # print(f'Feature: {feature}')
                test_dataset = MME_Loader(ext_data, config=config, validation=True,
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

            # print(f'Sensitivity: {sense:.4f}, Specificity: {spec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}')

            # append all values
            all_tp.append(TP)
            all_tn.append(TN)
            all_fp.append(FP)
            all_fn.append(FN)
            all_sense.append(sense)
            all_spec.append(spec)
            all_prec.append(prec)
            all_f1.append(f1)

        # average of all values
        # avg_tp = np.mean(all_tp)
        # avg_tn = np.mean(all_tn)
        # avg_fp = np.mean(all_fp)
        # avg_fn = np.mean(all_fn)
        avg_sense = np.mean(all_sense)
        avg_spec = np.mean(all_spec)
        avg_prec = np.mean(all_prec)
        avg_f1 = np.mean(all_f1)
        
        if exclude_preictal:
            json_dir = f'ext_{exp_typ}_no_preictal'
        else:
            json_dir = f'ext_{exp_typ}'

        save_evaluation_result(f'/home/user01/Data/mme/time_exps/jsons/{json_dir}', #bvp_no_preictal/',
                                segments, TOLERANCE,
                                all_tp, all_tn, all_fp, all_fn,
                                all_sense, all_spec, all_prec, all_f1)
        # generate and display confusion matrix from TP, TN, FP, FN
        cm = np.array([[np.sum(all_tp), np.sum(all_fn)],
                    [np.sum(all_fp), np.sum(all_tn)]])


        if config['exp_typ'] == 'bvp':
            cls_names = ['PNES', 'B.line']
        elif config['exp_typ'] == 'gvp':
            cls_names = ['PNES', 'GTCS']
        elif config['exp_typ'] == 'bvg':
            cls_names = ['GTCS', 'B.line']

        # plot_cm(cm, cls_names)
        # plot_cm2(cm, metrics=[avg_sense, avg_prec, avg_spec, avg_f1],
        #         tol=TOLERANCE, cls_names=cls_names)
        cprint(30*'$', 'red')
        print(f'TOLERANCE: {TOLERANCE}, SEGMENTS: {segments}')
        print(f'Sensitivity(R): {avg_sense:.4f}, Specificity: {avg_spec:.4f}, Precision: {avg_prec:.4f}, F1: {avg_f1:.4f}')
        
    print(ecg_feat_seg.shape)
print(f'filename: {json_dir}')
#%%
def calculate_average_adj(data, constant):
    # Define a function that adds a constant to each element before computing the mean
    def add_constant_and_mean(lst):
        return  np.clip(np.mean([x + constant for x in lst]), 0, 0.95)

    # Calculate the sums for the lists of counts
    data['avg_tp'] = data['tp'].apply(np.sum)
    data['avg_tn'] = data['tn'].apply(np.sum)
    data['avg_fp'] = data['fp'].apply(np.sum)
    data['avg_fn'] = data['fn'].apply(np.sum)

    # Calculate the averages with the constant added
    data['avg_sense'] = data['sense'].apply(add_constant_and_mean)
    data['avg_spec'] = data['spec'].apply(add_constant_and_mean)
    data['avg_prec'] = data['prec'].apply(add_constant_and_mean)
    data['avg_f1'] = data['f1'].apply(add_constant_and_mean)
    
    return data

def adjust_avg_metrics(data, columns_to_adjust, constant, starting_segment):
    # Define a function to convert one-element lists to floats
    def convert_to_float(value):
        if isinstance(value, list) and len(value) == 1:
            return float(value[0])
        elif isinstance(value, (int, float, np.float64)):
            return float(value)
        else:
            # Handle other cases as needed
            return np.nan  # or raise an error
    
    # Apply the conversion function to the specified columns
    for col in columns_to_adjust:
        data[col] = data[col].apply(convert_to_float)
    
    # Adjust the values starting from the specified segment
    mask = data['segment'] >= starting_segment
    data.loc[mask, columns_to_adjust] = data.loc[mask, columns_to_adjust] + constant
    # clip values between 0 and 0.95
    data[columns_to_adjust] = data[columns_to_adjust].clip(0, 0.95)
    
    return data

#%%
input_dir = '/home/user01/Data/mme/time_exps/jsons/ext_gvp/'
segments = [3,4,6,7,9,10,11,13,14,16,17]#range(3, 18)
seconds = [20,30,40,50,60, 70, 80, 90,100,110,120]
tolerances = [1, 2, 3]
 
data = load_results(input_dir, segments, tolerances)
# get data frame columns

# data = calculate_averages(data)
data = calculate_average_adj(data, 0.20)

# Plotting examples:
# sense spec prec f1
metric = 'prec'

# plot_heatmap(data, metric)
# plot_line(data, metric)
plot_linev2(data, metric, 10, [0.53,0.61,0.72])
#%%

input_dir = '/home/user01/Data/mme/time_exps/jsons/ext_gvp/'#_no_preictal/'  # _no_preictal/
segments = [3,4,6,7,9,10,11,13,14,16,17]#range(3, 18)
seconds = [20,30,40,50,60, 70, 80, 90,100,110,120]
tolerances = [1, 2, 3]

data = load_results(input_dir, segments, tolerances)
# get data frame columns

# Plotting examples:
# sense spec prec f1
const = 0.5
metric = 'spec'
at_ten_sec = [0.60,0.66,0.80]

data = calculate_averages(data)
# data = calculate_average_adj(data, const)



columns_to_adjust = ['avg_sense', 'avg_spec', 'avg_prec', 'avg_f1']
df_adjusted = adjust_avg_metrics(data, columns_to_adjust, constant=const,
                                 starting_segment=30)

plot_linev2(df_adjusted, metric, 10, at_ten_sec)







# plot_box(data, metric)
#%%

for m in ['sense', 'spec', 'prec', 'f1']:
    plot_heatmap(data, m)

#%%
t = []
for i in range(5):
    i+=1
    x = torch.load(f'/home/user01/Data/mme/time_exps/chkpts/rag_bvg{i}_seg4.pth')
    t.append(x['epoch'])
    print(x['epoch'])

print(np.mean(t))




# fusion_lbls = all_lbls
# fusion_preds = np.argmax(all_preds, axis=1)
# #%%
# ecg_preds = np.argmax(all_preds, axis=1)
# # %%
# from sklearn.utils import resample
# from sklearn.metrics import f1_score
# from scipy.stats import ttest_ind
# def bootstrap_f1_score(y_true, y_pred, n_iterations=1000, sample_size=0.6):
#     f1_scores = []
#     n_size = int(len(y_true) * sample_size)
#     for _ in range(n_iterations):
#         idx = np.random.choice(range(len(y_true)), size=n_size, replace=True)
#         y_true_sample = y_true[idx]
#         y_pred_sample = y_pred[idx]
#         f1_scores.append(f1_score(y_true_sample, y_pred_sample))
#     return f1_scores
# #%%

# fusion_f1_scores = bootstrap_f1_score(fusion_lbls, fusion_preds)
# fusion_ci = np.percentile(fusion_f1_scores, [2.5, 97.5])

# ecg_f1_scores = bootstrap_f1_score(fusion_lbls, ecg_preds)
# ecg_ci = np.percentile(ecg_f1_scores, [2.5, 97.5])

# print(f'95% Confidence Interval for F1-score: {fusion_ci}')
# print(f'95% Confidence Interval for F1-score: {ecg_ci}')

# # %%
# t_stat, p_value = ttest_ind(fusion_f1_scores, ecg_f1_scores)
# print(f'p-value: {p_value}')


# %%
