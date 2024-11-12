#%%
import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/mme/time_exps/')

from config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    # from datetime import datetime
    # my_id = datetime.now().strftime("%Y%m%d%H%M")
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
            #    resume='allow', id=my_id, # this one introduces werid behaviour in the app
               config_include_keys=config.keys(), config=config)
    # print(f'WANDB config ID : {my_id}')


import torch
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

from tools import calculate_confusion_matrix_gtcs_pnes, plot_cm
#%%
fold_number = os.getenv('NUM_FOLD', config['num_fold'])
config['num_fold'] = int(fold_number)

segment_number = os.getenv('NUM_SEGMENTS', config['num_segments'])
config['num_segments'] = int(segment_number)

experiment_name = os.getenv('EXPERIMENT_NAME', config['experiment_name'])
config['experiment_name']=experiment_name

#%%
# num_classes = len(config['sub_classes'])
# sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config)

train_data, test_data = data.get_data(config['num_fold'])
#%%
train_dataset = MME_Loader(train_data, config=config, validation=False,
                            val_feat='all', num_segments=config['num_segments'])

train_loader = DataLoader(train_dataset,
                          batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=1, persistent_workers=True,
                        # sampler=BiasedSampler(train_dataset)
                        )

test_dataset = MME_Loader(test_data, config=config, validation=False, val_feat='all',
                          t_val=True, num_segments=config['num_segments'])
test_loader = DataLoader(test_dataset,
                         batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=1, persistent_workers=True,
                        # sampler=BiasedSampler(test_dataset)
                        )
#%%
# DataLoader Sanity Checks
batch = next(iter(train_loader))
print(f"Batch Shape: {batch['ecg_feats'].shape}, {batch['lbl'].shape}")
print(f"Batch Shape: {batch['flow_feats'].shape}, {batch['pose_feats'].shape}")
print(batch['filename'][0:2])
#%%
# cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/mfi1.pth'
cpth = f'/home/user01/Data/mme/time_exps/fusion_modules/chkpt/gp_rag{config["num_fold"]}.pth'
# cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cmft5.pth'
# cpth = '/home/user01/Data/mme/gp_test/fusion_modules/chkpt/cat1.pth'
fusion = RAG(chkpt=cpth, num_super_classes=1) # CAT MFI CMFT RAG ##  num_super_classes=1 <- for gvp exps.
fusion.to(DEVICE)

# dim_l_hid_drp = [512, 6, 256, 0.3]
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
model.to(DEVICE)


optimizer = torch.optim.AdamW([{'params': model.parameters(),
                            'lr':config['learning_rate']}],
                            weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

# accuracy = Accuracy(task="multiclass", num_classes=num_classes)

trainer = Trainer(fusion, model, optimizer)
evaluator = Evaluator(fusion, model)
#%%
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({"ECG Acc": 0, "Test ECG Acc": 0,
               "ecg_loss": 10, "learning_rate": 0}, step=0)
#%%

start_epoch = 0
epoch, best_acc = 0, 0
total_avg_acc = []

for epoch in range(start_epoch, config['epochs']):
    epoch 
    # pbar = tqdm(train_loader)
    model.train() # <-set mode important
    tacc, tloss = [], []
    for _ in range(10):
        for step, data_batch in enumerate(train_loader):

            scheduler(optimizer, step, epoch)
            loss_value, acc = trainer.training_step(data_batch)
            tloss.append(loss_value)
            tacc.append(acc)
            
            # pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - Train Acc {acc:.4f}')
        # break
    # print(f'=> Average loss: {np.nanmean(tloss):.4f}, Average Acc: {np.nanmean(tacc):.4f}')

    all_preds, all_lbls = [], []
    if (epoch + 1) % 1 == 0: # eval every 2 epoch
        model.eval() # <-set mode important
        test_acc = []#, preds, lbls = [], [], []
        # vbar = tqdm(test_loader)
        for _ in range(5): 
            for step, test_batch in enumerate(test_loader):
                acc, preds, lbl_batch = evaluator.eval_step(test_batch)
                test_acc.append(acc)
                # vbar.set_description(f'Validation - Acc {acc:.4f}')
                all_preds.append(preds)
                all_lbls.append(lbl_batch)
                # break

        # print(f'=> Average Acc: {np.nanmean(test_acc):.4f}')
        total_avg_acc.append(np.nanmean(test_acc))
        current_acc = np.nanmax(total_avg_acc)

        #####################
        all_preds = np.asarray(all_preds).reshape(-1, 2)
        all_lbls = np.asarray(all_lbls).reshape(-1,)
        
        matrix = confusion_matrix(all_lbls, np.argmax(all_preds, axis=1), normalize='true')
        report = classification_report(all_lbls, np.argmax(all_preds, axis=1),
                                       output_dict=True,
                                       zero_division=0)
        p, r, f1 = values_fromreport(report)
        # cprint(f'Class Accuracies:: {matrix.diagonal()/matrix.sum(axis=1)}', 'blue')
        # cprint(f'Class-wise Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}', 'light_magenta')
        #####################
        if current_acc > best_acc and epoch != 0:
            best_acc = current_acc
            best_p, best_r, best_f1 = p, r, f1
            best_chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(tloss),
                                    acc=current_acc, return_chkpt=True)
    
    if config['LOG_WANDB']:
        wandb.log({"ecg_loss": np.nanmean(tloss), 
                   "Test ECG Acc": np.nanmean(test_acc),"ECG Acc": np.nanmean(tacc),
                   "learning_rate": optimizer.param_groups[0]['lr']}, step=epoch+1)

print(f'Best Pre: {best_p:.4f}, Rec: {best_r:.4f}, F1: {best_f1:.4f}, Acc: {best_acc:.4f}')  

if config['LOG_WANDB']:
    # wandb.log({"Precision": p, "Recall": r, "F1": f1})
    wandb.run.finish()


#%%
model.load_state_dict(torch.load(best_chkpt)['model_state_dict'])
model.eval() # <-set mode important
evaluator = Evaluator(fusion, model)
for feature in ['all']:#, 'all', 'ecg', 'flow', 'pose', 'pose_flow', 'ecg_flow', 'ecg_pose']:
    print(30*'=')
    print(f'Feature: {feature}')
    test_dataset = MME_Loader(test_data, config=config, validation=True,
                              val_feat=feature, num_segments=config['num_segments'])
    test_loader = DataLoader(test_dataset,
                            batch_size=1, shuffle=False,
                            num_workers=config['num_workers'], drop_last=False,
                            collate_fn=None, pin_memory=config['pin_memory'],
                            prefetch_factor=1, persistent_workers=True,
                            # sampler=BiasedSampler(test_dataset)
                            )

    # test_acc = []
    all_preds, all_lbls, files = [], [], []
    # for _ in range(10): 
    for step, test_batch in enumerate(test_loader):
        # slide over the whole recording 
        # for i in range(test_batch['ecg_feats'].shape[1]):

        #     ecg_feat_seg = test_batch['ecg_feats'][:,i:i+1,:]
        #     flow_feat_seg = test_batch['flow_feats'][:,i:i+1,:]
        #     pose_feat_seg = test_batch['pose_feats'][:,i:i+1,:]
        
        for i in range(0, test_batch['ecg_feats'].shape[1], config['num_segments']):

            ecg_feat_seg = test_batch['ecg_feats'][:,i:i+config['num_segments'],:]
            flow_feat_seg = test_batch['flow_feats'][:,i:i+config['num_segments'],:]
            pose_feat_seg = test_batch['pose_feats'][:,i:i+config['num_segments'],:]
            # print(ecg_feat_seg.shape)
            if ecg_feat_seg.shape[1] != config['num_segments']:
                # print('skipping')
                # continue
                # FIX the smaple by getting last 6 segments
                ecg_feat_seg = test_batch['ecg_feats'][:, -config['num_segments']:,:]
                flow_feat_seg = test_batch['flow_feats'][:, -config['num_segments']:,:]
                pose_feat_seg = test_batch['pose_feats'][:, -config['num_segments']:,:]

                # FIX the smaple by appending zeros
                # ecg_feat_seg = ecg_feat_seg.squeeze(0)
                # flow_feat_seg = flow_feat_seg.squeeze(0)
                # pose_feat_seg = pose_feat_seg.squeeze(0)
                # ecg_feat_seg = np.vstack([ecg_feat_seg, np.zeros((config['num_segments'] - ecg_feat_seg.shape[0], 256))])
                # flow_feat_seg = np.vstack([flow_feat_seg, np.zeros((config['num_segments'] - flow_feat_seg.shape[0], 256))])
                # pose_feat_seg = np.vstack([pose_feat_seg, np.zeros((config['num_segments'] - pose_feat_seg.shape[0], 256))])
                # ecg_feat_seg = torch.from_numpy(ecg_feat_seg[np.newaxis, ...])
                # flow_feat_seg = torch.from_numpy(flow_feat_seg[np.newaxis, ...])
                # pose_feat_seg = torch.from_numpy(pose_feat_seg[np.newaxis, ...])

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
    cprint(f'Average Acc: {np.mean(matrix.diagonal()/matrix.sum(axis=1)):.4f}', 'green')
    cprint(f'Class-wise Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}', 'light_magenta')







#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

op = calculate_confusion_matrix_gtcs_pnes(all_lbls, np.argmax(all_preds, axis=1),
                                          files, tolerance=3)

TP = op['TP']
TN = op['TN']
FP = op['FP']
FN = op['FN']

sense = TP / (TP + FN)
spec = TN / (TN + FP)
prec = TP / (TP + FP)
f1 = 2 * (prec * sense) / (prec + sense)
print(f'Sensitivity: {sense:.4f}, Specificity: {spec:.4f}, Precision: {prec:.4f}, F1: {f1:.4f}')
# generate and display confusion matrix from TP, TN, FP, FN
cm = np.array([[TP, FN],
               [FP, TN]])

if config['exp_typ'] == 'bvp':
    cls_names = ['PNES', 'B.line']
elif config['exp_typ'] == 'gvp':
    cls_names = ['PNES', 'GTCS']
elif config['exp_typ'] == 'bvg':
    cls_names = ['GTCS', 'B.line']

# plot_cm(cm, cls_names)
print(cm)
print(best_chkpt) 
print(cpth)
#%%
'''
# ERROR: IF we use the following as this will put two seprate classes  i.e., baseline and 
# seizuer of same patient under one filename so function will consider them to fall under
# one instance/pat_id, which will split the all_lbl array wrongly creating a wong evaluation
# criterion. baseline and seizure coming form one patient should be considered as two seprate 
# entries/pat_id in the confusion matrix evaluation function.
'''
# x = []
# for f in files:
#     x.append(f.replace('_baseline', ''))
# op = calculate_confusion_matrix_gtcs_pnes(1-all_lbls, 1-np.argmax(all_preds, axis=1),
#                                           x, tolerance=1)







# %%
