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

from tools import calculate_confusion_matrix_gtcs_pnes, plot_cm
#%%
fold_number = os.getenv('NUM_FOLD', config['num_fold'])
config['num_fold'] = int(fold_number)

segment_number = os.getenv('NUM_SEGMENTS', config['num_segments'])
config['num_segments'] = int(segment_number)

experiment_name = os.getenv('EXPERIMENT_NAME', config['experiment_name'])
config['experiment_name']=experiment_name

exp_typ = os.getenv('EXPERIMENT_TYPE', config['exp_typ'])
config['exp_typ']=exp_typ

if config['exp_typ'] == 'bvg':
    config["data_dir"] =  '/home/user01/Data/mme/dataset/gtcs_pnes_feats/'
    config['external_validation_dir'] = "/home/user01/Data/mme/dataset/ext_gtcs_feats/"

elif config['exp_typ'] == 'bvp':
    config["data_dir"] =  '/home/user01/Data/mme/dataset/gtcs_pnes_feats/'
    config['external_validation_dir'] = "/home/user01/Data/mme/dataset/ext_pnes_feats/"

elif config['exp_typ'] == 'gvp':
    config["data_dir"] =  '/home/user01/Data/mme/dataset/gvp_feats/'
    config['external_validation_dir'] = "/home/user01/Data/mme/dataset/ext_gvp_feats/"
print(60*'_')
print(config['data_dir'])
print(config['external_validation_dir'])
#%%
# num_classes = len(config['sub_classes'])
# sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config['data_dir'])
train_data, test_data = data.get_data(config['num_fold'])

ext_data = GEN_DATA_LISTS(config['external_validation_dir'])
_, ext_data = ext_data.get_data(1) # as external data is not splitted
#%%
train_dataset = MME_Loader(train_data, config=config, validation=False,
                            val_feat='all', num_segments=config['num_segments'])
test_dataset = MME_Loader(test_data, config=config, validation=False, val_feat='all',
                          t_val=True, num_segments=config['num_segments'])

all_dataset =  torch.utils.data.ConcatDataset([train_dataset, test_dataset])

train_loader = DataLoader(all_dataset,
                          batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=1, persistent_workers=True,
                        # sampler=BiasedSampler(train_dataset)
                        )

ext_dataset = MME_Loader(ext_data, config=config, validation=False, val_feat='all',
                          t_val=True, num_segments=config['num_segments'])
ext_loader = DataLoader(ext_dataset,
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

batch = next(iter(ext_loader))
print(f"Batch Shape: {batch['ecg_feats'].shape}, {batch['lbl'].shape}")
print(f"Batch Shape: {batch['flow_feats'].shape}, {batch['pose_feats'].shape}")
print(batch['filename'][0:2])
#%%
# cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/mfi1.pth'
# cpth = f'/home/user01/Data/mme/time_exps/fusion_modules/chkpt/gp_rag{config["num_fold"]}.pth'
# cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cmft5.pth'
# cpth = '/home/user01/Data/mme/gp_test/fusion_modules/chkpt/cat1.pth'

cpth = f"/home/user01/Data/mme/time_exps/fusion_modules/chkpt/ext_{config['exp_typ']}_rag.pth"
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
epoch, best_f1 = 0, 0
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
            for step, test_batch in enumerate(ext_loader):
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
        current_f1 = f1
        # cprint(f'Class Accuracies:: {matrix.diagonal()/matrix.sum(axis=1)}', 'blue')
        # cprint(f'Class-wise Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}', 'light_magenta')
        #####################
        if current_f1 > best_f1 and epoch != 0:
            best_f1 = current_f1
            best_p, best_r, best_f1 = p, r, f1
            best_acc = current_acc
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


# %%
