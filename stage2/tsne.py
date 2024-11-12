import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/talha/Data/mme/time_exps/')

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
import pprint
print(f'Printing Configuration File:\n{30*"="}\n')
# pprint.pprint(config)


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

from dataloader import GEN_DATA_LISTS, MME_Loader
# from data.utils import collate, values_fromreport
from vit import VIT
from model import EWT
from lr_scheduler import LR_Scheduler
from tools import save_chkpt, values_fromreport

from fusion_modules.rag import RAG
from fusion_modules.mfi import MFI
from fusion_modules.cmft import CMFT
from fusion_modules.cat import CAT

from training import Evaluator2

from sklearn.metrics import confusion_matrix, classification_report

from tsaug.visualization import plot
from IPython.display import HTML
from tools import plot_tsne, plot_tsne_3d
import numpy as np
from sklearn.manifold import TSNE
#%%
num_classes = len(config['sub_classes'])
sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config)
# for f in range(5):
FOLD = 5
train_data, test_data = data.get_data(FOLD)
#%%
cpth = f'/home/talha/Data/mme/gp_test/fusion_modules/chkpt/mfi{FOLD}.pth'
# cpth = f'/home/talha/Data/mme/gp_test/fusion_modules/chkpt/rag{FOLD}.pth'
# cpth = '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cmft5.pth'
# cpth = f'/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cat{FOLD}.pth'
fusion = MFI(chkpt=cpth) # CAT MFI CMFT RAG
fusion.to(DEVICE)
model = VIT(time_dim=6, 
            input_dim = 768, # 768 for MFI :: 
            embedding_dim = 256, 
            num_heads = 8, 
            num_layers = 3,
            hidden_dim = 128, 
            dropout_rate=0.5, 
            attn_dropout_rate=0.2,
            class_dim=2,
            return_embedding=False)
model.to(DEVICE)



best_chkpt = f'/home/talha/Data/mme/chkpts/mfi{FOLD}.pth'

model.load_state_dict(torch.load(best_chkpt)['model_state_dict'])
#%%
fusion.eval()
model.eval() # <-set mode important
evaluator = Evaluator2(fusion, model)
# save features and labels of all feature combinations in a dict with keys 
full_feat_dict = {}
for feature in ['all', 'ecg', 'flow', 'pose', 'pose_flow', 'ecg_flow', 'ecg_pose']:
    print(30*'=')
    print(f'Feature: {feature}')
    test_dataset = MME_Loader(train_data, config=config, validation=True,
                            val_feat=feature)
    test_loader = DataLoader(test_dataset,
                            batch_size=1, shuffle=False,
                            num_workers=config['num_workers'], drop_last=False,
                            collate_fn=None, pin_memory=config['pin_memory'],
                            prefetch_factor=1, persistent_workers=True,
                            # sampler=BiasedSampler(test_dataset)
                            )

    test_acc, all_preds, all_lbls = [], [], []
    all_feats = []
    # for _ in range(10): 
    for step, test_batch in enumerate(test_loader):
        
        for i in range(0, test_batch['ecg_feats'].shape[1], 6):

            ecg_feat_seg = test_batch['ecg_feats'][:,i:i+6,:]
            flow_feat_seg = test_batch['flow_feats'][:,i:i+6,:]
            pose_feat_seg = test_batch['pose_feats'][:,i:i+6,:]
            # print(ecg_feat_seg.shape)
            if ecg_feat_seg.shape[1] != 6:
                ecg_feat_seg = ecg_feat_seg.squeeze(0)
                flow_feat_seg = flow_feat_seg.squeeze(0)
                pose_feat_seg = pose_feat_seg.squeeze(0)
                ecg_feat_seg = np.vstack([ecg_feat_seg, np.zeros((6 - ecg_feat_seg.shape[0], 256))])
                flow_feat_seg = np.vstack([flow_feat_seg, np.zeros((6 - flow_feat_seg.shape[0], 256))])
                pose_feat_seg = np.vstack([pose_feat_seg, np.zeros((6 - pose_feat_seg.shape[0], 256))])
                ecg_feat_seg = torch.from_numpy(ecg_feat_seg[np.newaxis, ...])
                flow_feat_seg = torch.from_numpy(flow_feat_seg[np.newaxis, ...])
                pose_feat_seg = torch.from_numpy(pose_feat_seg[np.newaxis, ...])
                # print('Padded Segments: ', ecg_feat_seg.shape)

            # create new test_batch using segments
            test_batch_seg = {'ecg_feats': ecg_feat_seg,
                            'flow_feats': flow_feat_seg,
                            'pose_feats': pose_feat_seg,
                            'lbl': test_batch['lbl']}

            acc, preds, feats, lbl_batch = evaluator.eval_step(test_batch_seg)
            all_feats.append(feats)
            test_acc.append(acc)
            all_preds.append(preds)
            all_lbls.append(lbl_batch)
            # break
    # save the features and theri labels in np arrays for t-SNE plotting
    feat_dict = {'all_feats': np.vstack(all_feats),
                'all_lbls': np.vstack(all_lbls),
                }
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

    # save the features and theri labels in np arrays for t-SNE plotting
    full_feat_dict = {**full_feat_dict, **{f'{feature}_feats': feat_dict['all_feats'],
                                        f'{feature}_lbls': feat_dict['all_lbls']}} 
print(full_feat_dict.keys())
# save dict 
np.save(f'/home/talha/Data/mme/gp_test/mfi_feat/fold_{FOLD}_dict.npy', full_feat_dict)
print(f'/home/talha/Data/mme/gp_test/mfi_feat/fold_{FOLD}_dict.npy')
#%%
# read feat dicts
dicts = fmu.get_all_files('/home/talha/Data/mme/gp_test/mfi_feat/')

fuse, ecg, pose, flow, pose_flow, ecg_flow, ecg_pose = [], [], [], [], [], [], []
fuse_lbls, ecg_lbls, pose_lbls, flow_lbls, pose_flow_lbls, ecg_flow_lbls, ecg_pose_lbls = [], [], [], [], [], [], []
for i in range(len(dicts)):
    # i += 2
    feat_dict = np.load(dicts[i], allow_pickle=True).item()
    print(dicts[i])
    # stack the matching features types and their labels
    # for example, stack all pose features and their labels
    fuse.append(feat_dict['all_feats'])
    ecg.append(feat_dict['ecg_feats'])
    pose.append(feat_dict['pose_feats'])
    flow.append(feat_dict['flow_feats'])
    pose_flow.append(feat_dict['pose_flow_feats'])
    ecg_flow.append(feat_dict['ecg_flow_feats'])
    ecg_pose.append(feat_dict['ecg_pose_feats'])

    fuse_lbls.append(feat_dict['all_lbls'])
    ecg_lbls.append(feat_dict['ecg_lbls'])
    pose_lbls.append(feat_dict['pose_lbls'])
    flow_lbls.append(feat_dict['flow_lbls'])
    pose_flow_lbls.append(feat_dict['pose_flow_lbls'])
    ecg_flow_lbls.append(feat_dict['ecg_flow_lbls'])
    ecg_pose_lbls.append(feat_dict['ecg_pose_lbls'])
    # break
# stack all the features and their labels
fuse = np.vstack(fuse)
ecg = np.vstack(ecg)
pose = np.vstack(pose)
flow = np.vstack(flow)
pose_flow = np.vstack(pose_flow)
ecg_flow = np.vstack(ecg_flow)
ecg_pose = np.vstack(ecg_pose)

fuse_lbls = np.vstack(fuse_lbls)
ecg_lbls = np.vstack(ecg_lbls)
pose_lbls = np.vstack(pose_lbls)
flow_lbls = np.vstack(flow_lbls)
pose_flow_lbls = np.vstack(pose_flow_lbls)
ecg_flow_lbls = np.vstack(ecg_flow_lbls)
ecg_pose_lbls = np.vstack(ecg_pose_lbls)

print(fuse.shape, ecg.shape, pose.shape, flow.shape, pose_flow.shape, ecg_flow.shape, ecg_pose.shape)
print(fuse_lbls.shape, ecg_lbls.shape, pose_lbls.shape, flow_lbls.shape, pose_flow_lbls.shape, ecg_flow_lbls.shape, ecg_pose_lbls.shape)

# fuse_lbls.all() == ecg_lbls.all()


#%%
X = fuse

# reducer = umap.UMAP(n_neighbors=20)
# X_embedded = reducer.fit_transform(X)

X_embedded = TSNE(n_components=2, random_state=43,
                  perplexity=30, early_exaggeration=20.0,
                  # perplexity=3
                  ).fit_transform(X)


def plot_tsne(embeddings, labels, legends=True, exclude_class_0=False):
    # sns.set(style="whitegrid")

    # Filter out class 0 if required
    if exclude_class_0:
        mask = labels != 0
        embeddings = embeddings[mask]
        labels = labels[mask]
        # Recalculate unique classes after filtering
        classes = np.unique(labels)
        # Map labels to a continuous range starting from 0
        label_to_id = {label: id for id, label in enumerate(classes)}
        mapped_labels = np.array([label_to_id[label] for label in labels])
    else:
        classes = np.unique(labels)
        mapped_labels = labels  # Use original labels if not excluding class 0

    # Create a scatter plot
    # plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                          c=mapped_labels, cmap='Set1', s=30, alpha=0.5)
    plt.gca().set_aspect('equal', 'datalim')


    plt.axis('off')
    plt.tight_layout()  # Adjust layout to not cut off elements
    plt.show()
    

plot_tsne(X_embedded, ecg_lbls.squeeze(), legends=False, exclude_class_0=False)
# %%
X = fuse

X_embedded = TSNE(n_components=3, random_state=43,
                  perplexity=300, early_exaggeration=12.0,
                  # perplexity=3
                  ).fit_transform(X)


plot_tsne_3d(X_embedded, fuse_lbls.squeeze())
# %%
