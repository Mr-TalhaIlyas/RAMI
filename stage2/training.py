from config import config
import cv2, os, random
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Accuracy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dprint(input, debug=config['DEBUG']):
    if debug:
        print(input)


class Trainer(object):
    def __init__(self, fusion, model, optimizer):
        self.fusion = fusion
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss() # CrossEntropyLoss  BCEWithLogitsLoss
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    
    def training_step(self, batched_data):
        
        ecg_feat = batched_data['ecg_feats'].type(torch.float).to(DEVICE) # Bx6x256 B*T*C
        flow_feat = batched_data['flow_feats'].type(torch.float).to(DEVICE) # Bx6x256
        pose_feat = batched_data['pose_feats'].type(torch.float).to(DEVICE) # Bx6x256

        # fusing features
        all_fused_feats = []
        self.fusion.eval()
        for t in range(ecg_feat.shape[1]):
            fused_feats = self.fusion(ecg_feat[:,t:t+1,:],
                                      flow_feat[:,t:t+1,:],
                                      pose_feat[:,t:t+1,:])
            fused_feats = fused_feats.unsqueeze(1)
            all_fused_feats.append(fused_feats)
        fused_feats = torch.cat(all_fused_feats, dim=1)

        # print(fused_feats.shape)

        # print(joints.shape)
        lbls = batched_data['lbl'].type(torch.LongTensor).to(DEVICE) # B
                
        self.model.zero_grad()
        self.optimizer.zero_grad()

        _, preds = self.model.forward(fused_feats)
        # print(preds.shape, lbls.shape)
        loss = self.criterion(preds, lbls)
        acc = self.accuracy(preds.softmax(dim=-1).cpu().detach(),
                            lbls.cpu().detach())
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc.item()
    
class Evaluator(object):
    def __init__(self, fusion, model):
        self.fusion = fusion
        self.model = model
        self.sup_accuracy = Accuracy(task="multiclass", num_classes=2)

   
    
    def eval_step(self, batched_data):
        
        ecg_feat = batched_data['ecg_feats'].type(torch.float).to(DEVICE) # Bx6x256 B*T*C
        flow_feat = batched_data['flow_feats'].type(torch.float).to(DEVICE) # Bx6x256
        pose_feat = batched_data['pose_feats'].type(torch.float).to(DEVICE) # Bx6x256

        # fusing features
        all_fused_feats = []
        self.fusion.eval()
        for t in range(ecg_feat.shape[1]):
            fused_feats = self.fusion(ecg_feat[:,t:t+1,:],
                                      flow_feat[:,t:t+1,:],
                                      pose_feat[:,t:t+1,:])
            fused_feats = fused_feats.unsqueeze(1)
            all_fused_feats.append(fused_feats)
        fused_feats = torch.cat(all_fused_feats, dim=1)
        
        # print(fused_feats.shape)
        # print(joints.shape)
        lbls = batched_data['lbl'].type(torch.LongTensor).to(DEVICE) # Bx1
        
        with torch.no_grad():
            _, preds = self.model.forward(fused_feats) 

        preds = preds.softmax(dim=-1).cpu().detach()
        lbl_batch = lbls.cpu().detach()
        
        acc = self.sup_accuracy(preds, lbl_batch)

        return acc.item(), preds.numpy(), lbl_batch.numpy()
    

class Evaluator2(object):
    def __init__(self, fusion, model):
        self.fusion = fusion
        self.model = model
        self.sup_accuracy = Accuracy(task="multiclass", num_classes=2)

   
    
    def eval_step(self, batched_data):
        
        ecg_feat = batched_data['ecg_feats'].type(torch.float).to(DEVICE) # Bx6x256 B*T*C
        flow_feat = batched_data['flow_feats'].type(torch.float).to(DEVICE) # Bx6x256
        pose_feat = batched_data['pose_feats'].type(torch.float).to(DEVICE) # Bx6x256

        # fusing features
        all_fused_feats = []
        self.fusion.eval()
        for t in range(ecg_feat.shape[1]):
            fused_feats = self.fusion(ecg_feat[:,t:t+1,:],
                                      flow_feat[:,t:t+1,:],
                                      pose_feat[:,t:t+1,:])
            fused_feats = fused_feats.unsqueeze(1)
            all_fused_feats.append(fused_feats)
        fused_feats = torch.cat(all_fused_feats, dim=1)
        
        # print(fused_feats.shape)
        # print(joints.shape)
        lbls = batched_data['lbl'].type(torch.LongTensor).to(DEVICE) # Bx1
        
        with torch.no_grad():
            feats, preds = self.model.forward(fused_feats) 

        preds = preds.softmax(dim=-1).cpu().detach()
        lbl_batch = lbls.cpu().detach()
        
        acc = self.sup_accuracy(preds, lbl_batch)

        return acc.item(), preds.numpy(), feats.cpu().detach().numpy(), lbl_batch.numpy()