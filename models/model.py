# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:06:18 2024

@author: talha
"""
#%%
import os, psutil
# os.chdir(os.path.dirname(__file__))
# os.chdir('/home/user01/Data/mme/scripts_miccai/')
from configs.config import config
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
from models.gcn import PoseGCN
from models.slowfast import SlowFast
from models.ewt import EWT
from models.fusion import Fusion
from models.vit import DILVIT
from models.utils import graph

from models.fusion import get_pose_graph, get_batch_indices_n_types

import torch
import torch.nn as nn
import torch.functional as F

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MME_Model(nn.Module):
    def __init__(self, config):
        super(MME_Model, self).__init__()
        self.fusion_type = config['fusion_type']
        self.num_classes = config['num_sub_classes']
        self.sup_classes = config['num_sup_classes']
        self.pose_batch_edge_index, self.pose_batch_vector = get_pose_graph(batch_size = config['batch_size'],)
        self.batch_vector, self.batch_edge_index, self.batch_edge_types = get_batch_indices_n_types(
                                                                        batch_size=config['batch_size'],
                                                                        num_nodes=3,
                                                                        self_loops=True)

        # INPUT: # [B, C, T, H, W]
        self.slowfast = SlowFast(self.num_classes, self.sup_classes,
                                 dropout_ratio=config['slowfast_dropout_ratio'],
                                 pretrained_path=config['slowfast_pretrained_chkpts'])
        # INPUT:  [N, M, T, V, C]
        self.bodygcn = PoseGCN(num_classes=self.num_classes, sup_classes=self.sup_classes,
                               num_persons=config['num_persons'],
                                backbone_in_channels=config['backbone_in_channels'],
                                head_in_channels=config['head_in_channels'],
                                num_nodes=graph.coco_num_node, inward_edges=graph.coco_inward_edges,
                                checkpoint_path=config['body_pretrainned_chkpts']
                                )
        
        self.facegcn = PoseGCN(num_classes=self.num_classes, sup_classes=self.sup_classes,
                               num_persons=config['num_persons'],
                        backbone_in_channels=config['backbone_in_channels'],
                        head_in_channels=config['head_in_channels'],
                        num_nodes=graph.num_nodes_face, inward_edges=graph.face_inward_edges,
                        )
        
        self.rhgcn = PoseGCN(num_classes=self.num_classes, sup_classes=self.sup_classes,
                             num_persons=config['num_persons'],
                            backbone_in_channels=config['backbone_in_channels'],
                            head_in_channels=config['head_in_channels'],
                            num_nodes=graph.num_nodes_hand, inward_edges=graph.hand_inward_edges,
                            checkpoint_path=config['hand_pretrainned_chkpts']
                            )
        
        self.lhgcn = PoseGCN(num_classes=self.num_classes, sup_classes=self.sup_classes,
                             num_persons=config['num_persons'],
                            backbone_in_channels=config['backbone_in_channels'],
                            head_in_channels=config['head_in_channels'],
                            num_nodes=graph.num_nodes_hand, inward_edges=graph.hand_inward_edges,
                            checkpoint_path=config['hand_pretrainned_chkpts']
                            )
        # INPUT: # B*C*Time*(Scales or bins)
        # self.ewt = EWT(self.num_classes, self.sup_classes, in_channels = config['ewt_head_ch'],
        #                 mod_feats = config['mod_feats'], dropout_ratio= config['ewt_dropout_ratio'],
        #                 pretrained_path=config['ewt_pretrainned_chkpts'])
        self.ewt = DILVIT(pretrained_path=config['ewt_pretrainned_chkpts'])
        
        self.fusion = Fusion(in_channels=config['fusion_in_channels'],
                             heads = config['fusion_heads'],
                             num_relations=len(self.batch_edge_types.unique()),
                             pose_fusion_dropout=config['pose_fusion_dropout'], 
                             mod_fusion_dropout=config['mod_fusion_dropout'],
                             num_super_classes=self.sup_classes,
                             num_sub_classes=self.num_classes,
                             type = self.fusion_type)

    def forward(self, frames, body, face, rh, lh, ecg):
        # frames = [B, C, T, H, W]
        # body = [B, M, T, V, C]
        # face = [B, M, T, V, C]
        # rh = [B, M, T, V, C]
        # lh = [B, M, T, V, C]
        # ecg = [B, C, T, S]
        # sub_lbls = [B, N]
        # super_lbls = [B, N]

        # Slowfast
        of_cls_score, self.of_feats = self.slowfast(frames)
        # Body GCN
        body_cls_score, body_feats = self.bodygcn(body)
        # Face GCN
        face_cls_score, face_feats = self.facegcn(face)
        # Right Hand GCN
        rhand_cls_score, rhand_feats = self.rhgcn(rh)
        # Left Hand GCN
        lhand_cls_score, lhand_feats = self.lhgcn(lh)
        # ECG
        ecg_cls_score, self.ecg_feats = self.ewt(ecg)
        # Fusion
        fusion_cls_score, pose_scores = self.fusion(body_feats, face_feats, rhand_feats, lhand_feats,
                                        self.ecg_feats, self.of_feats,
                                        self.pose_batch_edge_index, self.pose_batch_vector,
                                        self.batch_edge_index, self.batch_edge_types)

        return {'flow_outpus': of_cls_score,
                'body_outputs': body_cls_score,
                'face_outputs': face_cls_score,
                'rhand_outputs': rhand_cls_score,
                'lhand_outputs': lhand_cls_score,
                'ecg_outputs': ecg_cls_score,
                'joint_pose_outputs': pose_scores,
                'fusion_outputs': fusion_cls_score}
#%%
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config['model']['batch_size'] = 1
# # config['model']['fusion_type'] = 'cmft'
# model = MME_Model(config['model'])
# model = model.to(DEVICE)

# B = 1

# x = model(torch.randn((B,3,48,224,224)).to(DEVICE),
#           torch.randn((B,1,150,17,3)).to(DEVICE),
#           torch.randn((B,1,150,70,3)).to(DEVICE),
#           torch.randn((B,1,150,21,3)).to(DEVICE),
#           torch.randn((B,1,150,21,3)).to(DEVICE),
#         # torch.randn((4,1,2500,128)).to(DEVICE),  # for AST
#           torch.randn((B,1,2500)).to(DEVICE), # for ViT
#           # torch.randn((4, 5)),
#           # torch.randn((4,1))
#           )

# # for k,v in x.items():
# #     print(k, v[0].shape, v[1].shape)
# #%%

# model.ecg_feats.shape
# model.of_feats.shape
# model.fusion.pose_fusion.pose_feats.shape
# model.fusion.mod_fusion.fusion_feats.shape