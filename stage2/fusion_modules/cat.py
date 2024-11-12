#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class CatFusion(nn.Module):
    def __init__(self, sub_classes, sup_classes, dropout_ratio=0.0):
        super(CatFusion, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(256 * 6, 512),  # Assuming concatenation of 6 feature vectors each of size 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.dropout = torch.nn.Dropout(dropout_ratio)
        # Classification head for 3-class problem
        self.head_3_classes = nn.Linear(256, sup_classes)
        self.pose_head = nn.Linear(256*4, sup_classes)
        # Classification head for 5-class problem
        # self.head_5_classes = nn.Linear(256, sub_classes)
        
    def forward(self, ecg, flow, pose):
        # Concatenate the features along the feature dimension
        pose_feat = torch.cat((ecg, flow, pose, pose), dim=1)
        concatenated_features = torch.cat((pose_feat, ecg, flow), dim=1)
        
        # Shared layers
        shared_features = self.shared_layers(concatenated_features)
        shared_features = self.dropout(shared_features)
        # Classification heads
        output_3_classes = self.head_3_classes(shared_features)
        pose_output = self.pose_head(pose_feat)
        # output_5_classes = self.head_5_classes(shared_features)
        
        self.pose_feats = shared_features # B*256
        self.fusion_feats = shared_features # B*256
        
        return shared_features#pose_output, output_3_classes

class CAT(nn.Module):
    def __init__(self, chkpt=None):
        super(CAT, self).__init__()
        self.cat_fusion = CatFusion(sub_classes=3, sup_classes=3)
        if chkpt:
            self.cat_fusion.load_state_dict(torch.load(chkpt))
    
    def forward(self, ecg, flow, pose):
        ecg = ecg.squeeze(1)
        flow = flow.squeeze(1)
        pose = pose.squeeze(1)
        return self.cat_fusion(ecg, flow, pose)


# c = CAT('/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cat1.pth')
# x = torch.randn((5,256))
# y = c(x,x,x)
#%%

# import torch
# from models.model import MME_Model
# from configs.config import config as model_config
# from models.fusion import get_pose_graph, get_batch_indices_n_types

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# B = 1
# pose_batch_edge_index, pose_batch_vector = get_pose_graph(batch_size = B)
# batch_vector, batch_edge_index, batch_edge_types = get_batch_indices_n_types(batch_size=B,
#                                                                   num_nodes=3,
#                                                                   self_loops=True)
# # Exp3_ecg-sig_fold5.pth   mfi_fold2
# #  /home/user01/Data/mme/chkpts/gpfold3.pth
# chkpt = "/home/talha/Data/mme/chkpts/cat_fold5.pth"
# chkpt = torch.load(chkpt)

# model_config['model']['batch_size'] = 1
# model_config['model']['fusion_type'] = 'cat'
# model = MME_Model(model_config['model'])
# model.to(DEVICE)
# model.load_state_dict(chkpt['model_state_dict'])
# model.eval()

# torch.save(
#             # model.fusion.mfi.mod_diff_code_block.state_dict(), # MFI
#         #    model.fusion.mod_fusion.state_dict(), # graph
#             # model.fusion.cmft.fusion.state_dict(), # cmft
#             model.fusion.cat_fusion.state_dict(),
#            '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cat5.pth')
# %%
