#%%
import torch
from models.model import MME_Model
from configs.config import config as model_config
from models.fusion import get_pose_graph, get_batch_indices_n_types

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

B = 1
pose_batch_edge_index, pose_batch_vector = get_pose_graph(batch_size = B)
batch_vector, batch_edge_index, batch_edge_types = get_batch_indices_n_types(batch_size=B,
                                                                  num_nodes=3,
                                                                  self_loops=True)

# chkpt = "/home/talha/Data/mme/chkpts/mfi_fold4.pth"            #Exp3_ecg-sig_fold5.pth"
# chkpt = torch.load(chkpt)

model_config['model']['batch_size'] = 1
model_config['model']['fusion_type'] = 'graph'
model = MME_Model(model_config['model'])
model.to(DEVICE)
# model.load_state_dict(chkpt['model_state_dict'])
model.eval()

#%%

# %%
pose = torch.randn((B, 256)).to(DEVICE) #  node-0

ecg = torch.randn((B, 256)).to(DEVICE)
flow = torch.randn((B, 256)).to(DEVICE)

with torch.no_grad():
    all_mod = torch.cat([ecg, flow, pose], dim=1)
    _ = model.fusion.mod_fusion(all_mod,
                                batch_edge_index, batch_edge_types)
fusion_feat = model.fusion.mode_fusion.fusion_feats
print(fusion_feat.shape)
#%%
fuse_feat = model.fusion.mfi.fusion_feats




def get_fusion_feat(model, pose, ecg, flow, fusion_type):
    if fusion_type == 'mfi':
        pose = pose.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ecg = ecg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        flow = flow.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dummy = torch.ones((B,3)).to(DEVICE)

        with torch.no_grad(): # 786 dim feature
            _, fusion_feat = model.fusion.mfi.mod_diff_code_block(flow, ecg, pose, dummy)

    return fusion_feat















#%%
body = torch.randn((B, 256)).to(DEVICE) #  node-0
face = torch.randn((B, 256)).to(DEVICE) # node-1 
r_hand = torch.randn((B, 256)).to(DEVICE)# node-2
l_hand = torch.randn((B, 256)).to(DEVICE) # node-3

ecg = torch.randn((B, 256)).to(DEVICE)
flow = torch.randn((B, 256)).to(DEVICE)

with torch.no_grad():
    _ = model.fusion(body, face, r_hand, l_hand, ecg, flow,
                    pose_batch_edge_index, pose_batch_vector,
                    batch_edge_index, batch_edge_types)

#%%
fuse_feat = model.fusion.mfi.fusion_feats
#%%
# Exp3_ecg-sig_fold5.pth   mfi_fold2
# chkpt = "/home/talha/Data/mme/chkpts/cmft_fold1.pth"
chkpt = '/home/user01/Data/mme/full_chkpt/full_data_gtcs_v2_epoch0.pth'
chkpt = torch.load(chkpt)

model_config['model']['batch_size'] = 1
model_config['model']['fusion_type'] = 'graph'  # graph  cmft
model = MME_Model(model_config['model'])
model.to(DEVICE)
model.load_state_dict(chkpt['model_state_dict'])
model.eval()

torch.save(
            # model.fusion.mfi.mod_diff_code_block.state_dict(), # MFI
        #    model.fusion.mod_fusion.state_dict(), # graph
            model.fusion.cmft.fusion.state_dict(), # cmft
           '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cmft1.pth')
# %%

