#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, TopKPooling, LayerNorm, RGCNConv
from torch_geometric.utils import (to_undirected, sort_edge_index,
                                   add_self_loops, to_dense_adj)

def generate_edge_types(edge_index):
    """
    only generate edge types for a graph where (i, j) and (j, i) have the same type.
    all self loops (i,i) will have different type of relation
    """
    sorted_edges = torch.sort(edge_index, dim=0).values
    unique_edges, edge_types = torch.unique(sorted_edges, dim=1, return_inverse=True)

    return edge_types

def get_batch_indices_n_types(batch_size, num_nodes, self_loops=False):
    '''
    only generates indices and relation types of a fully connected graph.
    '''
    edge_index = torch.combinations(torch.arange(0, num_nodes),
                                    r=2, with_replacement=False).t().contiguous()
    edge_index = to_undirected(edge_index)
    if self_loops:
        edge_index, _ = add_self_loops(edge_index)
    # create a batch vector to inform layer about indices belonging to one batch
    batch_vector = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    edge_types = generate_edge_types(edge_index)
    
    batch_edge_types = edge_types.repeat(batch_size)
    
    batch_edge_index = edge_index.clone()
    for i in range(1, batch_size):
        # offset the edge indices for each graph in the batch
        offset_edge_index = edge_index + i * num_nodes
        batch_edge_index = torch.cat([batch_edge_index, offset_edge_index], dim=1)
        
    return (batch_vector.to('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_edge_index.to('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_edge_types.to('cuda' if torch.cuda.is_available() else 'cpu'))

def get_pose_graph(batch_size = 5):
    pose_nodes = 4
    # shape: Batch * Nodes * Features
    # => (Batch*Nodes) * Features
    self_link = [(i, i) for i in range(pose_nodes)]
    
    inward = [(1, 0), # face -> body
              (2, 0), # r_hand -> body
              (3, 0)] # l_hand -> body
    
    outward = [(j, i) for (i, j) in inward]
    neighbor = self_link + inward + outward
    # Create edge index for PyTorch Geometric
    edge_index = torch.tensor(neighbor, dtype=torch.long).t().contiguous()
    
    batch_edge_index = edge_index.clone()
    for i in range(1, batch_size):
        # offset the edge indices for each graph in the batch
        offset_edge_index = edge_index + i * pose_nodes
        batch_edge_index = torch.cat([batch_edge_index, offset_edge_index], dim=1)
    
    # create a batch vector to inform layer about indices belonging to one batch
    batch_vector = torch.arange(batch_size).repeat_interleave(pose_nodes)
    
    return (batch_edge_index.to('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_vector.to('cuda' if torch.cuda.is_available() else 'cpu'))

class ModFusion(torch.nn.Module):
    def __init__(self, in_channels, num_relations, num_sub_classes,
                 num_super_classes, dropout_ratio=0.0):
        super(ModFusion, self).__init__()
        self.norm_before_rgcn = LayerNorm(in_channels)
        self.realation_gcn = RGCNConv(in_channels=in_channels,
                                      out_channels=in_channels,
                                      num_relations=num_relations)
        self.norm_after_rgcn = LayerNorm(in_channels)
        self.dropout = torch.nn.Dropout(dropout_ratio)  # Adjust dropout rate as needed
        # Task-specific heads
        # self.sub_class_head = torch.nn.Linear(in_channels, num_sub_classes)
        self.super_class_head = torch.nn.Linear(in_channels, num_super_classes)

    def forward(self, x, batch_edge_index, batch_edge_types):
        B, N, C = x.shape # (Batch, Nodes, Channels)
        x = self.norm_before_rgcn(x)
        x = self.realation_gcn(x.view(-1, C), batch_edge_index, batch_edge_types)
        x = F.relu(self.norm_after_rgcn(x.view(B, N, -1)))
        x = x.mean(dim=1)
        x = self.dropout(x)
        # sub_class_logits = self.sub_class_head(x)
        super_class_logits = self.super_class_head(x)
        
        self.fusion_feats = x
        return x#super_class_logits

class RAG(nn.Module):
    def __init__(self, in_channels=256, num_relations=6, num_sub_classes=3,
                     num_super_classes=3, dropout_ratio=0.0, chkpt=None):
        super(RAG, self).__init__()
        self.fusion = ModFusion(in_channels, num_relations, num_sub_classes,
                 num_super_classes, dropout_ratio)
        if chkpt:
            self.fusion.load_state_dict(torch.load(chkpt))
            # print(f'Matched: {len(matched)}, Unmatched: {len(unmatched)}')
            print('Loaded Fusion Module')
    def forward(self, ecg, flow, pose):
        all_mod = torch.cat([ecg, flow, pose], dim=1)
        B = all_mod.shape[0]
        _, batch_edge_index, batch_edge_types = get_batch_indices_n_types(batch_size=B,
                                                                        num_nodes=3,
                                                                        self_loops=True)
        return self.fusion(all_mod, batch_edge_index, batch_edge_types)
    
#%%
    
# batch_vector, batch_edge_index, batch_edge_types = get_batch_indices_n_types(batch_size=2,
#                                                                   num_nodes=3,
#                                                                   self_loops=True)
    
# fusion = ModFusion(in_channels=256, num_relations=6, num_sub_classes=3,
#                      num_super_classes=3, dropout_ratio=0.0)
# fusion.load_state_dict(torch.load('/home/talha/Data/mme/gp_test/fusion_modules/chkpt/rag5.pth'))

# write a module class to call the fusion module


#%%


# fusion.to('cuda')
# x = torch.randn((8,3,256)).to('cuda')
# x = fusion(x, batch_edge_index, batch_edge_types)






# %%
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
# chkpt = "/home/talha/Data/mme/chkpts/cmft_fold1.pth"
# chkpt = torch.load(chkpt)

# model_config['model']['batch_size'] = 1
# model_config['model']['fusion_type'] = 'cmft'
# model = MME_Model(model_config['model'])
# model.to(DEVICE)
# model.load_state_dict(chkpt['model_state_dict'])
# model.eval()

# torch.save(
#             # model.fusion.mfi.mod_diff_code_block.state_dict(), # MFI
#         #    model.fusion.mod_fusion.state_dict(), # graph
#             model.fusion.cmft.fusion.state_dict(), # cmft
#            '/home/talha/Data/mme/gp_test/fusion_modules/chkpt/cmft1.pth')