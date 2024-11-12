#%%
from torch import nn
from torch import cat
import torch
import torch.nn.functional as F
import time
import torch.nn.init as init
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###MFI block
class mod_diff_code_block(nn.Module):
    def __init__(self,in_channel, num_classes=5):
        super(mod_diff_code_block,self).__init__()
        self.Relation1 = nn.Sequential(
            nn.Linear(in_channel * 4, in_channel*2),
            nn.LeakyReLU(),
            nn.Linear(in_channel*2, in_channel)
            # nn.LeakyReLU(),
        )
        
        self.fc_out = nn.Linear(in_channel*3, num_classes)

    def forward(self, x1,x2,x3,mod_code):     ####mod_code: b*4
        b,c,h,w,l = x1.shape # bxcx1x1x1
        X_ori = torch.cat((x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1)),1)  ###(b*3*c*h*w*l)
        X = torch.mean(X_ori.view(b,3,c,h*w*l),-1) ##b*4*c
        X1 = X.unsqueeze(1).repeat(1,3,1,1)  ##b*3*3*c
        X2 = X.unsqueeze(2).repeat(1,1,3,1)
        X_R = torch.cat((X1, X2),-1)   ###b*3*3*2c

        mod_code = mod_code.unsqueeze(-1).repeat(1, 1, 2 * c)
        X_R_1, X_R_2, X_R_3 = self.Relation1(torch.cat((X_R[:,0,:,:],mod_code),dim=-1)),self.Relation1(torch.cat((X_R[:,1,:,:],mod_code),dim=-1)),\
                                     self.Relation1(torch.cat((X_R[:,2,:,:],mod_code),dim=-1))


        X_R_1,X_R_2,X_R_3 = F.softmax(X_R_1, 1),F.softmax(X_R_2, 1),F.softmax(X_R_3, 1)
        X_1_out = torch.matmul(X_ori.view(b,3,c,h*w*l).permute(0,2,3,1),X_R_1.permute(0,2,1).unsqueeze(-1)).squeeze(-1)  ##b*c*(h*w*l)*4 and b*c*4*1 -> b*c*(h*w*l)*1
        X_2_out = torch.matmul(X_ori.view(b, 3, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_2.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        X_3_out = torch.matmul(X_ori.view(b, 3, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_3.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)


        X_1_out,X_2_out,X_3_out = X_1_out.reshape(b,c,h,w,l),X_2_out.reshape(b,c,h,w,l),X_3_out.reshape(b,c,h,w,l)

        all_feat = torch.cat([x1+X_1_out,x2+X_2_out,x3+X_3_out], dim=1)
        all_feat = all_feat.squeeze()
        
        feat_out = self.fc_out(all_feat)
        
        if len(all_feat.shape) == 1: # out shap is Bx768 so in case of batch size 1, add a dim
            all_feat = all_feat.unsqueeze(0)
        # print(f'mfi=={all_feat.shape}')
        return all_feat

class MFI(nn.Module):
    def __init__(self, in_channel=256, num_classes=3, chkpt=None):
        super(MFI, self).__init__()
        self.mfi = mod_diff_code_block(in_channel, num_classes)
        if chkpt:
            self.mfi.load_state_dict(torch.load(chkpt))
            print('Loaded MFI Module')
    def forward(self, ecg, flow, pose):
        B = ecg.shape[0]
        ecg = ecg.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        flow = flow.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pose = pose.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return self.mfi(flow, ecg, pose, torch.ones((B,3)).to(DEVICE))
#%%
# f = MFI(chkpt='/home/talha/Data/mme/gp_test/fusion_modules/chkpt/mfi5.pth')
# x = torch.randn((5,1,256)).to(DEVICE)
# f.to(DEVICE)
# y = f(x,x,x)
    
# mfi = mod_diff_code_block(256, 3)
# mfi.load_state_dict(torch.load('/home/talha/Data/mme/gp_test/fusion_modules/chkpt/mfi5.pth'))
# %%
