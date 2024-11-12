#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from transformers import ASTModel, ASTConfig
# from ddf.ddf import DDFPack

class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)

class LearnableResizer(nn.Module):
    def __init__(self,):
        super().__init__()
        # Update for fixed output size (128x1024)
        self.output_size = (1024, 128)

        n = 16
        r = 2
        slope = 0.2

        # Change input and output channels to 1 for grayscale images
        self.module1 = nn.Sequential(
            nn.Conv2d(1, n, kernel_size=7, padding=3),  # Updated input channels
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(n, n, kernel_size=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.BatchNorm2d(n)
        )

        resblocks = []
        for i in range(r):
            resblocks.append(ResBlock(n, slope))
        self.resblocks = nn.Sequential(*resblocks)

        self.module3 = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n)
        )

        self.module4 = nn.Conv2d(n, 1, kernel_size=7, padding=3)  # Updated output channels

        # Removed scale factor and updated interpolate
        self.interpolate = partial(F.interpolate, size=self.output_size,
                                   mode='bilinear', align_corners=False)

    def forward(self, x):
        residual = self.interpolate(x)

        out = self.module1(x)
        out_residual = self.interpolate(out)

        out = self.resblocks(out_residual)
        out = self.module3(out)
        out = out + out_residual

        out = self.module4(out)
        out = out + residual

        return out

class EWT(nn.Module):
    def __init__(self, num_classes: int,
                 sup_classes: int,
                 in_channels: int = 768,
                 mod_feats: int = 256, # as all other modalities output same feature size
                 dropout_ratio: float = 0.3,
                 pretrained_path: str = None):
        super(EWT, self).__init__()
        
        self.my_dict = {
                "attention_probs_dropout_prob": 0.0,
                "frequency_stride": 10,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.0,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_length": 2500,
                "model_type": "audio-spectrogram-transformer",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "num_mel_bins": 128,
                "patch_size": 16,
                "qkv_bias": True,
                "time_stride": 10,
                "transformers_version": "4.36.2"
                }
        self.num_classes = num_classes
        self.sup_classes = sup_classes
        self.mod_feats = mod_feats
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.resizer = LearnableResizer()
        # configuration = ASTConfig()
        # configuration.update(self.my_dict)
        # Initializing a model (with random weights) from the MIT/ast-finetuned-audioset-10-10-0.4593 style configuration
        # self.ast = ASTModel(configuration)
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        self.fc_seizure = nn.Linear(self.in_channels, self.sup_classes)# sigmoid
       

        # if pretrained_path is not None:
        #     self.init_weights(pretrained_path)

    def forward(self, x):
        # B*C*H*W
        x = self.resizer(x).squeeze(1) # B*TIME*SCALE
        x = self.ast(x)
        x = x.pooler_output # B*TIME*SCALE
        
        if self.dropout is not None:
            x = self.dropout(x)

        # [N, mod_feats]
        x = x.view(x.size(0), -1)
        # [N x num_classes]
        seizure_conf = self.fc_seizure(x)
        

        return seizure_conf
    
    def init_weights(self, checkpoint_path):
        print('Loading EWT pretrianed chkpt...')
        pretrained_dict = torch.load(checkpoint_path)
            # load model state dict
        state = self.ast.state_dict()
        # loop over both dicts and make a new dict where name and the shape of new state match
        # with the pretrained state dict.
        matched, unmatched = [], []
        new_dict = {}
        for i, j in zip(pretrained_dict.items(), state.items()):
            pk, pv = i # pretrained state dictionary
            nk, nv = j # new state dictionary
            # if name and weight shape are same
            if pk.strip('backbone.') == nk:# and pv.shape == nv.shape: #.strip('backbone.')
                new_dict[nk] = pv
                matched.append(pk)
            elif pv.shape == nv.shape:
                new_dict[nk] = pv
                matched.append(pk)
            else:
                unmatched.append(pk)

        state.update(new_dict)
        self.ast.load_state_dict(state, strict=False)
        print('Pre-trained SlowFast state loaded successfully...')
        print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        print(40*'=')
#%%

# model = EWT(5,
# pretrained_path='/home/talha/Data/mme/scripts/models/pretrained/ast-finetuned-audioset-10-10-0.4593.pth')

# x = torch.randn([2,1,2500,128])

# y = model(x)
#%%
