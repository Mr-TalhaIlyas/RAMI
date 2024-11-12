

pretrained_root_dir = '/home/user01/Data/mme/scripts_miccai/models/pretrained/'

mme = dict(
            batch_size= 8,

            num_sub_classes = 5,
            num_sup_classes = 3, # wiht sigmoid activation output
            
            fusion_type = 'cmft', # 'cat', 'trans', 'graph', 'cmft'
            
            slowfast_pretrained_chkpts= f"{pretrained_root_dir}slowfast_r50_4x16x1_kinetics400-rgb.pth",
            slowfast_dropout_ratio=0.3,


            num_persons=1,
            backbone_in_channels=3,
            head_in_channels=256,
            body_pretrainned_chkpts= f"{pretrained_root_dir}gcn_body_17kpts_kinetic400.pth",
            hand_pretrainned_chkpts= f"{pretrained_root_dir}gcn_hand_21kpts_fphad45.pth",
            face_pretrainned_chkpts= None,

            ewt_head_ch = 768,
            mod_feats = 256,
            ewt_dropout_ratio = 0.3,
            ewt_pretrainned_chkpts=f'{pretrained_root_dir}ecg_vit_fold1flip.pth',

            fusion_in_channels = 256,
            fusion_heads = 3,
            pose_fusion_dropout=0.3, 
            mod_fusion_dropout=0.3,
            )
