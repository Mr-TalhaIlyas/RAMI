# from configs import *


config = dict(
                gpus_to_use = '0',
                DPI = 300,
                LOG_WANDB= False,
                BENCHMARK= False,
                DEBUG = False,
                USE_EMA_UPDATES = True,
                # alpha = 0.999,
                sanity_check = False,
                project_name= 'MME-Clinical-seq_trans',
                experiment_name= 'ext_rag_gvp_seg3',#'CMFT_fusion_fold3',

                log_directory= "/home/user01/Data/mme/logs/",
                checkpoint_path= "/home/user01/Data/mme/time_exps/chkpts/",

                pretrained_chkpts= "/home/user01/Data/mme/scripts/models/pretrained/",
                # features_porposed   features_mfi  features_cmft gtcs_feats***
                # gvp_feats   gtcs_feats gvp_feats
                data_dir =  '/home/user01/Data/mme/dataset/gtcs_pnes_feats/', # gtcs_feats
                external_validation_dir = '/home/user01/Data/mme/dataset/ext_pnes_feats/',
                exp_typ = 'bvp', # 'gvp', 'bvg', 'bvp', ['bvgp']<- not now.
                pin_memory=  True,
                num_workers= 1,# 2,,6
                
                # not used for dynamic fusion
                # feature_type = 'pose', # 'flow', 'ecg', 'fusion', 'pose'
                num_fold = 1,
                val_feat = 'all', # 'all', 'pose', 'flow', 'ecg'
                num_segments = 3,
                # training settings
                batch_size= 8,

                # learning rate
                learning_rate= 0.001,
                lr_schedule= 'cos',
                num_repeats_per_epoch = 30,
                epochs= 100, #100,
                warmup_epochs= 3,
                WEIGHT_DECAY= 0.0005,
                # AUX_LOSS_Weights= 0.4,

                # '''
                # Dataset
                # '''
                video_fps = 30, # FPS
                ecg_freq = 250, # Hz
                sample_duration = 10, # from [3,5,7,10]

                video_height= 224,
                video_width= 224,
                # sampling distribution
                alpha = 3,
                beta = 1,

                ignore_postictal = True,

                sub_classes = ['baseline', 'focal', 'tonic', 'clonic', 'pnes'],
                super_classes = ['baseline', 'gtcs', 'pnes'],
                # super_classes = ['baseline', 'seizure'],
                # ECG CWT settings
                steps = 128,
                wavelet = "mexh",


                # Model
                # model = config_model.mme
                # AUGMENTATIONS

                )
