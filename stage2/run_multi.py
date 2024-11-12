import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/mme/time_exps/')

from config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

# import time


# folds = [1,2,3,4,5]
folds = [1] # for external validation
segments = [3,4,6,7,9,10,11,13,14,16,17]
exp_types = ['bvg', 'gvp', 'bvp']

for exp_type in exp_types:
    for seg in segments:
        for fold in folds:
            # config['experiment_name']= f'rag_gvp{fold}_seg{seg}'
            config['exp_typ'] = exp_type
            config['experiment_name']= f'ext_rag_{exp_type}_seg{seg}'
            config['num_segments'] = seg
            config['num_fold'] = fold

            print(60*'$')
            print(f"Now running {config['experiment_name']}...")
            print(f'Fold: {fold}, Segments: {seg}')
            os.environ['EXPERIMENT_TYPE'] = config['exp_typ']
            os.environ['EXPERIMENT_NAME'] = config['experiment_name']
            os.environ['NUM_FOLD'] = str(config['num_fold'])
            os.environ['NUM_SEGMENTS'] = str(config['num_segments'])
            # run main.py file with updated config
            # os.system('python multi_main.py') # <- internal validation on 5 CV
            os.system('python external_main.py') # <- external validation on 1 CV
            # time.sleep(5)
            print(60*'$')
