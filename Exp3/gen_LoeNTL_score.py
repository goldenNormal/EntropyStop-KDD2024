import time
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from DeepODModel.NTL_LOE.config.base import Config, Grid
from DeepODModel.NTL_LOE.models import MyTrainer,LOE_Trainer
from DeepODModel.NTL_LOE.models.Losses import DCL,LOEDCL
from DeepODModel.NTL_LOE.models.NeutralAD import TabNeutralAD
from DeepODModel.NTL_LOE.models.TabNets import TabNets
from DeepODModel.NTL_LOE.utils import compute_pre_recall_f1,format_time
import pandas as pd
from exp_utils import dataLoading
from torch.optim import Adam


def run_one(path,file_name):
    # data_path = "/home/mist/od-data/20_letter.npz"
    data_path = os.path.join(path, file_name)
    x,y = dataLoading(data_path)

    import yaml

    f=open('./DeepODModel/NTL_LOE/config_files/config_tab.yml', 'r')

    model_config=yaml.load(f,Loader=yaml.FullLoader)

    for k in model_config.keys():
        model_config[k] = model_config[k][0]

    model_class = TabNeutralAD
    loss_class = LOEDCL
    optim_class = Adam
    sched_class = torch.optim.lr_scheduler.StepLR

    network = TabNets
    trainer_class = LOE_Trainer.MyLOE_trainer

    x_dim = x.shape[-1]


    model = model_class(network,x_dim, model_config)
    optimizer = optim_class(model.parameters(),
                            lr=  float(model_config['learning_rate']))

    if sched_class is not None:
        scheduler = sched_class(optimizer,step_size=200,gamma=0.5)
    else:
        scheduler = None

    trainer = trainer_class(model, loss_function=loss_class(  float(model_config['loss_temp'])),
                            device=  'cuda')
    X = torch.tensor(x).type(torch.FloatTensor).to('cuda')

    batch = False
    if x.shape[0] * x.shape[1] > 96000000: #;(300000 * 32 )
        batch = True
    od_scores = trainer.train(X,y,batch=batch,
                                     max_epochs=  train_epoch,
                                     optimizer=optimizer, scheduler=scheduler)


    return od_scores

if __name__ == '__main__':
    template_model_name = 'LoeNTL'
    train_epoch= 300
    g = os.walk(r"./data")
    cnt = 0
    Dict = dict()
    skip = -1


    dict_path = f'./od-predict-score/{template_model_name}_scores.npy'
    if skip !=-1:
        Dict = np.load(dict_path,allow_pickle=True)[()]

    for path,dir_list,file_list in g:
        for file_name in file_list:
            print(file_name)
            if cnt<=skip:
                cnt+=1
                continue
            scores = run_one(path,file_name)

            Dict[file_name] = scores

            print(cnt)
            if cnt %5==0:
                print('cnt save at ',cnt)
                np.save(dict_path,Dict)
            cnt+=1
    np.save(dict_path,Dict)



