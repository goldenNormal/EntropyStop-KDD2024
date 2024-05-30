import os
import sys
print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np

from DeepODModel.AE.run_ae import run_AE
from exp_utils import dataLoading, set_seed

def run_one(path,file_name):
    # data_path = "/home/mist/od-data/20_letter.npz"
    data_path = os.path.join(path, file_name)
    x,y = dataLoading(data_path)
    
    epoch,lr,batch_size = args.epoch,args.lr,args.batch_size
    
    if not earlyStop:
        ae_auc,ae_ap,ae_rt,ae_mem = run_AE(x,y,epoch,batch_size,lr)

        return (ae_auc,ae_ap,ae_rt,ae_mem)
    else:
        k,R_down,n_eval = args.k,args.Rdown,args.n_eval
        en_auc,en_ap,en_rt,en_mem = run_AE(x,y,epoch,batch_size,lr,
                                    entropyStop=True, k=k,R_down=R_down, n_eval=n_eval)
        return (en_auc,en_ap,en_rt,en_mem)

def str2bool(v):
    if v =='True':
        return True
    elif v == 'False':
        return False

import argparse
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--Rdown', type=float, default=0.01)
    parser.add_argument('--n_eval', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--earlyStop',type=str2bool,default=True) # 'EntropyAE' or 'VanillaAE'
    args = parser.parse_args()

    
    earlyStop = args.earlyStop

    print(earlyStop)


    if not earlyStop:
        template_model_name = 'VanillaAE'
    else:
        template_model_name = 'EntropyAE'
    
    for key,v in sorted(vars(args).items()):
        print(key,'=',v)
    # print()
    print('-'*50)

    set_seed(args.seed)

    try_times = 3
    mean_AUC = []
    mean_AP = []

    for th in range(try_times):
        g = os.walk(r"./data")

        cnt = 0

        Dataset = []
        AUC = []
        AP = []
        TrainTime = []
        Memory = []
        
        for path,dir_list,file_list in g:
            for file_name in file_list:
                print(file_name)
                
                auc,ap,rt, mem = run_one(path,file_name)

                AUC.append(auc)
                AP.append(ap)
                TrainTime.append(rt)
                Memory.append(mem)

                Dataset.append(file_name)
                print(cnt)

                cnt+=1

        df = pd.DataFrame({'Dataset':Dataset,
                            "AUC":AUC,
                            "AP":AP,
                            "RunTime":TrainTime,
                            "Memory":Memory
                            })
        
        df.to_csv(f'./Exp1/results/{template_model_name}-{th}.csv', index=False)
        
        auc_mean,ap_mean = np.mean(AUC),np.mean(AP)
        
        mean_AUC.append(auc_mean)
        mean_AP.append(ap_mean)

    print(f'mean across {try_times} times')
    print('AUC,AP:',np.mean(mean_AUC),np.mean(mean_AP))
    print('detailed:',mean_AUC,mean_AP)

