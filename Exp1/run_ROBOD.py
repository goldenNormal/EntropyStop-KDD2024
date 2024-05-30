import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd

from exp_utils import dataLoading


def run_one(path,file_name):
    # data_path = "/home/mist/od-data/20_letter.npz"
    data_path = os.path.join(path, file_name)
    x,y = dataLoading(data_path)
    from Ensemble.ROBOD import LinearROBOD

    model = LinearROBOD(input_dim=x.shape[-1],epochs=100,batch_size=1024,device='cuda')
    X = torch.FloatTensor(x)
    rt,mem,_ =model.fit(X)
    score =model.predict(X)
    auc = roc_auc_score(y,score)
    ap =average_precision_score(y,score)
    print('auc:',auc)
    return auc,ap,rt,mem


if __name__ == '__main__':
    template_model_name = 'ROBOD_nobatch'

    try_times = 1
    for th in range(try_times):
        g = os.walk(r"./data")
        # g = os.walk('./data')
        cnt =0
        Auc = []
        Dataset = []
        Ap =[]
        Mem = []
        RunTime = []
        for path,dir_list,file_list in g:
            for file_name in file_list:
                print(file_name)
                print(cnt)
                cnt +=1
                auc,ap,rt,mem = run_one(path,file_name)

                Auc.append(auc)
                Dataset.append(file_name)
                Ap.append(ap)
                RunTime.append(rt)
                Mem.append(mem)



        df = pd.DataFrame({'Dataset':Dataset,
                           "AUC":Auc,
                           "AP":Ap,
                           "RunTime":RunTime,
                           "Memory":Mem
                           })
        df.to_csv(f'./{template_model_name}-{th}.csv', index=False)
