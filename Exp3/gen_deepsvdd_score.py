import numpy as np
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from exp_utils import dataLoading

from DeepODModel.deepsvdd.svddmodel import LinearDeepSVDD

def template_train_eval(x,y):
    model =LinearDeepSVDD([x.shape[-1],64],train_epochs=train_epoch)
    Auc,S = model.fit(x,y)
    return Auc,S


def run_one(path,file_name):
    # data_path = "/home/mist/od-data/20_letter.npz"
    data_path = os.path.join(path, file_name)
    x,y = dataLoading(data_path)

    _,od_scores = template_train_eval(x,y)

    return od_scores

if __name__ == '__main__':
    template_model_name = 'svdd'
    train_epoch = 300
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

