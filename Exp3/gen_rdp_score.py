from sklearn.metrics import roc_auc_score
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from DeepODModel.RDP.model import RDP_Model
import numpy as np
import random


import pandas as pd
from tqdm import tqdm
from DeepODModel.RDP.util import random_list


from exp_utils import dataLoading


class RDPTree():
    def __init__(self,
                 t_id,
                 tree_depth,
                 filter_ratio=0.1):

        self.t_id = t_id
        self.tree_depth = tree_depth
        self.filter_ratio = filter_ratio
        self.thresh = []

    # include train and eval
    def training_process(self,
                         x,
                         labels,
                         batch_size,
                         node_batch,
                         node_epoch,
                         eval_interval,
                         out_c,
                         USE_GPU,
                         LR,
                         save_path,
                         logfile=None,
                         dropout_r=0.1,
                         svm_flag=False,
                         ):
        if svm_flag:
            x_ori = x.toarray()
        else:
            x_ori = x
        labels_ori = labels
        x_level = np.zeros(x_ori.shape[0])


        # form x and labels
        keep_pos = np.where(x_level == 0)
        x = x_ori[keep_pos]

        group_num = int(x.shape[0] / batch_size) + 1
        batch_x = np.array_split(x, group_num)
        model = RDP_Model(in_c=x.shape[1], out_c=out_c, USE_GPU=USE_GPU,
                          LR=LR, logfile=logfile, dropout_r=dropout_r)


        AUC = []

        Scores = []
        sample_interval = node_epoch / 300
        for epoch in tqdm(range(0, node_epoch)):
            if not is_batch_replace:
                random.shuffle(batch_x)
                batch_cnt = 0
                for batch_i in batch_x:
                    gap_loss = model.train_model(batch_i, epoch)
                    # print("epoch ", epoch, "loss: ", loss)
                    batch_cnt += 1
                    if batch_cnt >= node_batch:
                        break

            else:
                # random sampling with replacement

                for batch_i in range(node_batch):
                    random_pos = random_list(0, x.shape[0] - 1, batch_size)
                    batch_data = x[random_pos]
                    gap_loss = model.train_model(batch_data, epoch)

            if epoch % sample_interval == 0:
                scores = model.eval_model(x_ori)
                auc = roc_auc_score(labels_ori,scores)
                # print(scores.shape)

                Scores.append(scores.reshape(-1))

                AUC.append(auc)


        return AUC,np.stack(Scores,axis=-1)

def main(data_path):
    # global random_size
    x_ori, labels_ori = dataLoading(data_path)
    # build forest
    forest = []
    for i in range(forest_Tnum):
        forest.append(RDPTree(t_id=i+1,
                              tree_depth=tree_depth,
                              filter_ratio=filter_ratio,
                              ))

    x = x_ori

    labels = labels_ori

    AUC,Scores = forest[0].training_process(
        x=x,
        labels=labels,
        batch_size=batch_size,
        node_batch=node_batch,
        node_epoch=node_epoch,
        eval_interval=eval_interval,
        out_c=out_c,
        USE_GPU=USE_GPU,
        LR=LR,
        save_path=save_path,
        logfile=logfile,
        dropout_r=dropout_r,

    )

    return AUC,Scores


def run_one(path,file_name):
    data_path = os.path.join(path, file_name)
    _,Scores= main(data_path)
    return Scores

if __name__ == '__main__':
    template_model_name = 'rdp'
    train_epoch = 300
    g = os.walk(r"./data")

    save_path = "../DeepODModel/RDP/save_model/"
    log_path = "../DeepODModel/RDP/logs/log.log"
    logfile = None
    node_batch = 30
    node_epoch = train_epoch  # epoch for a node training
    eval_interval = 24
    batch_size = 192
    out_c = 50
    USE_GPU = True
    LR = 1e-1
    tree_depth = 1 # boost
    forest_Tnum = 1 # boost
    filter_ratio = 0.05  # filter those with high anomaly scores
    dropout_r = 0.1
    # random_size = 10000  # randomly choose 1024 size of data for training
    is_batch_replace = True
    is_eval = False
    test_1l_only = True

    dict_path= f'./od-predict-score/{template_model_name}_scores.npy'

    cnt = 0
    Dict = dict()
    for path,dir_list,file_list in g:
        for file_name in file_list:
            scores = run_one(path,file_name)

            Dict[file_name] = scores
            print(cnt)
            if cnt %5==0:
                print('cnt save at ',cnt)
                np.save(dict_path,Dict)
            cnt+=1
    np.save(dict_path,Dict)

