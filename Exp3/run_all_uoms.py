import pandas as pd
import numpy as np
import argparse
import torch
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from Exp3.UOMS_baselines import all_uoms

def run_one(path,file_name,OD_score_dict):
    print('dataset : ',file_name)
    filepath = os.path.join(path, file_name)
    data = np.load(filepath)

    y = data['y']
    print(np.sum(y==0),np.sum(y==1))

    OD_scores = OD_score_dict[file_name]
    if type(OD_scores) == type([]):
        OD_scores = np.stack(OD_scores,axis=-1)
    print(OD_scores.shape)
    res = all_uoms(OD_scores,y,args.k,args.Rsmooth)
    return res


if __name__ == '__main__':
    g = os.walk(r"./data")

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--Rsmooth', type=float, default=0.01)
    parser.add_argument('--skip', type=int, default=-1)
    parser.add_argument('--model', type=str, default='')

    args = parser.parse_args()
    if args.model=='':
        print('请设置模型名')
        raise Exception

    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    # print()
    print('-'*50)

    Best_AUC = []
    Entropy_AUC = []
    XB_AUC = []
    MC_AUC = []
    HIT_AUC = []
    Random_AUC =[]
    LastAUC = []

    Dataset = []

    Best_AP = []
    Entropy_AP = []
    XB_AP = []
    MC_AP = []
    HIT_AP = []
    Random_AP =[]
    LastAP = []

    Entropy_Time = []
    XB_Time = []
    MC_Time = []
    HIT_Time = []

    dict_path = f'./od-predict-score/{args.model}_scores.npy'

    OD_score_dict = np.load(dict_path,allow_pickle=True)[()]

    cnt =0
    for path,dir_list,file_list in g:
        for file_name in file_list:
            res = run_one(path,file_name,OD_score_dict)

            Best_AUC.append(res['Max'][0])
            Entropy_AUC.append(res['entropy'][0])
            XB_AUC.append(res['XB'][0])
            MC_AUC.append(res['MCS'][0])
            HIT_AUC.append(res['HITS'][0])

            LastAUC.append(res['End'][0])

            Random_AUC.append(res['Random'][0])

            Dataset.append(file_name)

            Best_AP.append(res['Max'][1])
            Entropy_AP.append(res['entropy'][1])
            XB_AP.append(res['XB'][1])
            MC_AP.append(res['MCS'][1])
            HIT_AP.append(res['HITS'][1])
            Random_AP.append(res['Random'][1])
            LastAP.append(res['End'][1])

            Entropy_Time.append(res['entropy'][2])
            XB_Time.append(res['XB'][2])
            MC_Time.append(res['MCS'][2])
            HIT_Time.append(res['HITS'][2])

            print(cnt)
            cnt+=1


    auc_df = pd.DataFrame({'Dataset':Dataset,
                           'Max': Best_AUC,
                           'Entropy': Entropy_AUC,
                           "XB":XB_AUC, # actually train epochs
                           'MCS': MC_AUC,
                           "HITS":HIT_AUC,
                           "Random":Random_AUC,
                           "End":LastAUC,
                           })

    ap_df = pd.DataFrame({'Dataset':Dataset,
                          'Max': Best_AP,
                          'Entropy': Entropy_AP,
                          "XB":XB_AP, # actually train epochs
                          'MCS': MC_AP,
                          "HITS":HIT_AP,
                          "Random":Random_AP,
                          "End":LastAP,
                          })

    time_df = pd.DataFrame({'Dataset':Dataset,
                          'Entropy': Entropy_Time,
                          "XB":XB_Time, # actually train epochs
                          'MCS': MC_Time,
                          "HITS":HIT_Time,
                          })

    auc_df.to_csv(f'./select_res/{args.model}_auc_.csv', index=False)
    ap_df.to_csv(f'./select_res/{args.model}_ap_.csv', index=False)
    time_df.to_csv(f'./select_res/{args.model}_time_.csv', index=False)



