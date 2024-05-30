import numpy as np
from time import time
from scipy.stats import kendalltau,rankdata
import math
from sklearn.metrics import roc_auc_score,average_precision_score
## best cluster metrics :XB
def cal_xb(s, y):
    num_anomaly = np.sum(y==1)
    s_rank = np.sort(s)
    threshold = s_rank[-num_anomaly]
    c_normal, c_anomaly = np.mean(s_rank[:-num_anomaly]),np.mean(s_rank[-num_anomaly:])
    c = [c_anomaly if i >= threshold else c_normal for i in s]
    return sum((s-c)**2) / (len(s) * ((c_normal - c_anomaly) ** 2))

def XB_select(OD_scores,y):
    epoch = len(OD_scores)
    metrics = []
    for i in range(epoch):
        metrics.append( cal_xb(OD_scores[i],y))
    stop_epoch = np.argmax(metrics)

    return stop_epoch

def MCS_select(OD_scores,_):
    epoch = len(OD_scores)
    rank_scores = [rankdata(score) for score in OD_scores]
    P = math.ceil(math.sqrt(epoch))
    MC_scores = []
    for i in range(epoch):
        corr_sum = 0
        weight = np.ones((epoch,))
        weight[i] = 0
        weight = weight/np.sum(weight)
        random_epochs = np.random.choice(np.arange(epoch),size=P,replace=False,p=weight).tolist()

        for j in random_epochs:
            i_s,j_s = rank_scores[i],rank_scores[j]
            corr = kendalltau(i_s, j_s)[0]
            corr_sum += corr
        corr_sum/=(epoch-1)
        MC_scores.append(corr_sum)

    return np.argmax(MC_scores)

def HITS_select(OD_scores,_):
    # score_mat: (n_samples, n_models)
    score_mat = np.stack(OD_scores,axis=-1)

    rank_mat = rankdata(score_mat, axis=0)
    inv_rank_mat = 1 / rank_mat
    n_samples, n_models = score_mat.shape[0], score_mat.shape[1]

    hub_vec = np.full([n_models, 1],  1/n_models)
    auth_vec = np.zeros([n_samples, 1])

    hub_vec_list = []
    auth_vec_list = []

    hub_vec_list.append(hub_vec)
    auth_vec_list.append(auth_vec)

    for i in range(500):
        auth_vec = np.dot(inv_rank_mat, hub_vec)
        auth_vec = auth_vec/np.linalg.norm(auth_vec)

        # update hub_vec
        hub_vec = np.dot(inv_rank_mat.T, auth_vec)
        hub_vec = hub_vec/np.linalg.norm(hub_vec)

        # stopping criteria
        auth_diff = auth_vec - auth_vec_list[-1]
        hub_diff = hub_vec - hub_vec_list[-1]


        if np.abs(auth_diff.sum()) <= 1e-10 and np.abs(auth_diff.mean()) <= 1e-10 and np.abs(hub_diff.sum()) <= 1e-10 and np.abs(hub_diff.mean()) <= 1e-10:
            print('break at', i)
            break

        auth_vec_list.append(auth_vec)
        hub_vec_list.append(hub_vec)

    return np.argmax(hub_vec)

def get_AUC_AP(OD_scores,y):
    epoch = len(OD_scores)
    AUC = []
    AP = []
    for i in range(epoch):
        AUC.append(roc_auc_score(y,OD_scores[i]))
        AP.append((average_precision_score(y,OD_scores[i])))
    return AUC,AP

from EntropyEarlyStop import cal_entropy,getStopOffline

def Entropy_select(OD_scores,k=100,Rsmooth=0.01):
    epoch = len(OD_scores)
    En = []
    for i in range(epoch):
        En.append(cal_entropy(OD_scores[i]))

    best_epoch,_ = getStopOffline(En,k,Rsmooth)

    return best_epoch

def all_uoms(OD_scores_mat,y,k,Rsmooth):
    if len(OD_scores_mat.shape)==3:
        OD_scores_mat = OD_scores_mat.reshape((OD_scores_mat.shape[0],OD_scores_mat.shape[-1]))
    epoch = OD_scores_mat.shape[-1]
    OD_scores = []
    for i in range(epoch):
        OD_scores.append(OD_scores_mat[:,i])
    print('begin ...')
    AUC,AP = get_AUC_AP(OD_scores,y)
    print('finish AUC AP')
    max_AUC = np.max(AUC)
    max_AP = np.max(AP)

    # baseline_fn =[XB_, MC_,Hits_,ENS_ ]
    baseline_fn =[XB_select, MCS_select,HITS_select]

    baseline_name = ['XB','MCS','HITS']
    data= dict()
    for i in range(len(baseline_fn)):
        fn = baseline_fn[i]

        name = baseline_name[i]
        t0 = time()
        stop_epoch = fn(OD_scores,y)
        t1 = time()
        data[name] = (AUC[stop_epoch], AP[stop_epoch], round(t1-t0,ndigits=4))
        print('finish ',name,' in ',round(t1-t0,ndigits=4))
    data['Max'] = (max_AUC,max_AP,0)

    t0 = time()
    stop_epoch = Entropy_select(OD_scores,k,Rsmooth)
    t1 = time()
    data['entropy'] = (AUC[stop_epoch], AP[stop_epoch], round(t1-t0,ndigits=4))
    print('finish entropy in ',round(t1-t0,ndigits=4))

    data['Random'] = (np.mean(AUC),np.mean(AP),0)
    data['End'] = (AUC[-1],AP[-1],0)

    print('\n')

    return data


