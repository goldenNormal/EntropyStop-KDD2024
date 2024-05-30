import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
import os
import time
from DeepODModel.AE.SimpleAE import Autoencoder
from exp_utils import dataLoading,weights_init_normal
from torch.optim import Adam

from EntropyEarlyStop import ModelEntropyEarlyStop, cal_entropy

def template_train_eval(x,y,lr=0.001,epoch=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(x).type(torch.FloatTensor).to(device)

    model = Autoencoder(X.size(1)).to(device)
    model.apply(weights_init_normal)
    opt = Adam(model.parameters(),lr=lr)

    from tqdm import tqdm
    OD_scores = []
    Auc = []
    En = []
    Ap = []
    for e in tqdm(range(epoch)):
        model.train()
        recon_score = model(X)

        recon_loss = torch.mean(recon_score)

        loss = recon_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if True:
            model.eval()
            recon_score = model(X)
            #
            numpy_recon_score = recon_score.cpu().detach().numpy()
            Auc.append(roc_auc_score(y,numpy_recon_score))
            Ap.append(average_precision_score(y,numpy_recon_score))
            En.append(cal_entropy(numpy_recon_score))
            OD_scores.append(numpy_recon_score)

    dataset_scores = np.stack(OD_scores,axis=-1)

    return Auc,En,dataset_scores,Ap

def run_just_one(path,file_name,lr=0.001,epoch=500):
    data_path = os.path.join(path, file_name)
    x,y = dataLoading(data_path)

    Auc,En,od_scores,Ap = template_train_eval(x,y)
    from EntropyEarlyStop import getStopOffline
    stop_epoch,total_epoch = getStopOffline(En)

    import matplotlib.pyplot as plt
    plt.subplot(3,1,1)
    plt.plot(Auc)
    plt.subplot(3,1,2)
    plt.plot(Ap)
    plt.subplot(3,1,3)
    plt.vlines([stop_epoch],np.max(En),np.min(En),color='red')
    plt.plot(En)
    savedir = './aeimg/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    plt.savefig(f'{savedir}{file_name}.png')
    plt.show()
    plt.clf()

    maxauc,entropyauc = max(Auc),Auc[stop_epoch]
    return maxauc,entropyauc,od_scores



#
def run_AE(x,y,epoch=250,batch_size=256,lr=0.001,entropyStop=False,n_eval=100,k =100,R_down=0.1):
    '''
    entropyStop: whether to use entropy early stop
    n_eval: number of samples used to calculate entropy
    '''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(x).type(torch.FloatTensor).to(device)
    # shuffle x index
    random_index = np.arange(X.size(0))
    np.random.shuffle(random_index)
    train_X = X[random_index]

    model = Autoencoder(X.size(1)).to(device)
    model.apply(weights_init_normal)
    opt = Adam(model.parameters(),lr=lr)

    if entropyStop:
        ES = ModelEntropyEarlyStop(k=k,R_down=R_down)
        N_eval = min(n_eval, X.shape[0])
        eval_index = np.random.choice(X.shape[0], N_eval, replace=False)
        x_eval = X[eval_index]
        isStop = False
    

    from tqdm import tqdm

    start = time.time()
    
    for _ in tqdm(range(epoch)):
        for i in range(0, X.size(0), batch_size):
            model.train()
            recon_score = model( train_X[i:i + batch_size] )
            loss = torch.mean(recon_score)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            
            if entropyStop:
                isStop = ES.step(x_eval,model)
                if isStop:
                    break
        if entropyStop and isStop:
            break

    endt = time.time()
    train_time =  round(endt - start,ndigits=4)

    if entropyStop:
        model = ES.getBestModel()
    
    model.eval()
    outlier_score_np = model(X).cpu().detach().numpy()
    auc, ap = roc_auc_score(y, outlier_score_np) , average_precision_score(y, outlier_score_np)

    memory_allocated = torch.cuda.max_memory_allocated('cuda') // (1024 ** 2)
    
    return auc,ap,train_time,memory_allocated



