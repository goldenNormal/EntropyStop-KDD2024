import time
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
from DeepODModel.NTL_LOE.utils import compute_pre_recall_f1,format_time
from tqdm import tqdm


class My_trainer:

    def __init__(self, model, loss_function,device='cuda'):

        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def _train(self,x,batch, optimizer):

        self.model.train()

        loss_all = 0

        if not batch:

            z = self.model(x)

            loss = self.loss_fun(z)
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            loss_all += loss.sum()
        else:

            for i in range(10):
                batch_size = x.shape[0]//20
                idx = torch.randperm(x.shape[0])[:batch_size]
                x_batch = x[idx]
                z = self.model(x_batch)
                loss = self.loss_fun(z)
                loss_mean = loss.mean()
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()
                loss_all += loss.sum()

        # cnt+=1
        # if cnt==10:
        #     break

        return loss_all.item()


    def detect_outliers(self, x,label):
        model = self.model
        model.eval()


        with torch.no_grad():
            z= model(x)
            score = self.loss_fun(z,eval=True)
            score = score.detach().cpu().numpy().reshape(-1)
            auc = roc_auc_score(label,score)

        return auc,score

    def train(self, x,y,batch,max_epochs=100, optimizer=None, scheduler=None):
        Od_scores = []
        sample_interval = max_epochs/300
        for epoch in tqdm(range(1, max_epochs+1)):
            train_loss = self._train(x,batch, optimizer)
            if epoch % sample_interval ==0:
                auc,score = self.detect_outliers(x,y)
                Od_scores.append(score)
            if scheduler is not None:
                scheduler.step()

        return np.stack(Od_scores,axis=-1)