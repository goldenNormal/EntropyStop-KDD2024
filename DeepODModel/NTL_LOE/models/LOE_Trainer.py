import time
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
from DeepODModel.NTL_LOE.utils import compute_pre_recall_f1,format_time
from tqdm import tqdm




class MyLOE_trainer:

    def __init__(self, model, loss_function,device='cuda'):

        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_method = 'loe_hard'
        self.max_epochs = 300
        self.warmup = 2
        self.contamination = 0.1

    def loe_opt_batch_or_x(self,batch_or_x,epoch,opt):
        z = self.model(batch_or_x)

        loss_n,loss_a = self.loss_fun(z)

        if epoch <=self.warmup:
            loss = loss_n
            # loss_mean= loss.mean()
        else:
            score = loss_n-loss_a
            _, idx_n = torch.topk(score, int(score.shape[0] * (1-self.contamination)), largest=False,
                                  sorted=False)
            _, idx_a = torch.topk(score, int(score.shape[0] * self.contamination), largest=True,
                                  sorted=False)
            loss = torch.cat([loss_n[idx_n], loss_a[idx_a]], 0)
            # loss_mean = loss.mean()


        loss_mean = loss.mean()
        opt.zero_grad()
        loss_mean.backward()
        opt.step()

    def _train(self,x,batch, optimizer,epoch):

        self.model.train()

        loss_all = 0

        if not batch:

            self.loe_opt_batch_or_x(x,epoch,optimizer)
        else:

            for i in range(10):
                batch_size = x.shape[0]//20
                idx = torch.randperm(x.shape[0])[:batch_size]
                x_batch = x[idx]
                self.loe_opt_batch_or_x(x_batch,epoch,optimizer)
        return 0


    def detect_outliers(self, x,label):
        model = self.model
        model.eval()


        with torch.no_grad():
            z= model(x)
            score,_ = self.loss_fun(z)
            score = score.detach().cpu().numpy().reshape(-1)
            auc = roc_auc_score(label,score)

        return auc,score

    def train(self, x,y,batch,max_epochs=100, optimizer=None, scheduler=None):

        Od_scores = []
        sample_interval = max_epochs/300
        for epoch in tqdm(range(1, max_epochs+1)):
            self._train(x,batch, optimizer,epoch)
            if epoch % sample_interval ==0:
                auc,score = self.detect_outliers(x,y)

                Od_scores.append(score)
            if scheduler is not None:
                scheduler.step()

        return np.stack(Od_scores,axis=-1)