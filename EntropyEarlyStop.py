import copy
import numpy as np
import torch
def cal_entropy(score):
    score = score.reshape(-1)
    score = score/np.sum(score) # to possibility
    entropy = np.sum(-np.log(score + 10e-8) * score)
    return entropy


# R_down = R_smooth: the threshold of the smoothness of the entropy
class EntropyEarlyStop:
    def __init__(self,k=100,R_smooth=0.01):
        self.k = k
        self.Rsmooth = R_smooth
        self.entropys = []
        self.patience = 0
        self.best_epoch = 0
        self.minEn = None
        self.G = 0


    def step(self,entropy_value):
        self.entropys.append(entropy_value)
        if self.minEn is None:
            self.minEn = entropy_value
            return False
        self.G +=abs(entropy_value - self.entropys[-2])

        if entropy_value < self.minEn and (self.minEn - entropy_value)/self.G > self.Rsmooth:
            self.minEn = entropy_value
            self.best_epoch = len(self.entropys) - 1
            self.patience =0
            self.G = 0
        else:
            self.patience+=1

        if self.patience == self.k:
            return True
        return False

    def getBestEpoch(self):
        return self.best_epoch

def getStopOffline(En,k=100,R_smooth=0.01): # an offline version of EntropyEarlyStop
    ES = EntropyEarlyStop(k,R_smooth)
    for i in range(len(En)):
        isStop = ES.step(En[i])
        if isStop:
            break
    return ES.getBestEpoch(),i

class ModelEntropyEarlyStop:  # version of saving parameters
    '''
    This entropystop version will save the parameters of the model when it encounters the minimum entropy. 
    The make ture model has two functions: load_state_dict() and state_dict().
    '''
    def __init__(self,k=100,R_down=0.01):
        self.k = k
        self.R_down = R_down
        self.entropys = []
        self.patience = 0
        self.best_epoch = 0
        self.minEn = None
        self.G = 0
        self.model = None
        self.isStop = False


    def step(self,x_eval,model): 
        if self.isStop:
            return True
        
        with torch.no_grad():
            model.eval()
            eval_score = model(x_eval).cpu().detach().numpy()
            entropy_value = cal_entropy(eval_score)
            model.train()
            
        self.entropys.append(entropy_value)
        if self.minEn is None:
            self.minEn = entropy_value
            self.model = copy.deepcopy(model)
            return False
        self.G +=abs(entropy_value - self.entropys[-2])

        if entropy_value < self.minEn and (self.minEn - entropy_value)/self.G > self.R_down:
            self.minEn = entropy_value
            self.model.load_state_dict(model.state_dict())
            self.best_epoch = len(self.entropys) - 1
            self.patience =0
            self.G = 0
        else:
            self.patience+=1

        if self.patience >=  self.k:
            self.isStop = True
   
        return self.isStop

    def getBestModel(self):
        return self.model
