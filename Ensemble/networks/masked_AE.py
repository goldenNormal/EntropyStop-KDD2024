import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ApplyMask:
    """Hook that applies a mask to a tensor.

    Parameters
    ----------
    mask: the mask on the certain layer
    """

    def __init__(self, mask):
        # Precompute the masked indices.
        self._zero_indices = mask  == 0.0

    def __call__(self, w):
        # Hooks are not supposed to modify the argument.
        w = w.clone()
        # A simple element-wise multiplication doesn't work if there are NaNs.
        w[self._zero_indices] = 0.0
        #print(f'\nPerformed operation successfuly for tensor of shape {w.shape}\n')
        return w

    
class MaskLinearAE(nn.Module):
    def __init__(self,
                 input_dim_list,
                 dropout = 0.2,
                 device = "cuda", 
                 param_lst = []):
        """ create masked autoencoders
        device: default is cuda, can be set to cpu
        input_dim_list: input dimension
        
        """
        super(MaskLinearAE, self).__init__()        
        self.device = device
        self.dropout = nn.Dropout(p = dropout)
        
        #input output dim initialization
        self.input_dim_list = input_dim_list 
        self.output_dim_list  = input_dim_list[::-1][1:]
        self.dim_list = self.input_dim_list + self.output_dim_list
        self.n_layer = len(self.dim_list)
        
        #create layers
        self.layer_list = nn.ModuleList() 
        for i in range(self.n_layer - 1):
            self.layer_list.append(nn.Linear(in_features= self.dim_list[i],
                                                      out_features = self.dim_list[i+1]))
        #if pretrained, use the weights and biases from param_lst
        if param_lst != []:
            self.reset_parameters(param_lst)
        
        #masked weights initialization
        self.mask_layer = self.get_mask(self.dim_list,mode="replacement")
        
        #mask the weights -> initialized to zero and 
        #register the backward prop hook that stops update the gradients
        for i in range(self.n_layer -1):
            self.layer_list[i].weight.data[self.mask_layer[i] == 0.0] = 0.0
            self.layer_list[i].weight.register_hook(ApplyMask(self.mask_layer[i]))
            
  
    def reset_parameters(self, param_lst):
        for i in range(self.n_layer -1):
            with torch.no_grad():
                self.layer_list[i].weight.copy_(param_lst[i][0])
                self.layer_list[i].bias.copy_(param_lst[i][1])
              
            
    def pretrain(self, x, layer):
        x = self.layer_list[0](x)
        x = torch.sigmoid(x)
        for i in range(1, 1+ layer):
            x = self.layer_list[i](x)
            x = self.dropout(torch.relu(x))
        for i in range( self.n_layer -2 - layer, self.n_layer -2):
            x = self.layer_list[i](x)
            x = self.dropout(torch.relu(x))
        x = self.layer_list[-1](x)
        x = torch.sigmoid(x)
        return x
                
    def forward(self, x):
        x = self.layer_list[0](x)
        x = self.dropout(torch.sigmoid(x))
        
        for i in range(1, self.n_layer -2):
            x = self.layer_list[i](x)
            x = self.dropout(torch.relu(x))
        
        output = self.layer_list[-1](x)
        output = torch.sigmoid(output)
        return output
    
    
    def get_mask(self, dim_list, mode="replacement") -> np.ndarray:
        """
        Build mask for a layer.
        """
        layer_masks = []
        for i in range(self.n_layer -1):
            if mode == "by_ratio":
                mask = np.random.choice([0, 1], size=(dim_list[i+1], dim_list[i]),
                                        p=[self.drop_ratio, 1 - self.drop_ratio])
                layer_masks.append(torch.LongTensor(mask))
            elif mode == "replacement":
                mask = np.ones(shape=(dim_list[i+1] * dim_list[i],))
                zero_idx = np.random.choice(mask.shape[0], size=mask.shape[0], replace=True)
                zero_idx = np.unique(zero_idx)
                mask[zero_idx] = 0
                mask = mask.reshape((dim_list[i+1], dim_list[i]))
                layer_masks.append(torch.LongTensor(mask))
            else:
                raise NotImplementedError(
                f"Mode {mode} not implemented, choose from 'replacement' or 'by_ratio'.")
        return layer_masks
