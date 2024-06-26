import sys

sys.path.append("../..")
from Ensemble.layers.batch_ensemble_layers import BatchEnsemble_Linear

import torch
import torch.nn as nn



class ROBOD_LinearNet(nn.Module):
    """
    Describe: base architecture to concatenate the BatchEnsemble layers and create a network 
              parameters are similar to the BatchEnsemble layers
              Note: this is Linear layer ensembles
              
              Parameters: input_dim_list: the maximum number of nodes between each layers
                          num_models: number of submodels in ROBOD
                          device: the model is running on which GPUs or CPUs
                          dropout: dropout rate between layers
                          bias: bool, if we want to use bias, default = True
                          is_masked: bool, if we want to mask out some layers, default = False
                          masks: list, should be the number of nodes to mask out in each layer                   
    """
    def __init__(self, 
                 input_dim_list = [784, 400],
                 num_models = 2, 
                 device = "cuda",
                 dropout= 0.2,
                 bias = True,
                 is_masked = False,
                 masks = None):
        super(ROBOD_LinearNet, self).__init__()
        
        assert len(input_dim_list) >= 2
        
        #initialize the variables
        self.device = device
        self.input_dim_list = input_dim_list
        self.num_models = num_models  
        
        self.dropout = nn.Dropout(p=dropout)
        self.input_layer_list = nn.ModuleList()
        self.output_layer_list = nn.ModuleList()
        self.activation = torch.relu
        
        if masks != None:
            output_masks = masks[::-1][1:]   
        else:
            masks = [None for i in range(len(input_dim_list) -1)]
            output_masks = masks[::-1][1:]
            
        
        for i in range(len(input_dim_list) - 1):
            if i == 0:
                first_layer = True
            else:
                first_layer = False
            self.input_layer_list.append(BatchEnsemble_Linear(in_channels= input_dim_list[i],
                                                             out_channels = input_dim_list[i+1], 
                                                             first_layer = first_layer, 
                                                             num_models =self.num_models, 
                                                             bias = True, 
                                                             constant_init = False,
                                                             device = "cuda",
                                                             is_masked = is_masked,
                                                             mask = masks[i]))      
        output_dim_list = input_dim_list[::-1]
        for i in range(len(output_dim_list) -2 ):
            self.output_layer_list.append(BatchEnsemble_Linear(in_channels = output_dim_list[i],
                                                               out_channels= output_dim_list[i+1],
                                                               first_layer = False, 
                                                               num_models =self.num_models, 
                                                               bias = True, 
                                                               constant_init = False,
                                                               device = device,
                                                               is_masked = is_masked,
                                                               mask = output_masks[i],))      
        self.output_layer_list.append(BatchEnsemble_Linear(in_channels = output_dim_list[-2],
                                                           out_channels= output_dim_list[-1],
                                                           first_layer = False, 
                                                           num_models =self.num_models, 
                                                           bias = True, 
                                                           constant_init = False,
                                                           device = device,
                                                           is_masked = False))
            
    def forward(self, x):
        output_list = []
        for i in range(len(self.input_dim_list) - 1):
            x = self.input_layer_list[i](x)
            x = self.dropout(self.activation(x))
            out = x
            for j in range(- (i+1), -1):
                out = self.output_layer_list[j](out)
                out = self.dropout(self.activation(out))
            out = torch.sigmoid(self.output_layer_list[-1](out))  
            output_list.append(out)
        return output_list
    
