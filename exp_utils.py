import random
import torch
import numpy as np
def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)

def kaiming_weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def kaiming_weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data, 0.0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def dataLoading(path):
    filepath = path
    data = np.load(filepath)

    x = data['X']
    from sklearn.preprocessing import StandardScaler
    enc = StandardScaler()
    x = enc.fit_transform(x)
    print(x.shape)
    y = data['y']
    return x,y