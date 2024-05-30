from DeepODModel.deepsvdd.networks.leaky_ae import LeakyLinearAE,LeakyLinearMLP
from DeepODModel.deepsvdd.utils.data_loader import CustomizeDataLoader
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

from tqdm import tqdm
import torch.optim as optim

class LinearDeepSVDD():
    def __init__(self,
                 input_dim_list = [784,400],
                 relu_slope = 0.1,
                 pre_train = False,
                 pre_train_weight_decay = 1e-6,
                 train_weight_decay = 1e-6,
                 pre_train_epochs = 100,
                 pre_train_lr = 1e-4,
                 pre_train_milestones = [0],
                 train_epochs = 300,
                 train_lr = 1e-3,
                 train_milestones = [0],
                 batch_size = 250,
                 device = "cuda",
                 objective = 'one-class',
                 nu = 0.1,
                 warm_up_num_epochs = 10,
                 symmetry = False,
                 dropout = 0.2):
        if pre_train:
            self.ae_net = LeakyLinearAE(input_dim_list = input_dim_list,
                                        symmetry = symmetry,
                                        device = device,
                                        dropout = dropout,
                                        negative_slope = relu_slope)

        self.net = LeakyLinearMLP(input_dim_list = input_dim_list,
                                  device = device,
                                  dropout = dropout,
                                  negative_slope = relu_slope)
        self.rep_dim = input_dim_list[-1]
        self.pre_train = pre_train
        self.device = device

        R = 0.0  # hypersphere radius R
        c = None  # hypersphere center c

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=device) if c is not None else None

        #Deep SVDD Hyperparameters
        self.nu = nu
        self.objective = objective
        self.batch_size = batch_size
        self.pre_train_weight_decay = pre_train_weight_decay
        self.train_weight_decay = train_weight_decay
        self.pre_train_epochs = pre_train_epochs
        self.pre_train_lr = pre_train_lr
        self.pre_train_milestones = pre_train_milestones
        self.train_epochs = train_epochs
        self.train_lr = train_lr
        self.train_milestones = train_milestones
        self.warm_up_n_epochs = warm_up_num_epochs

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(self.rep_dim, device=self.device)
        num_total_batches = train_loader.num_total_batches()
        net.eval()
        with torch.no_grad():
            for idx in range(num_total_batches):
                # get the inputs of the batch
                _,inputs = train_loader.get_next_batch(idx)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def get_radius(self ,dist: torch.Tensor):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu)

    def fit(self, train_X,y):
        #load the datainto loader
        #train_loader = DataLoader(train_data, batch_size= self.batch_size, num_workers=self.n_jobs_dataloader)
        dataloader = CustomizeDataLoader(data = train_X,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)
        total_time = 0.0
        #pretrain the autoencoder

        num_total_batches = dataloader.num_total_batches()

        # Initilalize the net
        self.net = self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters(),
                               lr=self.train_lr,
                               weight_decay=self.train_weight_decay,
                               amsgrad=False)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            #self.c = self.init_center_c(dataloader, self.net)
            self.c = torch.zeros(self.rep_dim, device="cuda")
            print('Center c initialized.')

        self.net.train()

        Auc = []

        S = []
        inputs = torch.FloatTensor(train_X).to('cuda')
        for epoch in tqdm(range(self.train_epochs)):
            loss_epoch = 0.0

            optimizer.zero_grad()
            # Update network parameters via backpropagation: forward + backward + optimize
            outputs = self.net(inputs)
            dist = torch.sum((outputs - self.c) ** 2, dim=1)
            if self.objective == 'soft-boundary':
                scores = dist - self.R ** 2
                loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)
            loss.backward()
            optimizer.step()
            # Update hypersphere radius R on mini-batch distances
            if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                self.R.data = torch.tensor(self.get_radius(dist, self.nu), device= self.device)

            score = self.predict(train_X,y)
            auc = roc_auc_score(y,score)
            Auc.append(auc)

            S.append(score)


        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return Auc,S


    def predict(self, test_data, test_labels):
        #load the overall data into a dataloader, so we can compute the score all together
        dataloader = CustomizeDataLoader(data = test_data,
                                         label = test_labels,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)
        num_total_batches = dataloader.num_total_batches()
        # Set device for network
        self.net = self.net.to(self.device)
        # Testing
        self.net.eval()
        inputs = torch.FloatTensor(test_data).to('cuda')
        with torch.no_grad():

            outputs = self.net(inputs)
            dist = torch.sum((outputs - self.c) ** 2, dim=1)
            if self.objective == 'soft-boundary':
                scores = dist - self.R ** 2
            else:
                scores = dist

        return scores.detach().cpu().numpy()