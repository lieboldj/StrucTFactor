# import torch packages
import torch
import torch.nn as nn

class DeepTFactor(nn.Module):
    def __init__(self, spatial_mode, out_features=[0] ):
        super(DeepTFactor, self).__init__()
        self.explainECs = out_features
        # different spatial optians and their various layer options (to keep the same ratios of length and width of the kernel)
        if spatial_mode == 3:
            self.layer_info = [[12, 12, 48], [36, 24, 12], [48, 12, 12]]
        elif spatial_mode == 4:
            self.layer_info = [[8, 8, 32], [24, 16, 8], [32, 8, 8]]
        else:
            self.layer_info = [[4, 4, 16], [12, 8, 4], [16, 4, 4]]
        
        self.cnn0 = CNN(spatial_mode, self.layer_info)
        self.fc1 = nn.Linear(in_features=128*3, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=len(out_features))
        self.bn2 = nn.BatchNorm1d(num_features=len(out_features))
        self.out_act = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        
    def forward(self, x):
        x = self.cnn0(x)
        x = x.view(-1, 128*3)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.out_act(self.bn2(self.fc2(x)))
        return x


class CNN(nn.Module):
    '''
    Use second level convolution.
    channel size: 4 -> 16 
                  8 -> 12
                  16 -> 4
    '''
    def __init__(self, spatial_mode, layer_info):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.layers = nn.ModuleList()
        pooling_sizes = []
        for subnetwork in layer_info:
            pooling_size = 0
            self.layers += [self.make_subnetwork(subnetwork, spatial_mode)]
            for kernel in subnetwork:
                pooling_size += (-kernel + 1)
            pooling_sizes.append(pooling_size)
              
        if len(set(pooling_sizes)) != 1:
            raise "Different kernel sizes between subnetworks"
        pooling_size = pooling_sizes[0]
        num_subnetwork = len(layer_info)
        self.conv = nn.Conv2d(in_channels=128*num_subnetwork, out_channels=128*3, kernel_size=(1,1))
        self.batchnorm = nn.BatchNorm2d(num_features=128*3)
        self.pool = nn.MaxPool2d(kernel_size=(1000+pooling_size,1), stride=1)


    def make_subnetwork(self, subnetwork, spatial_mode):
        subnetworks = []
        for i, kernel in enumerate(subnetwork):
            if i == 0:
                # Adjusting the kernel width to always represent the full width of the input matrix in the first layer
                if spatial_mode == 3 or spatial_mode == 1:
                    AAvari = 63
                elif spatial_mode == 4 or spatial_mode == 2:
                    AAvari = 42
                else:
                    AAvari = 21
                subnetworks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(kernel, AAvari)),
                        nn.BatchNorm2d(num_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                )
            else:
                subnetworks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(kernel, 1)),
                        nn.BatchNorm2d(num_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                )
        return nn.Sequential(*subnetworks)

        
    def forward(self, x):
        xs = []
        for layer in self.layers:
            xs.append(layer(x))

        x = torch.cat(xs, dim=1)
        x = self.relu(self.batchnorm(self.conv(x)))
        x = self.pool(x)
        return x