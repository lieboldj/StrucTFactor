import numpy as np
from torch.utils.data import Dataset

# Datasetclass for transfomration to one-hot encoding
class EnzymeDataset(Dataset):
    def __init__(self, data_X, data_Y):
        self.getAAmap()
        self.data_X = data_X
        self.data_Y = data_Y

        
    
    def __len__(self):
        return len(self.data_X)


    def getAAmap(self):
        aa_vocab = ['A', 'C', 'D', 'E', 
                    'F', 'G', 'H', 'I', 
                    'K', 'L', 'M', 'N', 
                    'P', 'Q', 'R', 'S',
                    'T', 'V', 'W', 'X', 
                    'Y', '_']
        map = {}
        for i, char in enumerate(aa_vocab):
            baseArray = np.zeros(len(aa_vocab)-1)
            if char != '_':
                baseArray[i] = 1
            map[char] = baseArray
        self.map = map
        return

        
    def convert2onehot(self, single_seq):
        single_onehot = []
        for x in single_seq:
            single_onehot.append(self.map[x])
        return np.asarray(single_onehot)
    
    
    def __getitem__(self, idx):
        x = self.data_X[idx]
        x = self.convert2onehot(x)
        y = self.data_Y[idx]
        
        x = x.reshape((1,) + x.shape)
        y = y.reshape(-1)
        return x, y

# Datasetclass for transfomration to one-hot encoding with spatial models
class EnzymeDataset_spatial(Dataset):
    def __init__(self, data_X, data_Y, spatial_seqs, spatial_mode):
        self.spatial_mode = spatial_mode
        self.getAAmap()
        self.data_X = data_X
        self.data_Y = data_Y
        self.spatial = spatial_seqs
        


    
    def __len__(self):
        return len(self.data_X)
    
    def getAAmap(self):

        # tertiary structure encoding
        if self.spatial_mode == 2 or self.spatial_mode == 4:
            aa_vocab = ['A_p', 'A_N', 'C_p', 'C_N', 'D_p', 
                    'D_N', 'E_p', 'E_N', 'F_p', 'F_N', 
                    'G_p', 'G_N', 'H_p', 'H_N', 'I_p', 
                    'I_N', 'K_p', 'K_N', 'L_p', 'L_N', 
                    'M_p', 'M_N', 'N_p', 'N_N', 'P_p', 
                    'P_N', 'Q_p', 'Q_N', 'R_p', 'R_N', 
                    'S_p', 'S_N', 'T_p', 'T_N', 'V_p', 
                    'V_N', 'W_p', 'W_N', 'X_p', 'X_N', 
                    'Y_p', 'Y_N', '_']
        else:
            # secondarry stcture encoding
            aa_vocab = ['A_a', 'A_b', 'A_N', 'C_a', 'C_b', 
                    'C_N', 'D_a', 'D_b', 'D_N', 'E_a', 
                    'E_b', 'E_N', 'F_a', 'F_b', 'F_N', 
                    'G_a', 'G_b', 'G_N', 'H_a', 'H_b', 
                    'H_N', 'I_a', 'I_b', 'I_N', 'K_a', 
                    'K_b', 'K_N', 'L_a', 'L_b', 'L_N', 
                    'M_a', 'M_b', 'M_N', 'N_a', 'N_b', 
                    'N_N', 'P_a', 'P_b', 'P_N', 'Q_a', 
                    'Q_b', 'Q_N', 'R_a', 'R_b', 'R_N', 
                    'S_a', 'S_b', 'S_N', 'T_a', 'T_b', 
                    'T_N', 'V_a', 'V_b', 'V_N', 'W_a', 
                    'W_b', 'W_N', 'X_a', 'X_b', 'X_N', 
                    'Y_a', 'Y_b', 'Y_N', '_']

        map = {}
        for i, char in enumerate(aa_vocab):     # char is not literal a char but a combination (string)
            baseArray = np.zeros(len(aa_vocab)-1)
            if char != '_':
                baseArray[i] = 1
            map[char] = baseArray
        self.map = map
        return
        
    def convert2onehot(self, single_seq, spatial_str):
        single_onehot = []
        for pos, x in enumerate(single_seq):
            if x == '_':
                single_onehot.append(self.map[x])
            else:
                single_onehot.append(self.map[(x + "_" + spatial_str[pos])])
        return np.asarray(single_onehot)
    
    
    def __getitem__(self, idx):
        x = self.data_X[idx]
        spatial = self.spatial[idx]
        y = self.data_Y[idx]
        x = self.convert2onehot(x, spatial)
        
        
        x = x.reshape((1,) + x.shape)
        y = y.reshape(-1)
        return x, y


