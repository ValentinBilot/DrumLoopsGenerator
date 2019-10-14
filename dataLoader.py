# @Author: bilotv
# @Date:   2019-03-13T09:59:33+01:00
# @Last modified by:   bilotv
# @Last modified time: 2019-03-13T15:37:52+01:00



# -*- coding: utf-8 -*-



import torch
from random import randint
from torch.utils import data
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa

def F_plot2(data_m, title, col_v=np.zeros(0), row_v=np.zeros(0), labelCol='', labelRow=''):
    #plt.imshow(data_m, origin='lower', aspect='auto', extent=[row_v[0], row_v[-1], col_v[0], col_v[-1]], interpolation='nearest')
    plt.figure()
    plt.imshow(data_m, origin='lower', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.set_cmap('magma')
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.title(title)
    #plt.grid(True)
    #time.sleep(1)
    plt.show()

class AllDrumSoundsSlicesDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mel = np.load("../GAN/MelAll/"+self.list_IDs[index])
        ID = self.list_IDs[index]
        mel = mel.reshape(240,101)
        mel = mel[:,0:100]
        out = np.zeros((20,1,240,5))
        for i in range(20):
            out[i,0]=mel[:,5*i:5*i+5]
        X = torch.tensor(out, dtype=torch.float32)
        X = X - X.min()
        X = X/X.max()
        #get labels
        if ID[0:4]=="kick":
            label=[1,0,0,0,0,0]
        if ID[0:4]=="snar":
            label=[0,1,0,0,0,0]
        if ID[0:3]=="hit":
            label=[0,0,1,0,0,0]
        if ID[0:4]=="taik":
            label=[0,0,0,1,0,0]
        if ID[0:4]=="mara":
            label=[0,0,0,0,1,0]
        if ID[0:3]=="tam":
            label=[0,0,0,0,0,1]
        label = torch.tensor(label, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.string)

        return X, label

class AnyDrumSoundsSlicesDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mel = np.load("MelLoops/"+self.list_IDs[index])
        ID = self.list_IDs[index]
        mel = mel.reshape(240,len(mel[0]))
        lenSeq = len(mel[0])//5
        mel = mel[:,0:5*lenSeq]
        #print(mel.shape)
        out = np.zeros((lenSeq,1,240,5))
        for i in range(lenSeq):
            out[i,0]=mel[:,5*i:5*i+5]
        X = torch.tensor(out, dtype=torch.float32)
        X = X - X.min()
        X = X/X.max()

        #label = torch.tensor(label, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.string)

        return X, lenSeq

class AnyDrumSoundsSlices25Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mel = np.load("MelLoops/"+self.list_IDs[index])
        ID = self.list_IDs[index]
        mel = mel.reshape(240,len(mel[0]))
        lenSeq = len(mel[0])//25
        mel = mel[:,0:25*lenSeq]
        #print(mel.shape)
        out = np.zeros((lenSeq,1,240,25))
        for i in range(lenSeq):
            out[i,0]=mel[:,25*i:25*i+25]
        out = np.log(out+1e-8)
        X = torch.tensor(out, dtype=torch.float32)
        sig = torch.nn.Sigmoid()
        #X = X - X.min()
        X = sig(X)
        X = X/X.max()

        #label = torch.tensor(label, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.string)

        return X, lenSeq

class MansDrumSoundsSlices25Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mel = np.load("DATA/AllMelLoops/"+self.list_IDs[index])
        ID = self.list_IDs[index]
        mel = mel.reshape(240,len(mel[0]))
        lenSeq = len(mel[0])//25
        mel = mel[:,0:25*lenSeq]
        #print(mel.shape)
        out = np.zeros((lenSeq,1,240,25))
        for i in range(lenSeq):
            out[i,0]=mel[:,25*i:25*i+25]
        out = np.log(out+1e-8)
        X = torch.tensor(out, dtype=torch.float32)
        sig = torch.nn.Sigmoid()
        #X = X - X.min()
        X = sig(X)
        X = X/X.max()
        X = X**0.5

        #label = torch.tensor(label, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.string)

        return X, lenSeq


class ReconstructionDrumSoundsSlices25Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mel = np.load("DATA/MelAll"+self.list_IDs[index][-9]+"/"+self.list_IDs[index])
        ID = self.list_IDs[index]
        mel = mel.reshape(240,len(mel[0]))
        lenSeq = 1
        mel = mel[:,0:3+25*lenSeq]
        #print(mel.shape)
        out = np.zeros((lenSeq,1,240,25))
        for i in range(lenSeq):
            out[i,0]=mel[:,3+25*i:25*i+25+3]
        out = np.log(out+1e-8)
        X = torch.tensor(out, dtype=torch.float32)
        sig = torch.nn.Sigmoid()
        #X = X - X.min()
        X = sig(X)
        X = X/X.max()

        #label = torch.tensor(label, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.string)

        return X

class Finder(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        mel = np.load("DATA/DoubleMelAll0/"+self.list_IDs[index])
        ID = self.list_IDs[index]
        mel = mel.reshape(240,len(mel[0]))
        lenSeq = 1
        mel = mel[:,0:3+25*lenSeq]
        #print(mel.shape)
        out = np.zeros((lenSeq,1,240,25))
        for i in range(lenSeq):
            out[i,0]=mel[:,3+25*i:25*i+25+3]
        out = np.log(out+1e-8)
        X = torch.tensor(out, dtype=torch.float32)
        sig = torch.nn.Sigmoid()
        #X = X - X.min()
        X = sig(X)
        X = X/X.max()
      #get labels
        if ID[0:4]=="kick":
            label=[1,0,0,0,0,0]
        if ID[0:4]=="snar":
            label=[0,1,0,0,0,0]
        if ID[0:3]=="hit":
            label=[0,0,1,0,0,0]
        if ID[0:4]=="taik":
            label=[0,0,0,1,0,0]
        if ID[0:4]=="mara":
            label=[0,0,0,0,1,0]
        if ID[0:3]=="tam":
            label=[0,0,0,0,0,1]
        label = torch.tensor(label, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.string)

        return X, label

class MCNNDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs):
        'Initialization'
        # .npy files are used as index
        self.list_IDs = list_IDs


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        mel = np.load("MelMCNN/"+ID)
        out = np.log(mel+1e-8)
        X = torch.tensor(out, dtype=torch.float32)
        sig = torch.nn.Sigmoid()
        X = sig(X)
        X = X/X.max()

        wav, sr = librosa.load("wavMCNN/"+ID[:-3]+"wav", sr=48000)
        wav = torch.tensor(wav, dtype=torch.float32)

        #label = torch.tensor(label, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.string)

        return X, wav

if __name__=="__main__":
    import os
    rootname = "MelAllSubSet1/"
    filenames = os.listdir(rootname)
    dataset = ReconstructionDrumSoundsSlices25Dataset(filenames)
    # Create generators
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 6}
    training_generator = torch.utils.data.DataLoader(dataset, **params)
    for local_batch in training_generator:
        print(local_batch.shape)
        spectrogram = local_batch[0,0,0]
        F_plot2(spectrogram, "test")
        lenSeq=lenSeq.item()
        break
