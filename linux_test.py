import os, sys
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

import IPython.display as display
from matplotlib import animation

sys.path.insert(0,'./lava-dl/tutorials/lava/lib/dl/slayer/nmnist')
print(os.getcwd())

from nmnist import augment, NMNISTDataset



training_set = NMNISTDataset(train=True, transform=augment)
testing_set  = NMNISTDataset(train=False)
            
train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(dataset=testing_set , batch_size=32, shuffle=True)

for i, (data, labels) in enumerate(train_loader):
    print(data.shape, labels.shape)
    print(data,labels)
    break;

print(5)

for i in range(5):
    spike_tensor, label = testing_set[np.random.randint(len(testing_set))]
    print(spike_tensor)
    spike_tensor = spike_tensor.reshape(2, 34, 34, -1)
    print(spike_tensor)
    event = slayer.io.tensor_to_event(spike_tensor.cpu().data.numpy())
    print(event)
print(5)