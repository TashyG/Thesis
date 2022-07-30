# Code adapted  from https://lava-nc.org/lava-lib-dl/slayer/notebooks/nmnist/train.html

import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import os

import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import Dataset, DataLoader

import IPython.display as display
from matplotlib import animation
import matplotlib.pyplot as plt



# The traverses
qcr_traverses = [
    "bags_2021-08-19-08-25-42_denoised.parquet",  # S11 side-facing, slow
    "bags_2021-08-19-08-28-43_denoised.parquet",  # S11 side-facing, slow
    "bags_2021-08-19-09-45-28_denoised.parquet",  # S11 side-facing, slow
    "bags_2021-08-20-10-19-45_denoised.parquet",  # S11 side-facing, fast
    "bags_2022-03-28-11-51-26_denoised.parquet",  # S11 side-facing, slow
    "bags_2022-03-28-12-01-42_denoised.parquet",  # S11 side-facing, fast
    "bags_2022-03-28-12-03-44_denoised.parquet",  # S11 side-facing, extra slow
]

gt_times = {
    "bags_2022-03-28-11-51-26_denoised.parquet": [0, 6.7, 13.2, 31, 57, 74, 97, 119, 141, 148, 154],
    "bags_2022-03-28-12-01-42_denoised.parquet": [0, 2.9, 5.8, 12.9, 23.5, 32, 41.5, 50, 59.3, 62, 64.5],
    "bags_2022-03-28-12-03-44_denoised.parquet": [0, 15.5, 30, 63.5, 110, 141, 185, 217, 246.5, 256, 263],
}

qcr_traverses_first_times = {
    "bags_2021-08-19-08-25-42": 2e6, # 1st 
    "bags_2021-08-19-09-45-28": 2e6, # 3rd
    "bags_2022-03-28-11-51-26": 8e6, # 5th
    "bags_2022-03-28-12-01-42": 7.7e6, # 6th
    "bags_2022-03-28-12-03-44": 12e6, # 7th
}

qcr_traverses_last_times = {
    "bags_2021-10-21-10-32-55": 165e6,
    "bags_2021-08-19-08-25-42": 166.2e6,
    "bags_2021-08-19-09-45-28": 166.2e6,
    "bags_2022-03-28-11-51-26": 8e6 + gt_times["bags_2022-03-28-11-51-26_denoised.parquet"][-1] * 1e6,
    "bags_2022-03-28-12-01-42": 7.7e6 + gt_times["bags_2022-03-28-12-01-42_denoised.parquet"][-1] * 1e6,
    "bags_2022-03-28-12-03-44": 12e6 + gt_times["bags_2022-03-28-12-03-44_denoised.parquet"][-1] * 1e6,
}

path_to_event_files = './Dataset/'


def load_event_streams(event_streams_to_load):
    event_streams = []
    for event_stream in tqdm(event_streams_to_load):
        event_streams.append(pd.read_parquet(os.path.join(path_to_event_files, event_stream)))
    return event_streams
  

def get_short_traverse_name(traverse_name):
    m = re.search(r"(\d)\D*$", traverse_name)
    traverse_short = traverse_name[: m.start() + 1]
    return traverse_short

# Synchronise
def sync_event_streams(event_streams, traverses_to_compare):
    event_streams_synced = []
    for event_stream_idx, (event_stream, name) in enumerate(zip(event_streams, traverses_to_compare)):
        short_name = get_short_traverse_name(name)
        start_time = event_stream.iloc[0]["t"]
        if short_name.startswith("bags_"):
            if short_name in qcr_traverses_first_times:
                first_idx = event_stream["t"].searchsorted(start_time + qcr_traverses_first_times[short_name])
            else:
                first_idx = 0
            if short_name in qcr_traverses_last_times:
                last_idx = event_stream["t"].searchsorted(start_time + qcr_traverses_last_times[short_name])
            else:
                last_idx = None
            event_streams_synced.append(event_stream[first_idx:last_idx].reset_index(drop=True))
        else:
            event_streams_synced.append(event_stream)
    return event_streams_synced
  
  
train_traverse1 = qcr_traverses[0]
train_traverse2 = qcr_traverses[1]
test_traverse = qcr_traverses[2]
  

#event_streams = load_event_streams([ref_traverse, qry_traverse])
#event_streams = sync_event_streams(event_streams, [ref_traverse, qry_traverse])
  
# print(event_streams[0]['t'])
# print(event_streams[0].loc[4])
# print(event_streams[0]['t'].loc[4] + 1)


def chopData(event_stream, start_seconds, end_seconds):
    stream_start_time  = event_stream['t'].iloc[0]

    chop_start = stream_start_time + start_seconds*1000000
    chop_end = stream_start_time + end_seconds*1000000 -1

    btwn = event_stream['t'].between(chop_start, chop_end, inclusive='both')
    chopped_stream = event_stream[btwn]

    chopped_stream['t'] -= chop_start

    # For now just choose the first 1000 rows
    chopped_800 = chopped_stream.iloc[0:800]

    return chopped_800
    



class VPRDataset(Dataset):
    """NMNIST dataset method
    Parameters
    ----------
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 2
    stream_length : int, optional
        the length in seconds of traversal that you want to use
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default None.
    """
    def __init__(
        self,
        train=True,
        sampling_time=1, place_duration = 2,
        stream_length=164,
        transform=None,
    ):
        super(VPRDataset, self).__init__()
        if train:
            # Load the training streams
            event_streams = load_event_streams([train_traverse1, train_traverse2])
            event_streams = sync_event_streams(event_streams, [train_traverse1, train_traverse2])

            # Subselect 34 x 34 pixels evenly spaced out - Create the filters
            x_select  = [int(i*10.176 + 10.176/2) for i in range(34)]
            y_select  = [int(i*7.647 + 7.647/2) for i in range(34)]

            # Create filters using the subselected pixels
            filter0x = event_streams[0]['x'].isin(x_select)
            filter0y = event_streams[0]['y'].isin(y_select)
            filter1x = event_streams[1]['x'].isin(x_select)
            filter1y = event_streams[1]['y'].isin(y_select)

            # Apply the filters
            event_streams[0] = event_streams[0][filter0x & filter0y]
            event_streams[1] = event_streams[1][filter1x & filter1y]
            
            # Now reset values to be between 0 and 33
            for i, x in zip(range(34), x_select):
                small_filt0x  = event_streams[0]['x'].isin([x])
                small_filt1x  = event_streams[1]['x'].isin([x])   
                event_streams[0]['x'].loc[small_filt0x] = i
                event_streams[1]['x'].loc[small_filt1x] = i  

            for i, y in zip(range(34), y_select):
                small_filt0y  = event_streams[0]['y'].isin([y])
                small_filt1y  = event_streams[1]['y'].isin([y])
                event_streams[0]['y'].loc[small_filt0y] = i
                event_streams[1]['y'].loc[small_filt1y] = i

            # Divide the event streams into 2 second windows
            sub_streams0 = []
            sub_streams1 = []
            for i in range(0,int(stream_length/place_duration)):
                sub_streams0.append(chopData(event_streams[0], i*place_duration, i*place_duration + place_duration))
                sub_streams1.append(chopData(event_streams[1], i*place_duration, i*place_duration + place_duration))

            self.samples = sub_streams0 + sub_streams1
            print("The number of training substreams is: " + str(len(self.samples)))
            
        else:
            # Load the test stream
            event_streams = load_event_streams([test_traverse])
            event_streams = sync_event_streams(event_streams, [test_traverse])

            # Subselect 34 x 34 pixels evenly spaced out - Create the filters
            x_select  = [int(i*10.176 + 10.176/2) for i in range(34)]
            y_select  = [int(i*7.647 + 7.647/2) for i in range(34)]
            filter0x = event_streams[0]['x'].isin(x_select)
            filter0y = event_streams[0]['y'].isin(y_select)

            # Apply the filters
            event_streams[0] = event_streams[0][filter0x & filter0y]

            # Now reset values to be between 0 and 33
            for i, x in zip(range(34), x_select):
                small_filt0x  = event_streams[0]['x'].isin([x])
                event_streams[0]['x'].loc[small_filt0x] = i

            for i, y in zip(range(34), y_select):
                small_filt0y  = event_streams[0]['y'].isin([y])
                event_streams[0]['y'].loc[small_filt0y] = i

            # Divide the test stream into 2 second windows
            sub_streams = []
            for i in range(0,int(stream_length/place_duration)):
                sub_streams.append(chopData(event_streams[0], i*place_duration, i*place_duration + place_duration))

            self.samples = sub_streams
            print("The number of testing substreams is: " + str(len(self.samples)))

        self.stream_length = stream_length # the length of the full stream in seconds
        self.place_duration = place_duration # The duration of a place in seconds 
        self.sampling_time = sampling_time # Default sampling time is 1
        #self.num_time_bins = int(sample_length/sampling_time)
        self.transform = transform

    def __getitem__(self, i):

        # Find the place label
        num_places = self.stream_length/self.place_duration
        label = int(i % (num_places))
        #print("The sample number is: " + str(i) + " with a label of: " + str(label))

        # Find the sample length and number of time bins
        sample_length = len(self.samples[i])
        num_time_bins = int(sample_length/self.sampling_time)
        #print("The sample Length is " + str(num_time_bins))

        # Turn the sample stream into events
        x_event = self.samples[i]['x'].to_numpy()
        y_event = self.samples[i]['y'].to_numpy()
        c_event = self.samples[i]['p'].to_numpy()
        t_event = self.samples[i]['t'].to_numpy()
        event = slayer.io.Event(x_event, y_event, c_event, t_event/1000)

        # Transform event
        if self.transform is not None:
            event = self.transform(event)

        # Turn the events into a tensor 
        spike = event.fill_tensor(
                torch.zeros(2, 34, 34, num_time_bins),
                sampling_time=self.sampling_time,
            )
        
        return spike.reshape(-1, num_time_bins), label

    def __len__(self):
        return len(self.samples)


# Ultimate test- loading the data
# training_set = VPRDataset(train=True)
# testing_set  = VPRDataset(train=False)
            
# train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
# test_loader  = DataLoader(dataset=testing_set , batch_size=32, shuffle=True)






# for i, (data, labels) in enumerate(train_loader):
#     print(data.shape, labels.shape)
#     print(data,labels)
#     break;


# print(training_set[0])
# spike_tensor, label = training_set[0]
# spike_tensor = spike_tensor.reshape(2, 34, 34, -1)
# print(spike_tensor)
# event = slayer.io.tensor_to_event(spike_tensor.cpu().data.numpy(), sampling_time=2)

# for i in range(5):
#     spike_tensor, label = testing_set[np.random.randint(len(testing_set))]
#     spike_tensor = spike_tensor.reshape(2, 34, 34, -1)
#     event = slayer.io.tensor_to_event(spike_tensor.cpu().data.numpy())
#     anim = event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
#     anim.save(f'gifs/input{i}.gif', animation.PillowWriter(fps=24), dpi=300)

# gif_td = lambda gif: f'<td> <img src="{gif}" alt="Drawing" style="height: 250px;"/> </td>'
# header = '<table><tr>'
# images = ' '.join([gif_td(f'gifs/input{i}.gif') for i in range(5)])
# footer = '</tr></table>'
# display.HTML(header + images + footer)
