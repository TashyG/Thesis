# Code adapted  from https://lava-nc.org/lava-lib-dl/slayer/notebooks/nmnist/train.html

import re
from weakref import ref

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os

import torch

#import lava.lib.dl.slayer as slayer

# os.chdir('C:\\Users\\natas\\lava-dl\\')
# print(os.getcwd())

import lava.lib.dl.slayer as slayer
#from nmnist import augment, NMNISTDataset


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
  

def chopData(event_stream, start_seconds, end_seconds):
    stream_start_time  = event_stream['t'].iloc[0]

    chop_start = stream_start_time + start_seconds*1000000
    chop_end = stream_start_time + end_seconds*1000000 -1

    btwn = event_stream['t'].between(chop_start, chop_end, inclusive='both')
    chopped_stream = event_stream[btwn]

    chopped_stream['t'] -= chop_start

    num_events_inplace = len(chopped_stream)
    print(start_seconds, num_events_inplace)

    return chopped_stream

train_traverse1 = qcr_traverses[0]
train_traverse2 = qcr_traverses[1]
test_traverse = qcr_traverses[2]

stream_length = 164
sampling_time = 2

# Load the training streams
event_streams = load_event_streams([train_traverse1, train_traverse2])
event_streams = sync_event_streams(event_streams, [train_traverse1, train_traverse2])

# Switch the columns around
#event_streams[0] = event_streams[0][['p', 'y', 'x', 't']]
#event_streams[1] = event_streams[1][['p', 'y', 'x', 't']]

# Subselect 34 x 34 pixels evenly spaced out
print(event_streams[0], len(event_streams[0]))

# Create the filters
x_select  = [int(i*10.176 + 10.176/2) for i in range(34)]
y_select  = [int(i*7.647 + 7.647/2) for i in range(34)]
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

print(event_streams[0], len(event_streams[0]))

# Divide the event streams into 2 second windows
sub_streams0 = []
sub_streams1 = []
for i in range(0,int(stream_length/sampling_time)):
    sub_streams0.append(chopData(event_streams[0], i*sampling_time, i*sampling_time + sampling_time))
    sub_streams1.append(chopData(event_streams[1], i*sampling_time, i*sampling_time + sampling_time))

samples = sub_streams0 + sub_streams1
print("The number of training substreams is: " + str(len(samples)))


# Get a single sample
i = 50
num_time_bins = int(len(samples[i])/sampling_time)
print("The number of samples: " + str(len(samples[i])))

# Find the place label
label = int(i % (stream_length/sampling_time))
print("The sample number is: " + str(i) + " with a label of: " + str(label))
print(samples[i])

# Turn the sample stream into events
x_event = samples[i]['x'].to_numpy()
y_event = samples[i]['y'].to_numpy()
c_event = samples[i]['p'].to_numpy()
t_event = samples[i]['t'].to_numpy()
event = slayer.io.Event(x_event, y_event, c_event, t_event/1000)
print(event)

t_event = t_event/1000
t_event_print = (np.round(t_event/ 1).astype(int) - 0).astype(np.float16)

print(event.t)

print("max x direction: " + str(np.max(x_event)), "max y direction :" + str(np.max(y_event)))

# Turn the events into a tensor
spike = event.fill_tensor(
        torch.zeros(2, 34, 34, num_time_bins),
        sampling_time=sampling_time,
    )


print(spike.reshape(-1, num_time_bins), label)






# ref_traverse = qcr_traverses[2]
# qry_traverse = qcr_traverses[1]
  
# event_streams = load_event_streams([ref_traverse])
# event_streams = sync_event_streams(event_streams, [ref_traverse])

# print(event_streams[0].to_numpy())
  
# end_event0 = len(event_streams[0]) -1;
# end_event1 = len(event_streams[1]) -1;
# #print(event_streams[0]['t'])
# #print(event_streams[0].loc[4])

# total_time0 = event_streams[0]['t'].loc[end_event0] - event_streams[0]['t'].loc[0]
# total_time1 = event_streams[1]['t'].loc[end_event1] - event_streams[1]['t'].loc[0]

# start0 = event_streams[0]['t'].loc[0]
# end0 = event_streams[0]['t'].loc[0] + 2000000
# so there are approx 164 seconds - divide up into 2 second windows 

#print(event_streams[0]['t'].loc[4] + 1)
# print(total_time0, total_time1)

# total = 0
# for i in range(0, 82):
#     start0 = event_streams[0]['t'].loc[0] + i*2000000
#     end0 = event_streams[0]['t'].loc[0] + i*2000000 + 2000000;

#     btwn = event_streams[0]['t'].between(start0, end0, inclusive=True)
#     num_events_inplace = len(event_streams[0][btwn])
#     total += 100*(num_events_inplace/end_event0)
#     print(i, num_events_inplace, total)

# combined_dataset = pd.concat(event_streams)
# print(combined_dataset)


#print(btwn)
#print(len(event_streams[0][btwn]), end_event0)

# length of event streams
#print(len(event_streams[0])/200000, len(event_streams[1])/200000)

# switched_cols = event_streams[0][['p', 'y', 'x', 't']]
# print(switched_cols)