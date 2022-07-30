from VPRDataset import VPRDataset
from VPRNetwork import VPRNetwork

import os, sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import xlsxwriter
import numpy as np

import lava.lib.dl.slayer as slayer


#----------------- Initialisation --------------------#

# Make a folder for the trained network
trained_folder = 'Trained'
os.makedirs(trained_folder, exist_ok=True)

# Use GPU
print(torch.cuda.is_available())
device = torch.device('cpu')

#---------------- Create the Network -----------------#

# Create the network
net = VPRNetwork().to(device)

#---------------- Load the Dataset -------------------#

# Load the data
training_set = VPRDataset(train=True)
testing_set  = VPRDataset(train=False)
            
train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(dataset=testing_set , batch_size=32, shuffle=True)

print(training_set[0])

#------------- Training the Network -----------------#

# Define an optimiser 
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training the network
error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device)

# Create a training assistant object
stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

epochs = 2


for epoch in range(epochs):

    for i, (input, label) in enumerate(train_loader): # training loop
        output = assistant.train(input, label)
        print(label)
    print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')

    for i, (input, label) in enumerate(test_loader): # testing loop
        output = assistant.test(input, label)
    print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')

    if epoch%20 == 19: # cleanup display
        print('\r', ' '*len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
        stats_str = str(stats).replace("| ", "\n")
        print(f'[Epoch {epoch:2d}/{epochs}]\n{stats_str}')

    if stats.testing.best_accuracy:
        torch.save(net.state_dict(), trained_folder + '/network.pt')
    stats.update()
    stats.save(trained_folder + '/')
    net.grad_flow(trained_folder + '/')

stats.plot(figsize=(15, 5))

# import the best network during training 
net.load_state_dict(torch.load(trained_folder + '/network.pt'))
net.export_hdf5(trained_folder + '/network.net')

# Get the output for the input to each place
num_places = int(testing_set.stream_length/testing_set.place_duration)
test_loader2  = DataLoader(dataset=testing_set , batch_size=num_places, shuffle=False)
for i, (input, label) in enumerate(test_loader2):
    output = net(input.to(device))
    guesses = assistant.classifier(output).cpu().data.numpy()
    labels = label.cpu().data.numpy()

# Make a confusion matrix
confusion_data = [[0 for i in range(num_places)] for j in range(num_places)]
for qryIndex in range(num_places):
    confusion_data[qryIndex][guesses[qryIndex]] += 1

# maximum number in confusion matrix
max_guess = np.amax(confusion_data)

# Make an excel spreadsheet for the confusion matrix data
workbook = xlsxwriter.Workbook('confusion_matrix.xlsx')
worksheet = workbook.add_worksheet()
worksheet.set_column(0, num_places, 2)

# Add place labels
colour = '#DDDDDD'
cell_format = workbook.add_format()
cell_format.set_bg_color(colour)
worksheet.write_row(0, 1, labels, cell_format)
worksheet.write_column(1, 0, labels, cell_format)

# Add data
for row in range(num_places):
    for col in range(num_places):
        data = confusion_data[row][col]
        colour_level = hex(int(255 - 255*(data/max_guess))).lstrip("0x").zfill(2)
        colour = '#'+ colour_level + colour_level + 'ff'
        cell_format = workbook.add_format()
        cell_format.set_bg_color(colour)
        worksheet.write_number(row + 1, col + 1, data, cell_format)

workbook.close()

# for i in range(1):
#     inp_event = slayer.io.tensor_to_event(input[i].cpu().data.numpy().reshape(2, 34, 34, -1))
#     out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 82, -1))
#     inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
#     out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
#     inp_anim.save(f'gifs/inp{i}.gif', animation.PillowWriter(fps=24), dpi=300)
#     out_anim.save(f'gifs/out{i}.gif', animation.PillowWriter(fps=24), dpi=300)
