#!/usr/bin/env python
# coding: utf-8


############### Import libraries ###############
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull
from progress.bar import Bar
import train_test_functions
import intermediate_output_functions
import models_and_classes

import sys
import time


def updt(total, progress, status):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, "" + status
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    #text = "\r[{}] {:.0f}% {}".format(
    #    "#" * block + "-" * (barLength - block), round(progress * 100, 0),
    #    status)
    a = round(progress * total)
    text = "\r{} {} {} {}".format( status, a, "out of", total-1)
    sys.stdout.write(text)
    sys.stdout.flush()

# In[1]:

############### Some variables ###############
cwd = os.getcwd()
train_dir = cwd + "\\cifar10_dataset\\train" 
test_dir = cwd + "\\cifar10_dataset\\test"
filescwd = cwd + "\\files"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

batch_size = 256
num_classes = 10
no_sams_train = 50000
no_sams_test = 10000
noExperiments = 100
no_layers = 35
np.set_printoptions(precision=3)

"""progresscount = 0
for i in range( no_layers):
    updt(no_layers+1, progresscount + 1, 'Exit Location')
    progresscount += 1
    time.sleep(0.2)
"""  
# In[2]:

############### Dataset ###############
print("-----------")
print("Dataset: CIFAR-10")
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, num_workers=0)
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=0)

# In[3]:

############### Load the model ###############
print("-----------")
print("Loading pretrained WideResNet-101 Model")
wideresnet101 = models_and_classes.wide_resnet101_2().to(device)
wideresnet101.load_state_dict( torch.load( filescwd + "\\wideresnet101_state_dict"))
criterion_r = nn.CrossEntropyLoss()
train_test_functions.test( wideresnet101, criterion_r, 350, testloader)

# In[4]:

############### Save the intermediate layer outputs for Class Means ###############
layerNumbersToConsider = [0,4,5,6,7,9]
print("-----------")
print("Saving Intermediate Outputs Using Train Set [%d Exit Locations - this will take a while]" % (no_layers))
# trainset
progresscount = 0
layerlist = list(wideresnet101.children())
for layerNo in layerNumbersToConsider:
    if isinstance( layerlist[layerNo], nn.Sequential):
        layerlist_for_sequential_sublayer = list( layerlist[layerNo].children())
        for i in range( len( layerlist_for_sequential_sublayer)):
            layersToConsider = layerlist[0:layerNo] + layerlist_for_sequential_sublayer[0:i+1]
            intermediateOut = intermediate_output_functions.getIntermediateOut( layersToConsider, trainloader)
            np.save( filescwd + "\\wideresnet101_" + str(layerNo) + "_" + str(i) + "_train", intermediateOut)
            del intermediateOut
            updt(no_layers+1, progresscount + 1, 'Exit Location')
            progresscount += 1
    else:
        layersToConsider = layerlist[0:layerNo+1]
        last = (layerNo == (len(layerlist) - 1))
        intermediateOut = intermediate_output_functions.getIntermediateOut( layersToConsider, trainloader, last)
        np.save( filescwd + "\\wideresnet101_" + str(layerNo) + "_train", intermediateOut)
        del intermediateOut
        updt(no_layers+1, progresscount + 1, 'Exit Location')
        progresscount += 1

print("\n-----------")
print("Saving Intermediate Outputs Using Test Set [%d Exit Locations - this will take a while]" % (no_layers))
# testset
progresscount = 0
for layerNo in layerNumbersToConsider:
    if isinstance( layerlist[layerNo], nn.Sequential):
        layerlist_for_sequential_sublayer = list( layerlist[layerNo].children())
        for i in range( len( layerlist_for_sequential_sublayer)):
            layersToConsider = layerlist[0:layerNo] + layerlist_for_sequential_sublayer[0:i+1]
            intermediateOut = intermediate_output_functions.getIntermediateOut( layersToConsider, testloader)
            np.save( filescwd + "\\wideresnet101_" + str(layerNo) + "_" + str(i) + "_test", intermediateOut)
            del intermediateOut
            updt(no_layers+1, progresscount + 1, 'Exit Location')
            progresscount += 1
    else:
        layersToConsider = layerlist[0:layerNo+1]
        last = (layerNo == (len(layerlist) - 1))
        intermediateOut = intermediate_output_functions.getIntermediateOut( layersToConsider, testloader, last)
        np.save( filescwd + "\\wideresnet101_" + str(layerNo) + "_test", intermediateOut)
        del intermediateOut
        updt(no_layers+1, progresscount + 1, 'Exit Location')
        progresscount += 1

# In[5]:

print("\n-----------")
print("FLOPs per Layer Evaluated Manually")
############### Architecture variables and FLOP computations for Class Means ###############
individual_layerFlops = np.asarray([  3*(7*7*7*7+48)*16*16*64 + 10*(64*16*16 + 64*16*16 + 64*16*16-1),
                            64*(1*1*1*1)*8*8*128 + 128*(3*3*3*3+8)*8*8*128 + 128*(1*1*1*1)*8*8*256 + 64*(1*1*1*1)*8*8*256 + 10*(256*8*8 + 256*8*8 + 256*8*8-1),
                            256*(1*1*1*1)*8*8*128 + 128*(3*3*3*3+8)*8*8*128 + 128*(1*1*1*1)*8*8*256 + 10*(256*8*8 + 256*8*8 + 256*8*8-1),
                            256*(1*1*1*1)*8*8*128 + 128*(3*3*3*3+8)*8*8*128 + 128*(1*1*1*1)*8*8*256 + 10*(256*8*8 + 256*8*8 + 256*8*8-1),
                                256*(1*1*1*1)*8*8*256 + 256*(3*3*3*3+8)*4*4*256 + 256*(1*1*1*1)*4*4*512 + 256*(1*1*1*1)*4*4*512 + 10*(512*4*4 + 512*4*4 + 512*4*4-1),
                                512*(1*1*1*1)*4*4*256 + 256*(3*3*3*3+8)*4*4*256 + 256*(1*1*1*1)*4*4*512 + 10*(512*4*4 + 512*4*4 + 512*4*4-1),
                                512*(1*1*1*1)*4*4*256 + 256*(3*3*3*3+8)*4*4*256 + 256*(1*1*1*1)*4*4*512 + 10*(512*4*4 + 512*4*4 + 512*4*4-1),
                                512*(1*1*1*1)*4*4*256 + 256*(3*3*3*3+8)*4*4*256 + 256*(1*1*1*1)*4*4*512 + 10*(512*4*4 + 512*4*4 + 512*4*4-1),
                            512*(1*1*1*1)*4*4*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                            1024*(1*1*1*1)*2*2*512 + 512*(3*3*3*3+8)*2*2*512 + 512*(1*1*1*1)*2*2*1024 + 10*(1024*2*2 + 1024*2*2 + 1024*2*2-1),
                                    1024*(1*1*1*1)*2*2*1024 + 1024*(3*3*3*3+8)*1*1*1024 + 1024*(1*1*1*1)*1*1*2048 + 1024*(1*1*1*1)*1*1*2048 + 10*(2048*1*1 + 2048*1*1 + 2048*1*1-1),
                                    2048*(1*1*1*1)*1*1*1024 + 1024*(3*3*3*3+8)*1*1*1024 + 1024*(1*1*1*1)*1*1*2048 + 10*(2048*1*1 + 2048*1*1 + 2048*1*1-1),
                                    2048*(1*1*1*1)*1*1*1024 + 1024*(3*3*3*3+8)*1*1*1024 + 1024*(1*1*1*1)*1*1*2048 + 10*(2048*1*1 + 2048*1*1 + 2048*1*1-1),
                             (2048+2047)*10
                          ])
savethis = individual_layerFlops[1]
individual_layerFlops = individual_layerFlops / individual_layerFlops[1] # to prevent overflow
divider = np.sum( individual_layerFlops) * savethis 
individual_layerFlops = individual_layerFlops / np.sum( individual_layerFlops)
thresholding_layerFlops = np.empty( [no_layers])
for i in range( no_layers):
    thresholding_layerFlops[i] = np.sum( individual_layerFlops[:i+1])
print("Individual:", individual_layerFlops)
print("Cumulative:", thresholding_layerFlops)

predicts_train, ground_truths_train, list_of_intermediate_outs_array_train = train_test_functions.predict( wideresnet101, trainloader, printing=False)
predicts_test, ground_truths_test, list_of_intermediate_outs_array_test = train_test_functions.predict( wideresnet101, testloader, printing=False)

layerOutSizes = [(64,16,16),
                (256,8,8),
                (256,8,8),
                (256,8,8),
                (512,4,4),
                (512,4,4),
                (512,4,4),
                (512,4,4),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (1024,2,2),
                (2048,1,1),
                (2048,1,1),
                (2048,1,1),
                (2048)
                ]
print("""##################################""")

# In[6]:

print("-----------")
print("Computing Class Means on Train Set")
############### For trainset, compute class means and distances to the class means ###############
list_of_layerouts_train = []
list_of_classFeatureMeans_train = []
list_of_dist_of_layerouts_to_class_means_train = []
progresscount = 0
layerlist = list(wideresnet101.children())
for layerNo in layerNumbersToConsider:
    if isinstance( layerlist[layerNo], nn.Sequential):
        layerlist_for_sequential_sublayer = list( layerlist[layerNo].children())
        for i in range( len( layerlist_for_sequential_sublayer)):
            intermediateOut = np.load( filescwd + "\\wideresnet101_" + str(layerNo) + "_" + str(i) + "_train.npy")
            #list_of_layerouts_train.append( intermediateOut)
            classMeans = intermediate_output_functions.classFeatureMeansOf( intermediateOut, ground_truths_train)
            list_of_classFeatureMeans_train.append( classMeans)
            list_of_dist_of_layerouts_to_class_means_train.append( intermediate_output_functions.distanceOf( intermediateOut, classMeans))
            del intermediateOut
            updt(no_layers+1, progresscount + 1, 'Class means after layer (train set)')
            progresscount += 1
    else:
        intermediateOut = np.load( filescwd + "\\wideresnet101_" + str(layerNo) + "_train.npy")
        #list_of_layerouts_train.append( intermediateOut)
        classMeans = intermediate_output_functions.classFeatureMeansOf( intermediateOut, ground_truths_train)
        list_of_classFeatureMeans_train.append( classMeans)
        list_of_dist_of_layerouts_to_class_means_train.append( intermediate_output_functions.distanceOf( intermediateOut, classMeans))
        del intermediateOut
        updt(no_layers+1, progresscount + 1, 'Class means for layer (train set)')
        progresscount += 1

for i in range( len( list_of_classFeatureMeans_train)):
    np.save( filescwd + "\\wideresnet101_list_of_classFeatureMeans_train_" + str(i), list_of_classFeatureMeans_train[i])

for i in range( len( list_of_dist_of_layerouts_to_class_means_train)):
    np.save( filescwd + "\\wideresnet101_dist_of_layerouts_to_class_means_train_" + str(i), list_of_dist_of_layerouts_to_class_means_train[i])


list_of_classFeatureMeans_train = []
for i in range( no_layers):
    list_of_classFeatureMeans_train.append( np.load( filescwd + "\\wideresnet101_list_of_classFeatureMeans_train_" + str(i) + ".npy"))

list_of_dist_of_layerouts_to_class_means_train = []
for i in range( no_layers):
    list_of_dist_of_layerouts_to_class_means_train.append( np.load( filescwd + "\\wideresnet101_dist_of_layerouts_to_class_means_train_" + str(i) + ".npy"))

layers_accuracy_train = np.empty([ no_layers, no_sams_train])
for i in range( no_layers):
    layers_accuracy_train[i] = (np.argmin( list_of_dist_of_layerouts_to_class_means_train[i], axis=1) == ground_truths_train).astype(int)
print("\n-----------")
print("Number of training samples that would have been correctly classified if exited at Layer i (i=0,1,...,34)")
print( np.sum( layers_accuracy_train, axis=1))

meanOfDistsClasses = []
for i in range( len( list_of_dist_of_layerouts_to_class_means_train)):
    dists = list_of_dist_of_layerouts_to_class_means_train[i]
    mean = np.mean( dists, axis=0)
    meanOfDistsClasses.append( mean)
    for j in range( 10):
        dists[:,j] /= mean[j]
    list_of_dist_of_layerouts_to_class_means_train[i] = dists

def softmax( distances):
    #distances: 50000x10
    probs = np.empty(distances.shape)
    for i in range( probs.shape[0]):
        neg_dists = -1*distances[i]
        probs[i] = np.exp( neg_dists) / np.sum( np.exp( neg_dists))
    return probs

softmax_list_of_dist_of_layerouts_to_class_means_train = []
for i in range( len( list_of_dist_of_layerouts_to_class_means_train)):
    softmax_list_of_dist_of_layerouts_to_class_means_train.append( softmax( list_of_dist_of_layerouts_to_class_means_train[i]))
print("""##################################""")

# In[7]:

print("-----------")
progresscount = 0
############### Class means thresholding on trainset ###############
thresholds_flops_accuracies = np.empty( [noExperiments, no_layers-1+2])
thresholds_flops_accuracies = np.load( filescwd + "\\thresholds_flops_accuracies_wideresnet101_softmax.npy")

accuracies_train = np.zeros(noExperiments)
flops_train = np.zeros(noExperiments)
for expNo in range( noExperiments):
    thresholds = thresholds_flops_accuracies[expNo, 2:]#np.random.uniform(0, 1, no_layers-1)

    layers_trust_train = np.empty([ no_layers, no_sams_train])
    for i in range( no_layers - 1):
        layers_trust_train[i] = np.max( softmax_list_of_dist_of_layerouts_to_class_means_train[i], axis=1) > thresholds[i]
    layers_trust_train[no_layers - 1] = np.ones( no_sams_train)

    layers_choice_mask_train = layers_trust_train
    for i in range(1, no_layers):
        for j in range( i):
            layers_choice_mask_train[i] = layers_choice_mask_train[i] * (1 - layers_choice_mask_train[j])
    
    #plt.hist( np.argmax( layers_choice_mask_train, axis=0))
    to_layer_num_train = np.argmax( layers_choice_mask_train, axis=0)
    to_layer_num_train = train_test_functions.to_categorical( to_layer_num_train, no_layers)
    flops = np.sum( to_layer_num_train*thresholding_layerFlops) / no_sams_train
    correct = np.sum( layers_accuracy_train*np.transpose( to_layer_num_train, (1,0))) / no_sams_train
    flops_train[expNo] = flops
    accuracies_train[expNo] = correct
    thresholds_flops_accuracies[expNo, 0:2] = flops_train[expNo], accuracies_train[expNo]
    thresholds_flops_accuracies[expNo, 2:] = thresholds
    updt(noExperiments+1, progresscount + 1, 'Optimizing thresholds')
    progresscount += 1
#thresholds_flops_accuracies = thresholds_flops_accuracies[ thresholds_flops_accuracies[:,0].argsort()]
"""np.save( filescwd + "\\thresholds_flops_accuracies_wideresnet101_softmax.npy", thresholds_flops_accuracies)"""

# convex hull for threshold
points = np.column_stack( (flops_train, accuracies_train))
hull = ConvexHull(points)
descending_side = np.sort(hull.vertices)#[np.argmax( hull.vertices):]

fig = plt.figure()
ax = fig.gca()
ax.grid(True)
plt.plot( flops_train, accuracies_train, 'ro', alpha=0.2)
temp = np.copy(points[descending_side])
temp = temp[temp[:, 0].argsort()]
#plt.plot(points[descending_side,0], points[descending_side,1], 'go-', lw=3)
plt.plot(temp[:,0], temp[:,1], 'go-', lw=3)
plt.ylabel('Accuracy')
plt.xlabel('FLOPs')
plt.title("On training set")
plt.savefig( filescwd + "\\optimizing_thresholds_on_trainset.png")
plt.show()

# In[8]:

print("-----------")
print("Computing Class Means on Test Set")
############### For testset, compute class means and distances to the class means ###############
list_of_layerouts_test = []
list_of_classFeatureMeans_test = list_of_classFeatureMeans_train
list_of_dist_of_layerouts_to_class_means_test = []

progresscount = 0
layerlist = list(wideresnet101.children())
layerNumbersToConsider = [0,4,5,6,7,9]
count = 0
for layerNo in layerNumbersToConsider:
    if isinstance( layerlist[layerNo], nn.Sequential):
        layerlist_for_sequential_sublayer = list( layerlist[layerNo].children())
        for i in range( len( layerlist_for_sequential_sublayer)):
            intermediateOut = np.load( filescwd + "\\wideresnet101_" + str(layerNo) + "_" + str(i) + "_test.npy")
            list_of_layerouts_test.append( intermediateOut)
            list_of_dist_of_layerouts_to_class_means_test.append( intermediate_output_functions.distanceOf( intermediateOut, list_of_classFeatureMeans_test[count]))
            count += 1
            del intermediateOut
            updt(no_layers+1, progresscount + 1, 'Class means for layer (test set)')
            progresscount += 1
    else:
        intermediateOut = np.load( filescwd + "\\wideresnet101_" + str(layerNo) + "_test.npy")
        list_of_layerouts_test.append( intermediateOut)
        list_of_dist_of_layerouts_to_class_means_test.append( intermediate_output_functions.distanceOf( intermediateOut, list_of_classFeatureMeans_test[count]))
        count += 1
        del intermediateOut
        updt(no_layers+1, progresscount + 1, 'Class means after layer (test set)')
        progresscount += 1

layers_accuracy_test = np.empty([ no_layers, no_sams_test])
for i in range( no_layers):
    layers_accuracy_test[i] = (np.argmin( list_of_dist_of_layerouts_to_class_means_test[i], axis=1) == ground_truths_test).astype(int)
print("\n-----------")
print("Number of test samples that would have been correctly classified if exited at Layer i (i=0,1,...,34)")
print( np.sum( layers_accuracy_test, axis=1))

for i in range( len( list_of_dist_of_layerouts_to_class_means_test)):
    dists = list_of_dist_of_layerouts_to_class_means_test[i]
    mean = meanOfDistsClasses[i]
    for j in range( 10):
        dists[:,j] /= mean[j]
    list_of_dist_of_layerouts_to_class_means_test[i] = dists

softmax_list_of_dist_of_layerouts_to_class_means_test = []
for i in range( len( list_of_dist_of_layerouts_to_class_means_test)):
    softmax_list_of_dist_of_layerouts_to_class_means_test.append( softmax( list_of_dist_of_layerouts_to_class_means_test[i]))
print("""##################################""")

# In[9]:

print("-----------")
print("Evaluating the thresholds")
print("-----------")
############### Class means thresholding on testset ###############
accuracies_test = np.zeros(noExperiments)
flops_test = np.zeros(noExperiments)
progresscount = 0
for expNo in range( noExperiments):
    thresholds = thresholds_flops_accuracies[expNo, 2:]
    
    layers_trust_test = np.empty([ no_layers, no_sams_test])
    for i in range( no_layers - 1):
        layers_trust_test[i] = np.max( softmax_list_of_dist_of_layerouts_to_class_means_test[i], axis=1) > thresholds[i]
    layers_trust_test[no_layers - 1] = np.ones( no_sams_test)


    layers_choice_mask_test = layers_trust_test
    for i in range(1, no_layers):
        for j in range( i):
            layers_choice_mask_test[i] = layers_choice_mask_test[i] * (1 - layers_choice_mask_test[j])
    
    #plt.hist( np.argmax( layers_choice_mask_test, axis=0))
    to_layer_num_test = np.argmax( layers_choice_mask_test, axis=0)
    to_layer_num_test = train_test_functions.to_categorical( to_layer_num_test, no_layers)
    flops = np.sum( to_layer_num_test*thresholding_layerFlops) / no_sams_test
    correct = np.sum( layers_accuracy_test*np.transpose( to_layer_num_test, (1,0))) / no_sams_test
    #print( flops, correct)
    flops_test[expNo] = flops
    accuracies_test[expNo] = correct
    updt(noExperiments+1, progresscount + 1, 'Evaluating thresholds')
    progresscount += 1

# convex hull for threshold
points = np.column_stack( (flops_test, accuracies_test))

fig = plt.figure()
ax = fig.gca()
ax.grid(True)
plt.plot( flops_test, accuracies_test, 'ro', alpha=0.2)
temp = np.copy(points[descending_side])
temp = temp[temp[:, 0].argsort()]
#plt.plot( points[descending_side,0], points[descending_side,1], 'go-', lw=3)
plt.plot(temp[:,0], temp[:,1], 'go-', lw=3)
#plt.plot( thisExperimentsResults[:,3], thisExperimentsResults[:,4], 'co')
plt.ylabel('Accuracy')
plt.xlabel('FLOPs')
plt.title("On test set")
plt.savefig( filescwd + "\\thresholds_evaluated_on_testset.png")
plt.show()

print("""##################################""")

# In[10]:

print("-----------")
print("Timing evaluation")
batch_size = 1

############### Dataset ###############
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, num_workers=0)
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=0)

# In[11]:
"""plt.plot( flops_test, accuracies_test, "ro", label="Class Means")

plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('FLOPs')
plt.title("Class Means on CIFAR10 Test Set")
plt.savefig( filescwd + "\\comparison.png")
plt.show()"""

cm = np.column_stack( (flops_test, accuracies_test, thresholds_flops_accuracies[0:noExperiments,2:]))
cm = cm[cm[:, 0].argsort()]

# In[12]:

models_and_classes.TIME = []
train_test_functions.test( wideresnet101, criterion_r, 350, testloader, printing=False)
    
print("Testing on %d images" % no_sams_test)
print("***********************")
print('{:<45s}{:<.2f}{:<13s}{:<.2f}{:<10s}{:<.6f}{:<10s}'.format( "Plain prediction without any early exit" , np.sum( np.argmax( predicts_test, axis=1) == ground_truths_test) / no_sams_test, " Accuracy", thresholding_layerFlops[-1], " FLOPs", np.mean( np.asarray( models_and_classes.TIME)), "s\n"))
print('{:<45s}'.format( "Class Means"))
with open( filescwd + "\\results.txt", "a") as f:
    f.write("Testing on %d images\n" % no_sams_test)
    f.write("***********************\n")
    f.write('{:<45s}{:<.2f}{:<13s}{:<.2f}{:<10s}{:<.6f}{:<10s}'.format( "Plain prediction without any early exit" , np.sum( np.argmax( predicts_test, axis=1) == ground_truths_test) / no_sams_test, " Accuracy", thresholding_layerFlops[-1], " FLOPs", np.mean( np.asarray( models_and_classes.TIME)), "s\n"))
    f.write('{:<45s}'.format( "Class Means\n"))

demo_flops = np.arange( 0.1, 1.1, 0.1)
mean_times_test = []
mean_times_test_branchynet = []
mean_times_test_sd = []
mean_times_test_sastic_tcm = []
for demoflop in demo_flops:
    """
    Choose the closest points to the 0.1, 0.2, ..., 1.0 FLOP points for each method
    """
    
    index_cm = np.argmin( np.abs(cm[:,0] - demoflop))
    
    #######################################################
    wideresnet101.setMeans( list_of_classFeatureMeans_train)
    wideresnet101.setExperimentation( True)
    wideresnet101.setThresholds( thresholds_flops_accuracies[index_cm,2:])

    models_and_classes.TIME = []
    predindexlist_test, lnlist_test = train_test_functions.predict__seq( wideresnet101, testloader)
    mean_times_test.append( np.mean( np.asarray( models_and_classes.TIME)))
    plt.plot( np.asarray( models_and_classes.TIME), "o")
    plt.title("Class Means Inference Time on 1 Sample")
    plt.xlabel("Test Sample No.")
    plt.ylabel("Seconds")
    #plt.ylim([0,0.1])
    plt.savefig( filescwd + "\\cm_time_" + str(int(10*demoflop))+ ".png")
    plt.show()
    plt.plot( lnlist_test, "o")
    plt.title("Class Means Layer Exits")
    plt.xlabel("Test Sample No.")
    plt.ylabel("Layer Number")
    #plt.ylim([0,34])
    plt.savefig( filescwd + "\\cm_exitpoints_" + str(int(10*demoflop))+ ".png")
    plt.show()

    #######################################################
    
    print("-----------------------")
    print('{:<28s}{:<.1f}{:<10s}{:<.2f}{:<13s}{:<.2f}{:<10s}{:<.6f}{:<10s}'.format( "Operating point closest to ", demoflop, " FLOPs: ", cm[index_cm,1], " Accuracy", cm[index_cm,0], " FLOPs", mean_times_test[-1], " s"))
    

    with open( filescwd + "\\results.txt", "a") as f:
        f.write('{:<28s}{:<.1f}{:<10s}{:<.2f}{:<13s}{:<.2f}{:<10s}{:<.6f}{:<10s}'.format( "Operating point closest to ", demoflop, " FLOPs: ", cm[index_cm,1], " Accuracy", cm[index_cm,0], " FLOPs", mean_times_test[-1], " s\n"))


