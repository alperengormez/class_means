import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def softmax_from_dist( distances):
    # distances: N x num_classes
    distances = distances / torch.max(distances) # TODO: normalize using all distances in the trainset
    probs = torch.zeros(distances.shape)
    for i in range( probs.shape[0]):
        neg_dists = -1*distances[i]
        probs[i] = torch.exp( neg_dists) / torch.sum( torch.exp( neg_dists))
    return probs

def distance_to_classmeans( x, classMeans):
    """
    x: layer output for one sample
    classMeans: [ mean tensors], len is num_classes
    
    returns: distance of sample to each class mean -> 1 x num_classes
    """
    assert x.shape[0] == 1 # batch_size 1 for now
    num_classes = len( classMeans)
    dist_of_sample_to_class_means = torch.zeros( (x.shape[0], num_classes))
    for i in range( num_classes):
        #print(x.shape, classMeans[i].avg.shape)
        dist_of_sample_to_class_means[:,i] = torch.linalg.norm( x - classMeans[i].avg)
    return dist_of_sample_to_class_means