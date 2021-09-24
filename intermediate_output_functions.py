import torch
import numpy as np
num_classes = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.backends.cudnn as cudnn
if device == 'cuda':
    cudnn.benchmark = True
def getIntermediateOut( listOfLayers, loader, last=False):
    intermediate_outs = []
    initFlag = True
    interFlag = False
    
    with torch.no_grad(): # testing
        for batch_idx, (inputs, targets) in enumerate( loader):
            inputs, targets = inputs.to(device), targets.to(device) # gpu
            
            inputToLayer = inputs
            for k,layer in enumerate( listOfLayers):
                if last and k == len(listOfLayers)-1:
                    inputToLayer = torch.flatten( inputToLayer, 1)
                out = layer( inputToLayer)
                inputToLayer = out
            
            intermediate_outs.append( out)
    
    
    intermediate_outs_array = intermediate_outs[0].cpu().numpy()
    for i in range( 1, len( intermediate_outs)):
        intermediate_outs_array = np.concatenate( (intermediate_outs_array, intermediate_outs[i].cpu().numpy()), axis=0)
    
    return intermediate_outs_array

def distanceOf( features, classFeatureMeans):
    """
    features: no_samples x num_filters x height x width
    classFeatureMeans: num_classes x num_filters x height x width
    
    returns: distance of each sample to each classFeatureMean -> no_samples x num_classes
    """
    dist_of_samples_to_class_means = np.empty( (features.shape[0], classFeatureMeans.shape[0]))
    
    b = classFeatureMeans.reshape( num_classes, -1)
    for i in range( features.shape[0]):
        a = features[i].reshape( -1)
        dist_of_samples_to_class_means[i] = np.linalg.norm( a - b, axis=1)
    
    return dist_of_samples_to_class_means

def classFeatureMeansOf( layerOutput, ground_truths):
    classFeatureMeans = np.zeros( (num_classes,) + layerOutput.shape[1:])
    for i in range( num_classes):
        classFeatureMeans[i] = np.mean( layerOutput[ np.where( ground_truths == i)[0]], axis=0)
    return classFeatureMeans