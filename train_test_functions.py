import torch
import torch.nn.functional as F
import numpy as np
#from .utils import load_state_dict_from_url
from typing import Any
from typing import Type, Any, Callable, Union, List, Optional
import torch.backends.cudnn as cudnn

NORMAL = 0
BRANCHYNET = 1
SHALLOWDEEP = 2
CM_AND_SD = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
batch_size = 256

def fit( model, optimizer, criterion, my_lr_scheduler, start_epoch, numEpochs, trainloader, testloader, state, flops, distancestoclassmeans):
    for epoch in range( start_epoch, start_epoch + numEpochs):
        if state == NORMAL:
            _ = train( model, optimizer, criterion, epoch, trainloader)
        else:
            _ = train_withIC( model, optimizer, criterion, epoch, trainloader, state, flops, distancestoclassmeans)
        if my_lr_scheduler != None:
            my_lr_scheduler.step()

def train( model, optimizer, criterion, epochNo, loader):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for batch_idx, (inputs, targets) in enumerate( loader):
        inputs, targets = inputs.to(device), targets.to(device) # gpu
        optimizer.zero_grad() # zero the parameter gradients
        
        # forward + backward + optimize
        outputs = model(inputs)
        if isinstance( outputs, list): # if we take intermediate outputs too
            outputs = outputs[-1]
        
        loss = criterion( outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = train_loss / (batch_idx+1)
    train_acc = 100*correct/total
    print('[Epoch no: %d] Training loss: %.5f  |  Training accuracy: %.3f%%, with Correct: %d, Total: %d' % (epochNo, train_loss, train_acc, correct, total))    
    return (train_loss, train_acc)

def test( model, criterion, epochNo, loader, printing=True):
    model.eval() 
    test_loss, correct, total = 0, 0, 0
    
    with torch.no_grad(): # testing
        for batch_idx, (inputs, targets) in enumerate( loader):
            inputs, targets = inputs.to(device), targets.to(device) # gpu

            outputs = model(inputs)
            if isinstance( outputs, list): # if we take intermediate outputs too
                outputs = outputs[-1]
            loss = criterion( outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1) # max_value, the column index
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / (batch_idx + 1)
    test_acc = 100*correct/total
    if printing:
        print('[Trained for %d epochs]. Test loss: %.5f  |  Test accuracy: %.3f%%, with Correct: %d, Total: %d' % (epochNo, test_loss, test_acc, correct, total))
    return (test_loss, test_acc)

def predict( model, loader, printing=True):
    model.eval() 
    correct = 0
    total = 0
    input_list = []
    target_list = []
    preds_list = []
    list_of_intermediate_outs = [] # elements are a list
    initFlag = True
    interFlag = False

    
    with torch.no_grad(): # testing
        for batch_idx, (inputs, targets) in enumerate( loader):
            inputs, targets = inputs.to(device), targets.to(device) # gpu
            input_list.append( inputs)
            target_list.append( targets)
            
            outputs = model(inputs)
            if isinstance( outputs, list):
                if initFlag:
                    initFlag = False
                    for i in range( len( outputs)):
                        list_of_intermediate_outs.append( [])
                interFlag = True
                for i,layerOutput in enumerate( outputs):
                    if i == len( outputs) - 1:
                        list_of_intermediate_outs[i].append( F.softmax( layerOutput, dim=1))
                    else:
                        list_of_intermediate_outs[i].append( layerOutput)
                
                outputs = outputs[-1]
            out_softmax = F.softmax( outputs, dim=1)
            preds_list.append( out_softmax)
            
            _, predicted = outputs.max(1) # max_value, the column index
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    pred_acc = 100*correct/total
    if printing:
        print('Predict accuracy: %.3f%%, with Correct: %d, Total: %d' % (pred_acc, correct, total))
    
    input_array = input_list[0].cpu().numpy()
    for i in range(1, len(input_list)):
        input_array = np.concatenate( (input_array, input_list[i].cpu().numpy()), axis=0)
    
    target_array = target_list[0].cpu().numpy()
    for i in range( 1, len( target_list)):
        target_array = np.concatenate( (target_array, target_list[i].cpu().numpy()), axis=0)
    
    preds_array = preds_list[0].cpu().numpy()
    for i in range( 1, len( preds_list)):
        preds_array = np.concatenate( (preds_array, preds_list[i].cpu().numpy()), axis=0)
    
    list_of_intermediate_outs_array = []
    if interFlag:
        for list_layerOutput in list_of_intermediate_outs:
            layer_array = list_layerOutput[0].cpu().numpy()
            for i in range( 1, len( list_layerOutput)):
                layer_array = np.concatenate( (layer_array, list_layerOutput[i].cpu().numpy()), axis=0)
            list_of_intermediate_outs_array.append( layer_array)
    
    return (preds_array, target_array, list_of_intermediate_outs_array)

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def train_withIC( model, optimizer, criterion, epochNo, loader, state, flops, distancestoclassmeans):
    model.train()
    train_loss, correct, total = 0, 0, 0
    corrects = [0] * len(flops)
    totals = [0] * len(flops)
    
    for batch_idx, (inputs, targets) in enumerate( loader):
        inputs, targets = inputs.to(device), targets.to(device) # gpu
        optimizer.zero_grad() # zero the parameter gradients
        
        if state == CM_AND_SD:
            distances = distancestoclassmeans[0:7,batch_idx*batch_size:(batch_idx+1)*batch_size,:] # 6 x batch_size x 10
            distances = torch.from_numpy( distances).type(torch.FloatTensor).to( device)
            outputs = model(inputs, distances)
        else:
            outputs = model(inputs)
        loss = criterion( outputs[-1], targets)
        for i in range( len(outputs) - 1):
            if state == BRANCHYNET or state == CM_AND_SD:
                loss = loss + (flops[i])*criterion( outputs[i], targets)
            elif state == SHALLOWDEEP:
                loss = loss + (flops[i]*min(epochNo,50)/50)*criterion( outputs[i], targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs[-1].max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        for i in range( len(outputs)):
            _, predicted = outputs[i].max(1)
            totals[i] += targets.size(0)
            corrects[i] += predicted.eq(targets).sum().item() 
    
    train_loss = train_loss / (batch_idx+1)
    train_acc = 100*correct/total
    print('[Epoch no: %d] Training loss: %.5f  |  Training accuracy: %.3f%%, with Correct: %d, Total: %d' % (epochNo, train_loss, train_acc, correct, total))    
    print('[Epoch no:', epochNo, "] Corrects: ", corrects)
    return (train_loss, train_acc)

def test___( model, criterion, epochNo, loader, distancestoclassmeans):
    model.eval() 
    test_loss, correct, total = 0, 0, 0
    
    with torch.no_grad(): # testing
        for batch_idx, (inputs, targets) in enumerate( loader):
            inputs, targets = inputs.to(device), targets.to(device) # gpu
            
            distances = distancestoclassmeans[0:7,batch_idx*batch_size:(batch_idx+1)*batch_size,:] # 6 x batch_size x 10
            distances = torch.from_numpy( distances).type(torch.FloatTensor).to( device)
            
            outputs = model(inputs, distances)
            if isinstance( outputs, list): # if we take intermediate outputs too
                outputs = outputs[-1]
            loss = criterion( outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1) # max_value, the column index
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / (batch_idx + 1)
    test_acc = 100*correct/total
    print('[Epoch no: %d]. Test loss: %.5f  |  Test accuracy: %.3f%%, with Correct: %d, Total: %d' % (epochNo, test_loss, test_acc, correct, total))
    return (test_loss, test_acc)

def predict___( model, loader, distancestoclassmeans):
    model.eval() 
    correct = 0
    total = 0
    input_list = []
    target_list = []
    preds_list = []
    list_of_intermediate_outs = [] # elements are a list
    initFlag = True
    interFlag = False

    
    with torch.no_grad(): # testing
        for batch_idx, (inputs, targets) in enumerate( loader):
            inputs, targets = inputs.to(device), targets.to(device) # gpu
            input_list.append( inputs)
            target_list.append( targets)
            
            distances = distancestoclassmeans[0:7,batch_idx*batch_size:(batch_idx+1)*batch_size,:] # 6 x batch_size x 10
            distances = torch.from_numpy( distances).type(torch.FloatTensor).to( device)
            
            outputs = model(inputs, distances)
            if isinstance( outputs, list):
                if initFlag:
                    initFlag = False
                    for i in range( len( outputs)):
                        list_of_intermediate_outs.append( [])
                interFlag = True
                for i,layerOutput in enumerate( outputs):
                    if i == len( outputs) - 1:
                        list_of_intermediate_outs[i].append( F.softmax( layerOutput, dim=1))
                    else:
                        list_of_intermediate_outs[i].append( F.softmax(layerOutput, dim=1))
                
                outputs = outputs[-1]
            out_softmax = F.softmax( outputs, dim=1)
            preds_list.append( out_softmax)
            
            _, predicted = outputs.max(1) # max_value, the column index
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    pred_acc = 100*correct/total
    print('Predict accuracy: %.3f%%, with Correct: %d, Total: %d' % (pred_acc, correct, total))
    
    input_array = input_list[0].cpu().numpy()
    for i in range(1, len(input_list)):
        input_array = np.concatenate( (input_array, input_list[i].cpu().numpy()), axis=0)
    
    target_array = target_list[0].cpu().numpy()
    for i in range( 1, len( target_list)):
        target_array = np.concatenate( (target_array, target_list[i].cpu().numpy()), axis=0)
    
    preds_array = preds_list[0].cpu().numpy()
    for i in range( 1, len( preds_list)):
        preds_array = np.concatenate( (preds_array, preds_list[i].cpu().numpy()), axis=0)
    
    list_of_intermediate_outs_array = []
    if interFlag:
        for list_layerOutput in list_of_intermediate_outs:
            layer_array = list_layerOutput[0].cpu().numpy()
            for i in range( 1, len( list_layerOutput)):
                layer_array = np.concatenate( (layer_array, list_layerOutput[i].cpu().numpy()), axis=0)
            list_of_intermediate_outs_array.append( layer_array)
    
    return (preds_array, target_array, list_of_intermediate_outs_array)

def predict__seq( model, loader):
    model.eval() 
    correct = 0
    total = 0
    input_list = []
    target_list = []
    preds_list = []
    list_of_intermediate_outs = [] # elements are a list
    initFlag = True
    interFlag = False
    predindexlist = []
    lnlist = []
    
    with torch.no_grad(): # testing
        for batch_idx, (inputs, targets) in enumerate( loader):
            inputs, targets = inputs.to(device), targets.to(device) # gpu
            input_list.append( inputs)
            target_list.append( targets)
            
            outputs, predindex, ln = model(inputs)
            predindexlist.append( predindex)
            lnlist.append( ln)
            if isinstance( outputs, list):
                if initFlag:
                    initFlag = False
                    for i in range( len( outputs)):
                        list_of_intermediate_outs.append( [])
                interFlag = True
                for i,layerOutput in enumerate( outputs):
                    if i == len( outputs) - 1:
                        list_of_intermediate_outs[i].append( F.softmax( layerOutput, dim=1))
                    else:
                        list_of_intermediate_outs[i].append( layerOutput)
                
                outputs = outputs[-1]
            out_softmax = F.softmax( outputs, dim=1)
            preds_list.append( out_softmax)
            
            _, predicted = outputs.max(1) # max_value, the column index
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    
    return predindexlist, lnlist

def softmax__seq( distances):
    #distances: 50000x10
    probs = np.empty(distances.shape)
    for i in range( probs.shape[0]):
        neg_dists = -1*distances[i]
        probs[i] = np.exp( neg_dists) / np.sum( np.exp( neg_dists))
    return probs

def distanceOf__seq( features, classFeatureMeans):
    """
    features: no_samples x num_filters x height x width
    classFeatureMeans: num_classes x num_filters x height x width
    
    returns: distance of each sample to each classFeatureMean -> no_samples x num_classes
    """
    dist_of_samples_to_class_means = np.empty( (features.shape[0], classFeatureMeans.shape[0]))
    num_classes = 10
    b = classFeatureMeans.reshape( num_classes, -1)
    for i in range( features.shape[0]):
        a = features[i].reshape( -1)
        dist_of_samples_to_class_means[i] = np.linalg.norm( a.cpu() - b, axis=1)
    
    return dist_of_samples_to_class_means

