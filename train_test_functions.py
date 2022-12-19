import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

def train( model, optimizer, criterion, start_epoch, numEpochs, trainloader):
    for epoch in range( start_epoch, start_epoch + numEpochs):
        _ = train_oneepoch( model, optimizer, criterion, epoch, trainloader)

def train_oneepoch( model, optimizer, criterion, epochNo, loader):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for batch_idx, (inputs, targets) in enumerate( loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs, targets)
        loss = criterion( outputs, targets)
        
        optimizer.zero_grad()
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

            outputs = model(inputs, targets)
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



def prepare_classmeans( model, criterion, loader):
    model.eval() # on trainset
    assert model.usual == False and model.e2cm_preparation == True
    
    with torch.no_grad(): # preparation
        for batch_idx, (inputs, targets) in enumerate( loader):
            assert inputs.shape[0] == 1 # batch_size 1 for now
            inputs, targets = inputs.to(device), targets.to(device)
            _ = model(inputs, targets)
            
            print('\rProcessed %d training samples' % (batch_idx+1), end='')
    
    listlist_avgmeter = model.classmeans_at_layers
    
    if not os.path.isdir('./cm'):
        os.makedirs('./cm')
    for i in range( len( listlist_avgmeter)): # num exits
        for j in range( len( listlist_avgmeter[0])): # num classes
            classmean = listlist_avgmeter[i][j].avg
            np.save( './cm/cm_%d%d.npy' % (i,j), classmean.cpu().numpy())


def inference_via_classmeans( model, criterion, loader):
    model.eval()  # on testset
    assert model.usual == False and model.e2cm_inference == True
    correct, total = 0, 0
    sample_exit_layers = []
    sample_predicted_correctly = []
    
    with torch.no_grad(): # testing
        for batch_idx, (inputs, targets) in enumerate( loader):
            assert inputs.shape[0] == 1 # batch_size 1 for now
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, predicted, layerNumber = model(inputs, targets)
            #_, predicted = outputs.max(1) # max_value, the column index
            total += targets.size(0)
            predicted = predicted.to(device)
            correct += predicted.eq(targets).sum().item()
            sample_exit_layers.append( layerNumber)
            sample_predicted_correctly.append( predicted.eq(targets).sum().item())


    test_acc = 100*correct/total
    print('E2CM Test accuracy: %.3f%%, with Correct: %d, Total: %d' % (test_acc, correct, total))
    return test_acc, sample_exit_layers, sample_predicted_correctly