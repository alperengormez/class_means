#!/usr/bin/env python
# coding: utf-8


############### Import libraries ###############
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np
import os
from scipy.spatial import ConvexHull
import sys
import argparse
import models
from train_test_functions import train, test, prepare_classmeans, inference_via_classmeans


def parse_option():
    parser = argparse.ArgumentParser('E2CM: Early Exit via Class Means for Efficient Supervised and Unsupervised Learning (IJCNN 2022)')

    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--model', type=str, default='wideresnet101', choices=['wideresnet101'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--ckpt', type=str, default='', help='path to pre-trained model')

    opt = parser.parse_args()
    
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './models/{}_models'.format(opt.dataset)
    opt.model_name = '{}_batchsize_{}'.format( opt.model, opt.batch_size)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_loader(opt):
    print("Dataset: {}".format(opt.dataset))
    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    if opt.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=opt.data_folder, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=opt.data_folder, train=False, download=True, transform=transform_test)
        opt.num_classes = 10
        opt.no_sams_train = len(trainset)
        opt.no_sams_test = len(testset)
    else:
        raise ValueError(opt.dataset)
    
    trainloader = torch.utils.data.DataLoader(trainset, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    testloader = torch.utils.data.DataLoader(testset, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    e2cm_trainloader = torch.utils.data.DataLoader(trainset, 1, shuffle=False, num_workers=opt.num_workers) # batch size 1 for now
    e2cm_testloader = torch.utils.data.DataLoader(testset, 1, shuffle=False, num_workers=opt.num_workers)
    return trainloader, testloader, e2cm_trainloader, e2cm_testloader

def main():
    np.set_printoptions(precision=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
    
    opt = parse_option()
    trainloader, testloader, e2cm_trainloader, e2cm_testloader = set_loader(opt)
    
    ############### Load the model ###############
    model = models.wideresnet101().to(device) # for now
    if opt.ckpt:
        print('Loading pretrained {}'.format(opt.model))
        model.load_state_dict( torch.load( opt.ckpt))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam( model.parameters()) 
    else:
        print('Training the model first')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam( model.parameters())
        model.usual = True
        train( model, optimizer, criterion, 1, opt.epochs, trainloader)
        torch.save( model.state_dict(), opt.save_folder + './ckpt.pth')
    test( model, criterion, opt.epochs, testloader) # (1.018023082613945, 83.17)
    
    
    ############### Prepare class means ###############
    print('Preparing class means using trainset')
    model.usual = False # change forward graph to prepare class means
    model.e2cm_preparation = True
    prepare_classmeans( model, criterion, e2cm_trainloader)
    model.e2cm_preparation = False
    
    # Get distances to class means on trainset, convert to softmax
    # TODO
    
    # Perform thresholding experiments on trainset, get thresholds that give the convex hull
    # TODO
    
    ############### Inference via class means ############### # TODO: + Shallow-Deep Networks
    print('Inference via class means')
    model.e2cm_inference = True
    model.setThresholds( torch.rand( model.total_number_of_exits)) # thresholds should be selected based on the trainset
    inference_via_classmeans( model, criterion, e2cm_testloader)


if __name__ == '__main__':
    main()