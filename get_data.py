#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:08:11 2017

@author: strong
"""

import os 
import torch
#from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets, models
from PIL import Image
import time
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
from torch.autograd import Variable
from densenet import densenet121
import torch.nn.functional as F

from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score


def read_txt(path):
    file_dict={}
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        if line == '\n':
            continue
        new_line = line.split()
        file_dict[new_line[0]]=int(new_line[1])
    return file_dict

def mul_read_txt(path):
    file_dict={}
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        if line == '\n':
            continue
        new_line = line.split()
        label = np.zeros(14)
        for i in range(1,15):
            label[i-1] = int(new_line[i])
        file_dict[new_line[0]]=label
    return file_dict    

class chestDataset(Dataset):
    
    def __init__(self,root_dir,txt_dir,transform=None,is_train=False):
        
        self.root_dir = root_dir
        self.transform = transform
        self.file_dict = {}
        self.n_p_file_list =[]
        self.is_train = is_train
        
        for i in txt_dir:
            tmp = read_txt(i)
            if is_train:
                self.n_p_file_list.append(list(tmp.keys()))
            self.file_dict.update(tmp)
        
        self.img_list = os.listdir(root_dir)
        self.to_read_img_list = []
        
        if is_train:
            self.to_read_n_list = list(set(self.img_list)&set(self.n_p_file_list[0]))
            self.to_read_p_list = list(set(self.img_list)&set(self.n_p_file_list[1]))
            n_p_ratial = int(len(self.to_read_n_list)/len(self.to_read_p_list))
            #print(len(self.to_read_n_list),len(self.to_read_p_list),n_p_ratial)
            
            #simple balance or not
            self.to_read_p_list = self.to_read_p_list*(int(n_p_ratial/4))
            #simple balance or not
            self.to_read_img_list = self.to_read_n_list + self.to_read_p_list
            #print(len(self.to_read_img_list))
        else:
            self.file_dict_keys = list(self.file_dict.keys())
            self.to_read_img_list = list(set(self.file_dict_keys)&set(self.img_list))
            #print(len(self.to_read_img_list))
        #self.inter_file_dict ={}
        # intersect 
        #for i in self.img_list:
        #   if i in self.file_dict:
                #self.inter_file_dict[i]=self.file_dict[i]
        #        self.to_read_img_list.append(i)
        ### 
        if(is_train):
            print('train')
        else:
            print('valid')
        print(len(self.to_read_img_list))
    def __len__(self):
        
        return len(self.to_read_img_list)
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,self.to_read_img_list[idx])
        image = Image.open(img_name).convert('RGB')
        #print(self.to_read_img_list[idx])
        #print(self.to_read_img_list[idx] in self.file_dict)
        label = self.file_dict[self.to_read_img_list[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

root_dir = "/home/strong/chest/images"
#root_dir = "/home/strong/chest/large_image"
train_txt_dir = ['/home/strong/chest/label/new_n_train.txt','/home/strong/chest/label/new_p_train.txt']
valid_txt_dir = ['/home/strong/chest/label/new_n_valid.txt','/home/strong/chest/label/new_p_valid.txt']



train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
valid_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


train_dataset = chestDataset(root_dir,train_txt_dir,train_transform,is_train=True)
valid_dataset = chestDataset(root_dir,valid_txt_dir,valid_transform,is_train=False)

batch_size = 16
n_train = len(read_txt(train_txt_dir[0]))
p_train = len(read_txt(train_txt_dir[1]))
class_sample_count = [10, 1] 
weights = 1.0 / torch.Tensor(class_sample_count)
weights = weights.double()
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)

train_dataloader = DataLoader(train_dataset,batch_size = 16,
                              shuffle=True,num_workers = 4)

#train_dataloader = DataLoader(train_dataset,batch_size = 16,
#                              shuffle=True, num_workers = 4)
valid_dataloader = DataLoader(valid_dataset,batch_size = 16,
                              shuffle=False, num_workers = 4)

dataloaders = {'train':train_dataloader,'val':valid_dataloader}
dataset = {'train':train_dataset, 'val':valid_dataset}
dataset_sizes={}
dataset_sizes['train'] =train_dataset.__len__()
dataset_sizes['val'] =valid_dataset.__len__()


use_gpu = torch.cuda.is_available()
#weight = torch.FloatTensor([1.0/80,79.0/80])

def get_label(tmp_dataset,tmp_dataloader):
    tmp_len = tmp_dataset.__len__()
    label_array = np.zeros((tmp_len,1))
    i = 0 
    for step,(inputs, labels) in enumerate(tmp_dataloader):
        #print(step) 
        #print(labels.shape, i)
        labels = labels.float()
        labels_len = labels.size()[0]
        labels = labels.resize_(labels_len,1)
        #print("ttt")
        np_labels = labels.numpy()
        label_array[i:i+labels_len,0] = np_labels[:,0]
        #print('eee')
        i = i+labels_len
    print(label_array.shape)
    return label_array[0:tmp_len,0]

#valid_label = get_label(valid_dataset,valid_dataloader)

def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    
    since = time.time()
    #state_dict() 
    #eturns a dictionary containing a whole state of the module.
    #Both parameters and persistent buffers (e.g. running averages) are included. 
    #Keys are corresponding parameter and buffer names.
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        
        val_loss = 0
        
        #each epoch has a training and validation phase
        for phase in ['train','val']:
            # torch.optim.lr_scheduler provides several methods to adjust the learning rate 
            #based on the number of epochs. torch.optim.lr_scheduler.ReduceLROnPlateau allows 
            #dynamic learning rate reducing based on some validation measurements.
            if phase =='train':
                #In this fuction scheduler = torch.optim.lr_scheduler.StepLR(optunuzer,step_size,gamma=0.1,last_epoch =-1)
                #Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs
                #when last_epoch = -1, set initial lr as lr
                #http://pytorch.org/docs/0.3.0/optim.html?highlight=scheduler%20step#torch.optim.lr_scheduler.ReduceLROnPlateau
                # optimizer.step() means performs a single optimization step
                
                #didn't change learn rate
                #scheduler.step(val_loss)
                #
                model.train(True)
            else:
                model.train(False)
            
            running_loss = 0.0
            running_corrects = 0
            # get total lebel and prediction to calculate the auc score
            total_label = np.zeros((dataset_sizes[phase],1))
            label_index = 0 
            total_pred = np.zeros((dataset_sizes[phase],1))
            #
            #interate over data 
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                #F.binary_cross_entropy() need input and target are FLoatTensor
                labels = labels.float()
                labels_len = labels.size()[0]
                # resize_ notice there is a underline afret resize
                labels = labels.resize_(labels_len,1)
                
                #
                np_labels = labels.numpy()
                total_label[label_index:label_index+labels_len,0] = np_labels[:,0]
                
                #                
                #wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs , labels = Variable(inputs), Variable(labels)
                
                # zero the parameter gradients 
                optimizer.zero_grad()

                #forward
                outputs = model(inputs)
                
                #add F.sigmoid()
                outputs = F.sigmoid(outputs)
                
                total_pred[label_index:label_index+labels_len,0] = outputs.data.cpu() .numpy()[:,0]
                
                # torch.max() returns the maximum value of each row of the input
                # Tensor in the given dimension dim
                #_ , preds = torch.max(outputs.data,1)
                # For criterion function,the first parameter is outputs
                # the second is labels
                #outputs.data = outputs.data.resize_(outputs.size()[0])
                #labels.data = labels.data.resize_(labels_len)
                class_weight = torch.Tensor(labels_len,1).zero_()
                for i in range(labels_len):
                    if labels.data[i][0]<0.1:
                        class_weight[i][0]= 10.0/83
                    else:
                        class_weight[i][0]= 820.0/83
                class_weight = class_weight.cuda()
                loss = F.binary_cross_entropy(outputs,labels,weight=class_weight)
                #loss = F.binary_cross_entropy(outputs,labels)
                #loss = criterion(outputs,labels)
                # using 0.5 as thresshold
                preds = (outputs.data>0.5).float()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.data[0]
                #label is a Variable
                running_corrects += torch.sum(torch.abs(preds-labels.data)<0.001)
                
                label_index = label_index + labels_len

            auc = roc_auc_score(total_label,total_pred)
            epoch_loss = running_loss / dataset_sizes[phase] 
            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == 'val':
                val_loss = epoch_loss
                scheduler.step(val_loss)
            print('{} Loss: {:.4f} Acc: {:.4f} auc: {:.4f}'.format(phase,epoch_loss,epoch_acc, auc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f} m {:.0f}'.format(time_elapsed // 60, time_elapsed % 60))
    
    #load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

