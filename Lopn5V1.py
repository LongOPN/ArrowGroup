"""Forked from Video clip order prediction and edited!!!."""

import os
import math
import itertools
import argparse
import time
import random
import torchvision.utils as tor
import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from LopnmodelV1 import LOPN
from r3d import R3DNet
from r21d import R2Plus1DNet
from PIL import Image
from shutil import copyfile
from torchvision import transforms
import cv2
import numpy
import matplotlib.pyplot as plt
from  ucf101 import UCF101FOPDataset

from tensorboardX import SummaryWriter
 
from PIL import Image

from shutil import copyfile

def order_class_index(order):
    """Return the index of the order in its full permutation.
    
    Args:
        order (tensor): e.g. [0,1,2]
    """
    classes = list(itertools.permutations(list(range(len(order)))))
    m=classes.index(tuple(order.tolist()))
    #-----------------------our Grouping -------------
    if(m>=60):
      m=119-m
    return m  
    #means
    # if(m==119):
    #   m=0
    # # elif(m==118):
    # #   m=1
    # # elif(m==117):
    # #   m=2
    # # elif(m==116):
    # #   m=3
    # # elif(m==115):
    # #   m=4
    # # elif(m==114):
    # #   m=5
    # # elif(m==113):
    # #   m=6
    # # elif(m==112):
    # #   m=7
    # # elif(m==111):
    # #   m=8
    # # elif(m==110):
    # #   m=9
    # # elif(m==109):
    # #   m=10
    # # elif(m==108):
    # #   m=11  
    # # elif(m==107):
    # #   m=12
    # # elif(m==106):
    # #   m=13
    # # elif(m==105):
    # #   m=14
    # # elif(m==104):
    # #   m=15
    # # elif(m==103):
    # #   m=16
    # # elif(m==102):
    # #   m=17
    # # elif(m==101):
    # #   m=18
    # # elif(m==100):
    # #   m=19
    # # elif(m==99):
    # #   m=20
    # # elif(m==98):
    # #   m=21    
    # # elif(m==97):
    # #   m=22
    # # elif(m==96):
    # #   m=23
    # # elif(m==95):
    # #   m=24
    # # elif(m==94):
    # #   m=25
    # # elif(m==93):
    # #   m=26
    # # elif(m==92):
    # #   m=27
    # # elif(m==91):
    # #   m=28
    # # elif(m==90):
    # #   m=29
    # # elif(m==89):
    # #   m=30
    # # elif(m==88):
    # #   m=31    

    # # elif(m==87):
    # #   m=32
    # # elif(m==86):
    # #   m=33
    # # elif(m==85):
    # #   m=34
    # # elif(m==84):
    # #   m=35
    # # elif(m==83):
    # #   m=36
    # # elif(m==82):
    # #   m=37
    # # elif(m==81):
    # #   m=38
    # # elif(m==80):
    # #   m=39
    # # elif(m==79):
    # #   m=40
    # # elif(m==78):
    # #   m=41    
    # # elif(m==77):
    # #   m=42
    # # elif(m==76):
    # #   m=43
    # # elif(m==75):
    # #   m=44
    # # elif(m==74):
    # #   m=45
    # # elif(m==73):
    # #   m=46
    # # elif(m==72):
    # #   m=47
    # # elif(m==71):
    # #   m=48
    # # elif(m==70):
    # #   m=49
    # # elif(m==69):
    # #   m=50
    # # elif(m==68):
    # #   m=51  
    # # elif(m==67):
    # #   m=52
    # # elif(m==66):
    # #   m=53
    # # elif(m==65):
    # #   m=54
    # # elif(m==64):
    # #   m=55
    # # elif(m==63):
    # #   m=56
    # # elif(m==62):
    # #   m=57
    # # elif(m==61):
    # #   m=58
    # # elif(m==60):
    # #   m=59
   

    #return classes.index(tuple(order.tolist()))


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        #print('a112')
        # compute loss and acc
        running_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        #print('a113')
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            #print('a115')
            avg_acc = correct / (args.pf * args.bs)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch-1)*len(train_dataloader) + i
            writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0
            #print('a3')
    # summary params and grads per eopch
    for name, param in model.named_parameters():
        writer.add_histogram('params/{}'.format(name), param, epoch)
        writer.add_histogram('grads/{}'.format(name), param.grad, epoch)


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss,avg_acc


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        pts=[]
        targets=[]
        outputs=[]

        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        if(i%10==0):
          torch.set_printoptions(profile="full")
          print('targ is',targets)
          print('pts  is',pts)
          #print('out  is',outputs)
          torch.set_printoptions(profile="default") 
           
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss

def parse_args():

    
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--it', type=int, default=8, help='interval')
    parser.add_argument('--tl', type=int, default=5, help='tuple length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str,default='/content/drive/My Drive/opnmodel/vcopopntl5half/', help='log directory')
    parser.add_argument('--ckpt', type=str,help='checkpoint path')#default= 
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=900, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=778, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=70, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=40, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print("device",device)
    print(torch.cuda.get_device_properties(device))
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    if args.model == 'c3d':
        base = C3D(with_classifier=False)
    elif args.model == 'r3d':
        base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    elif args.model == 'AlexNet':
        base = AlexNet(with_classifier=False, return_conv=False)
    elif args.model == 'r21d':   
        base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    #opn = OPN(base_network=base, feature_size=256, tuple_len=args.tl).to(device)
    opn = LOPN(base_network=base, feature_size=256, tuple_len=args.tl).to(device)   

    if args.mode == 'train':  ########### Train #############
        if args.ckpt:  # resume training
            #print('happent')
            opn.load_state_dict(torch.load(args.ckpt))
            log_dir = os.path.dirname(args.ckpt)
            #print('1')
        else:
            if args.desp:
                exp_name = '{}_cl{}_it{}_tl{}_{}_{}'.format(args.model, args.cl, args.it, args.tl, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_cl{}_it{}_tl{}_{}'.format(args.model, args.cl, args.it, args.tl, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)
            #print('exp',exp_name)
        writer = SummaryWriter(log_dir)

        train_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            #transforms.RandomCrop(20,30),
            transforms.ToTensor()
        ])
        #train_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, True, train_transforms)
        train_dataset = UCF101FOPDataset('data/ucf101/ucf16',  args.it, args.tl, True, train_transforms)

        #val_size = 800
        #print('t1',train_dataset)
        train_size = int(0.9 * len(train_dataset))
        test_size = len(train_dataset) - train_size        
        train_dataset, val_dataset = random_split(train_dataset, (train_size, test_size))
        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)
        #print('train_dataloader',train_dataloader)
        if args.ckpt:
            pass
        else:
            # save graph and clips_order samples
            for data in train_dataloader:
                #print('data1 ')
                tuple_frame, tuple_orders = data

                for i in range(args.tl):
                        writer.add_images('train/tuple_frame', tuple_frame[:, i, :, :, :], i)
                        writer.add_text('train/tuple_orders', str(tuple_orders[:, i].tolist()), i)
                #tuple_clips = tuple_clips.to(device)
                tuple_frame = tuple_frame.to(device)
                print('tps',tuple_frame.size())
                #writer.add_graph(opn, tuple_frame,verbose=False)
                writer.add_graph(opn, tuple_frame)
                #writer.flush()
                #writer.close()
                break
            # save init params at step 0
            for name, param in opn.named_parameters():
                writer.add_histogram('params/{}'.format(name), param, 0)

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(opn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

        prev_best_val_loss = float('inf')
        prev_best_val_acc=0
        val_loss = float('inf')
        val_acc=0
        prev_best_model_path = None
        model_path_prev=None
        print('here i am')
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            if args.start_epoch==epoch:
              time_start = time.time()
            train(args, opn, criterion, optimizer, device, train_dataloader, writer, epoch)
            if args.start_epoch==epoch:
              print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            # scheduler.step(val_loss)         
            val_loss,val_acc = validate(args, opn, criterion, device, val_dataloader, writer, epoch)
            # scheduler.step(val_loss)         
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)            # save model every 1 epoches
            if epoch % 1 == 0:
                torch.save(opn.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
                if model_path_prev:
                  open(model_path_prev, 'w').close() #overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                  os.remove(model_path_prev) #delete the blank file from google drive will move the file to bin instead
                model_path_prev = os.path.join(log_dir, 'model_{}.pt'.format(epoch))
   
                #model_path2 = os.path.join('/content/drive/My Drive/opnmodel/cpu128/', 'model_{}.pt'.format(epoch))
                #model_path = os.path.join(log_dir, 'model_{}.pt'.format(epoch))
                #copyfile(model_path, model_path2)
                # Create & upload a text file.
                #uploaded = drive.CreateFile({'title': 'Sample file.txt'})
                #model_path.SetContentString('Sample upload file content')

            # save model for the best val
            if val_loss < prev_best_val_loss:
                model_path = os.path.join(log_dir, 'best_model_{}_{:.3f}__{:.3f}.pt'.format(epoch,val_loss,val_acc))
                torch.save(opn.state_dict(), model_path)
                prev_best_val_loss = val_loss
                #if prev_best_model_path:
                    #os.remove(prev_best_model_path)
                prev_best_model_path = model_path
                if(val_acc > prev_best_val_acc):
                  prev_best_val_acc = val_acc
            elif(val_acc > prev_best_val_acc):
                model_path = os.path.join(log_dir, 'best_model_{}_{:.3f}_{:.3f}.pt'.format(epoch,val_loss,val_acc))
                torch.save(opn.state_dict(), model_path)
                prev_best_val_acc = val_acc
                #if prev_best_model_path:
                    #os.remove(prev_best_model_path)
                prev_best_model_path = model_path

    elif args.mode == 'test':  ########### Test #############
        #model1 = torch.load('model_1.pt')
        #model.load_state_dict(model1)
        opn.load_state_dict(torch.load(args.ckpt))
        #opn.load_state_dict(model1)
        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            #transforms.RandomCrop(20,30),
            transforms.ToTensor()
        ])
        #test_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, False, test_transforms)
        test_dataset = UCF101FOPDataset('data/ucf101/ucf16',  args.it, args.tl, False, test_transforms)

        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, opn, criterion, device, test_dataloader)