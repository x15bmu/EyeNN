import math, shutil, os, time
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel

'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


# Change there flags to control what happens.
doLoad = True  # Load checkpoint at the beginning
doTest = False  # Only run test, no training
useCuda = torch.cuda.is_available()

workers = 8
epochs = 100
if useCuda:
    torch.cuda.device_count()*30  # Change if out of cuda memory
else:
    batch_size = 30

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0

CHECKPOINTS_PATH = '.'
INCEPTION_FILENAME = 'checkpoint_inception.pth.tar'
INIT_FILENAME = 'checkpoint.pth.tar'

def main():
    global args, best_prec1, weight_decay, momentum

    inception_exists = os.path.isfile(get_checkpoint_path(INCEPTION_FILENAME))
    model = ITrackerModel(use_pretrained_inception=not inception_exists)
    if useCuda:
        model = torch.nn.DataParallel(model)
    model = try_cuda(model)
    imSize=(224, 224)
    cudnn.benchmark = True   

    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with error %.5f...' % (saved['epoch'], saved['best_prec1']))
            state_dict = model.state_dict()
            state = saved['state_dict']
            del state['eyesFC.0.weight']
            del state['eyesFC.0.bias']
            state_dict.update(state)
            try:
                model.module.load_state_dict(state_dict)
            except:
                model.load_state_dict(state_dict)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!');

    
    dataTrain = ITrackerData(split='train', imSize = imSize)
    dataVal = ITrackerData(split='test', imSize = imSize)

    # Gotta mess with the train.
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    criterion = try_cuda(nn.MSELoss())
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # Quick test
    if doTest:
        validate(val_loader, model, criterion, epoch)
        return

    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
        
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        imFace = try_cuda(torch.autograd.Variable(imFace), async=True)
        imEyeL = try_cuda(torch.autograd.Variable(imEyeL), async=True)
        imEyeR = try_cuda(torch.autograd.Variable(imEyeR), async=True)
        faceGrid = try_cuda(torch.autograd.Variable(faceGrid), async=True)
        gaze = try_cuda(torch.autograd.Variable(gaze), async=True)

        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        losses.update(loss.data[0], imFace.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count += 1

        print('Epoch (train): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))


def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = try_cuda(imFace, async=True)
        imEyeL = try_cuda(imEyeL, async=True)
        imEyeR = try_cuda(imEyeR, async=True)
        faceGrid = try_cuda(faceGrid, async=True)
        gaze = try_cuda(gaze, async=True)

        imFace = torch.autograd.Variable(imFace, volatile=True)
        imEyeL = torch.autograd.Variable(imEyeL, volatile=True)
        imEyeR = torch.autograd.Variable(imEyeR, volatile=True)
        faceGrid = torch.autograd.Variable(faceGrid, volatile=True)
        gaze = torch.autograd.Variable(gaze, volatile=True)

        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data[0], imFace.size(0))
        lossesLin.update(lossLin.data[0], imFace.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch (val): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                    loss=losses, lossLin=lossesLin))

    return lossesLin.avg


def load_checkpoint():
    # First try loading the inception file
    filename = get_checkpoint_path(INCEPTION_FILENAME)
    print('Inception filename: ', filename)

    if not os.path.isfile(filename):
        # If the inception file doesn't exist, try loading the init file
        filename = get_checkpoint_path(INIT_FILENAME)
        print('Init filename: ', filename)

    if not os.path.isfile(filename):
        return None

    if useCuda:
        state = torch.load(filename)
    else:
        state = torch.load(filename, map_location=lambda storage, loc: storage)
    return state


def save_checkpoint(state, is_best):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    best_filename = get_checkpoint_path('best_' + INCEPTION_FILENAME)
    filename = get_checkpoint_path(INCEPTION_FILENAME)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def get_checkpoint_path(filename):
    return os.path.join(CHECKPOINTS_PATH, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def try_cuda(x, **kwargs):
    """
    Calls cuda on x if available
    :param x: The object to call cuda on.
    :return: Returns the object with cuda called if available.
    """
    if useCuda:
        return x.cuda(**kwargs)
    return x


if __name__ == "__main__":
    main()
    print('DONE')
