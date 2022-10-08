'''
    main process for retrain a subnetwork from beginning
'''
from distutils.command.build import build
import os
import pdb
import time
import pickle
import random
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from thop import profile

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# from models import model_dict
import models
from dataset import cifar10_dataloaders, cifar100_dataloaders, svhn_dataloaders, mnist_dataloaders
from logger import prepare_seed, prepare_logger
from utils import save_checkpoint, warmup_lr, AverageMeter, accuracy

from pdb import set_trace as bp

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training Subnetworks')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
# parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='mlp', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)') 
parser.add_argument('--dag', type=str, default='2_02_002', help='from-to edges separated by underscore. 0: broken edge; 1: skip-connect; 2: linear or conv') 
parser.add_argument('--imagenet_arch', action="store_true", help="back to imagenet architecture (conv1, maxpool)")
parser.add_argument('--width', type=int, default=2048, help='hidden width')
parser.add_argument('--bn', action="store_true", help="use BN")
# parser.add_argument('--gap', action="store_true", help="use GAP in ConvNet")
parser.add_argument('--fc', action="store_true", help="use FC in ConvNet")

##################################### General setting ############################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--inference', action="store_true", help="testing")
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='./experiment', type=str)
parser.add_argument('--exp_name', help='additional names for experiment', default='', type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0., type=float, help='momentum') # 0.9
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay') # 1e-4
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
# parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')
parser.add_argument('--decreasing_lr', default=None, help='decreasing strategy')
parser.add_argument('--save_ckeckpoint_freq', default=-1, type=int, help='save intermediate checkpoint per epoch')
parser.add_argument('--pretrained', action="store_true", help="use official pretrained checkpoint")


best_acc = 0
args = parser.parse_args()

if args.seed is None:
    args.seed = random.randint(0, 999)
prepare_seed(args.seed)

PID = os.getpid()
print("<< ============== PID = %d ============== >>"%(PID))


def build_model(args, classes=10, dummy_shape=(3, 32, 32)):
    if args.arch.startswith("resnet"):
        model = models.__dict__[args.arch](pretrained=args.pretrained, num_classes=classes, imagenet=args.imagenet_arch)
    elif args.arch == "mlp":
        model = models.mlp(args.dag, np.prod(dummy_shape), args.width, out_dim=classes, bn=args.bn)
    elif args.arch == "convnet":
        model = models.convnet(args.dag, dummy_shape[0], args.width, out_dim=classes, bn=args.bn, fc=args.fc)
    return model


def main():
    global args, best_acc

    job_name = "{dataset}-{arch}{width}{bn}{dag}-LR{lr:.7f}-BS{batch_size}-Epoch{epoch}{exp_name}".format(
        dataset=args.dataset, arch=args.arch, width=args.width, bn=".BN" if args.bn else "",
        dag="_"+args.dag if args.arch in ["mlp", "convnet"] else "", lr=args.lr, batch_size=args.batch_size, epoch=args.epochs,
        exp_name="" if args.exp_name == "" else "-"+args.exp_name)
    timestamp = "{:}".format(time.strftime("%m%d%H%M%S", time.gmtime()))
    args.save_dir = os.path.join(args.save_dir, job_name, "seed%d_"%args.seed+timestamp)
    logger = prepare_logger(args)

    torch.cuda.set_device(int(args.gpu))
    if not args.inference:
        os.makedirs(args.save_dir, exist_ok=True)

    # prepare dataset
    NUM_VAL_IMAGE = 50
    c_in = 3
    if args.dataset == 'cifar10':
        classes = 10
        dummy_shape = (3, 32, 32)
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset == 'cifar100':
        classes = 100
        dummy_shape = (3, 32, 32)
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers,
                                                                     val_size=NUM_VAL_IMAGE * classes, flatten=args.arch == "mlp")
    elif args.dataset == 'svhn':
        classes = 10
        dummy_shape = (3, 32, 32)
        train_loader, val_loader, test_loader = svhn_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset == 'mnist':
        c_in = 1
        classes = 10
        dummy_shape = (1, 28, 28)
        train_loader, val_loader, test_loader = mnist_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset is None:
        pass
    else:
        raise ValueError('Dataset not supprot yet!')

    model = build_model(args, classes, dummy_shape)
    logger.log(str(model))
    macs, params = profile(model, inputs=(torch.randn(1, *dummy_shape), ), verbose=True)
    logger.log("param = %d, flops = %d"%(params, macs))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(','))) if args.decreasing_lr else None

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = None
    if decreasing_lr:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    if args.inference:
        # test
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint)

        test_acc = validate(test_loader, model, criterion, 0)
        logger.log('* Test Accuracy = {}'.format(test_acc))
        return 0

    if args.resume:
        logger.log('resume from checkpoint {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = None
        if decreasing_lr:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
            scheduler.load_state_dict(checkpoint['scheduler'])

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.log('loading from epoch: ',start_epoch, 'best_acc=', best_acc)

    else:
        all_result = {}
        all_result['train_acc'] = []
        all_result['test_acc'] = []
        all_result['val_acc'] = []
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, split="Val")
        test_loss, test_acc = validate(test_loader, model, criterion, epoch, split="Test")
        logger.writer.add_scalar("train/loss", train_loss, epoch)
        logger.writer.add_scalar("train/accuracy", train_acc, epoch)
        logger.writer.add_scalar("validation/loss", val_loss, epoch)
        logger.writer.add_scalar("validation/accuracy", val_acc, epoch)
        logger.writer.add_scalar("test/loss", test_loss, epoch)
        logger.writer.add_scalar("test/accuracy", test_acc, epoch)
        logger.log("Epoch {} Train {:.2f} Validation {:.2f} Test {:.2f}".format(epoch, train_acc, val_acc, test_acc))

        if scheduler: scheduler.step()

        all_result['train_acc'].append(train_acc)
        all_result['val_acc'].append(val_acc)
        all_result['test_acc'].append(test_acc)

        # remember best prec@1 and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        checkpoint = {
            'result': all_result,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None
        }
        save_checkpoint(checkpoint, is_best=is_best, save_path=args.save_dir)
        if args.save_ckeckpoint_freq > 0 and epoch % args.save_ckeckpoint_freq == 0:
            save_checkpoint(checkpoint, is_best=False, save_path=args.save_dir, filename="epoch%d.pth.tar"%epoch)

        # plot training curve
        plt.plot(all_result['train_acc'], label='train_acc')
        plt.plot(all_result['val_acc'], label='val_acc')
        plt.plot(all_result['test_acc'], label='test_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    #report result
    val_pick_best_epoch = np.argmax(np.array(all_result['val_acc']))
    logger.log('* best accuracy = {}, Epoch = {}'.format(all_result['test_acc'][val_pick_best_epoch], val_pick_best_epoch+1))


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    pbar = tqdm(train_loader, position=0, leave=True)
    for i, (image, target) in enumerate(pbar):
        if epoch < args.warmup:
            warmup_lr(args.warmup, args.lr, epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target)[0]
        # top1.update(prec1.item(), image.size(0))
        prec1, gt_num = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], gt_num[0])

        losses.update(loss.item(), image.size(0))

        # pbar.set_description("Epoch{} Train | Loss {:.4f} | Accuracy {:.3f} | LR {:.4f}".format(epoch, float(losses.avg), float(top1.avg), optimizer.state_dict()['param_groups'][0]['lr']))
        pbar.set_description("Epoch{} Train | Loss {:.4f} | Accuracy {:.3f} | LR {:.4f}".format(epoch, float(losses.avg), float(top1.vec2sca_avg), optimizer.state_dict()['param_groups'][0]['lr']))

    # return float(losses.avg), float(top1.avg)
    return float(losses.avg), float(top1.vec2sca_avg)


def validate(val_loader, model, criterion, epoch, split="Test"):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    pbar = tqdm(val_loader, position=0, leave=True)
    for i, (image, target) in enumerate(pbar):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target)[0]
        # top1.update(prec1.item(), image.size(0))
        prec1, gt_num = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], gt_num[0])
        losses.update(loss.item(), image.size(0))

        # pbar.set_description("Epoch{} {} | Loss {:.4f} | Accuracy {:.3f}".format(epoch, split, float(losses.avg), float(top1.avg)))
        pbar.set_description("Epoch{} {} | Loss {:.4f} | Accuracy {:.3f}".format(epoch, split, float(losses.avg), float(top1.vec2sca_avg)))

    # return float(losses.avg), float(top1.avg)
    return float(losses.avg), float(top1.vec2sca_avg)


if __name__ == '__main__':
    main()