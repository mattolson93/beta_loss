import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import numpy as np

import copy
import matplotlib
matplotlib.use('agg')
import seaborn as sns

from utils import CustomLayers, save_openset_all, Losses

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default="./data/open_imagenet/", metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--frozen-layers', default=7, type=int,
                    metavar='N', help='how many layers to not update (default: 9)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--loss', default='ce', type=str, metavar='PATH',
                    help='kliep or crossentropy (default: ce)')
parser.add_argument('--flayer', default='NormalLinear', type=str, 
                    help='')
parser.add_argument('--marstype', default='artifact', type=str, 
                    help='')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0




class MarsData(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split, transform):
        data_file = data_split +'-set-v2.1.txt'
        self.images = np.loadtxt(os.path.join(data_path,data_file), usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(os.path.join(data_path,data_file), usecols=[1], dtype=np.int)
        self.data_path = os.path.join(data_path,"images")

        self.num_classes = 19
        
        self.tf = transform
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))

        lab = torch.tensor(self.labels[index])
        return img, lab
    
    def __len__(self):
        return len(self.images)

class MarsDataSolDay100(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split, transform, inlier):
        data_file = data_split +'-set-v2.1.txt'
        self.images = np.loadtxt(os.path.join(data_path,data_file), usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(os.path.join(data_path,data_file), usecols=[1], dtype=np.int)
        self.data_path = os.path.join(data_path,"images")
        self.num_classes = 19


        inlier_labels = self.get_inlier_labels(data_path)
        
        label_map = [-1] * self.num_classes
        for i, c in enumerate(inlier_labels):
            label_map[c] = i
        label_map = np.array(label_map)

        outlier_labels = np.setxor1d(inlier_labels, np.arange(self.num_classes))

        y = inlier_labels if inlier else outlier_labels
        inds = np.in1d(self.labels, y)


        self.images = self.images[inds]
        self.labels = self.labels[inds]

        if inlier: self.labels = label_map[self.labels]
        self.num_classes = np.unique(self.labels).shape[0]
        
        self.tf = transform

    def get_inlier_labels(self, data_path, data_file="train-set-v2.1.txt", day=100):
        image_paths = np.loadtxt(os.path.join(data_path,data_file), usecols=[0], dtype=np.str)
        labels = np.loadtxt(os.path.join(data_path,data_file), usecols=[1], dtype=np.int)
        
        first_x_days = np.array([int(f[:4]) <= day for f in image_paths])

        seen_labels = np.unique(labels[first_x_days])
        return seen_labels

    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))

        lab = torch.tensor(self.labels[index])
        return img, lab
    
    def __len__(self):
        return len(self.images)

class MarsDataSolDay50(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split, transform, inlier):
        data_file = data_split +'-set-v2.1.txt'
        self.images = np.loadtxt(os.path.join(data_path,data_file), usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(os.path.join(data_path,data_file), usecols=[1], dtype=np.int)
        self.data_path = os.path.join(data_path,"images")
        self.num_classes = 19


        inlier_labels = self.get_inlier_labels(data_path)
        
        label_map = [-1] * self.num_classes
        for i, c in enumerate(inlier_labels):
            label_map[c] = i
        label_map = np.array(label_map)

        outlier_labels = np.setxor1d(inlier_labels, np.arange(self.num_classes))

        y = inlier_labels if inlier else outlier_labels
        inds = np.in1d(self.labels, y)


        self.images = self.images[inds]
        self.labels = self.labels[inds]

        if inlier: self.labels = label_map[self.labels]
        self.num_classes = np.unique(self.labels).shape[0]
        
        self.tf = transform

    def get_inlier_labels(self, data_path, data_file="train-set-v2.1.txt", day=50):
        image_paths = np.loadtxt(os.path.join(data_path,data_file), usecols=[0], dtype=np.str)
        labels = np.loadtxt(os.path.join(data_path,data_file), usecols=[1], dtype=np.int)
        
        first_x_days = np.array([int(f[:4]) <= day for f in image_paths])

        seen_labels = np.unique(labels[first_x_days])
        return seen_labels

    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))

        lab = torch.tensor(self.labels[index])
        return img, lab
    
    def __len__(self):
        return len(self.images)


class MarsDataArtifactOpen(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split, transform, inlier):
        data_file = data_split +'-set-v2.1.txt'
        self.images = np.loadtxt(os.path.join(data_path,data_file), usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(os.path.join(data_path,data_file), usecols=[1], dtype=np.int)
        self.data_path = os.path.join(data_path,"images")
        self.num_classes = 18

        if inlier:
            notartifact_ind = self.labels != 2
            self.images = self.images[notartifact_ind]
            self.labels = self.labels[notartifact_ind]
            new_labels = []
            for label in self.labels:
                if label == 18:
                    label = 2
                new_labels.append(label)
            self.labels = new_labels
        else:
            artifact_ind = self.labels == 2
            self.images = self.images[artifact_ind]
            self.labels = np.array([self.num_classes] * len(self.images))



        self.tf = transform
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))

        lab = torch.tensor(self.labels[index])
        return img, lab
    
    def __len__(self):
        return len(self.images)

class MarsHiRiseEjectaSpiderOpen(torch.utils.data.Dataset):
    def __init__(self, data_path, data_split, transform, inlier):
        data_file = 'labels-map-proj_v3_2_train_val_test.txt'
        images = np.loadtxt(os.path.join(data_path,data_file), usecols=[0], dtype=np.str)
        labels = np.loadtxt(os.path.join(data_path,data_file), usecols=[1], dtype=np.int)
        datatype = np.loadtxt(os.path.join(data_path,data_file), usecols=[2], dtype=np.str)
        

        self.images = []
        self.labels = []
        for i, l ,d in zip(images, labels,datatype):
            if d == data_split:
                self.images.append(i)
                self.labels.append(l)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        self.data_path = os.path.join(data_path,"map-proj-v3_2")
        self.num_classes = 6

        if inlier:
            normal_ind = (self.labels != 5) & (self.labels != 7) 
            self.images = self.images[normal_ind]
            self.labels = self.labels[normal_ind]
            new_labels = []
            for label in self.labels:
                if label == 6:
                    label = 5
                new_labels.append(label)
            self.labels = new_labels
        else:
            outlier_ind = (self.labels == 5) | (self.labels == 7) 
            self.images = self.images[outlier_ind]
            self.labels = np.array([self.num_classes] * len(self.images))
            

        self.tf = transform
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index]))).repeat(3,1,1)
        lab = torch.tensor(self.labels[index])
        return img, lab
    
    def __len__(self):
        return len(self.images)




def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def make_plot2d(x, y, title, outfile):
    sns.set(rc={'figure.figsize':(25,25)})
    palette = sns.color_palette("bright", len(set(y)))
    myplot = sns.scatterplot(x[:,0], x[:,1], hue=y, legend='full', palette=palette).set_title(title)
    fig = myplot.get_figure()
    fig.savefig(outfile)
    fig.clf()

@torch.no_grad()
def get_labels_and_embeddings(model, dataset):

    activation = {}
    def get_activation(name):
        # name is the dictionary key that will be used to store the activation
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.avgpool.register_forward_hook(get_activation('model.avgpool'))


    model.eval()
    device = next(model.parameters()).device
    zs = torch.empty(0)

    labs = []
    for images, labels in tqdm(dataset):
        _ = model(images.to(device))
        z = activation['model.avgpool']
        labs.extend(labels.numpy())

        #z = model.z 
        zs = torch.cat([zs, z.cpu().squeeze()], dim = 0)


    return np.array(labs).squeeze(), zs.numpy()

import umap

def latent_analysis(model, traindata, valdata, valopen_data, testdata, opendata, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    train_labs, train_zs = get_labels_and_embeddings(model, traindata)
    val_labs,     val_zs = get_labels_and_embeddings(model, valdata)
    _,     valopen_zs = get_labels_and_embeddings(model, valopen_data)
    test_labs,   test_zs = get_labels_and_embeddings(model, testdata)
    _ ,          open_zs = get_labels_and_embeddings(model, opendata)
    valopen_labs = np.array([model.num_classes] * valopen_zs.shape[0])
    open_labs = np.array([model.num_classes] * open_zs.shape[0])
    trans = umap.UMAP(n_neighbors=15,verbose=True).fit(train_zs)
    embs_train      = trans.transform(train_zs)
    embs_test       = trans.transform(test_zs)
    embs_open       = trans.transform(open_zs)
    eval_labs      = np.concatenate([test_labs, open_labs])
    embs_eval_zs   = np.concatenate([embs_test, embs_open])

    np.save(os.path.join(out_dir,"train_zs.npy"),train_zs)
    np.save(os.path.join(out_dir,"val_zs.npy"),val_zs)
    np.save(os.path.join(out_dir,"valopen_zs.npy"),valopen_zs)
    np.save(os.path.join(out_dir,"test_zs.npy"),test_zs)
    np.save(os.path.join(out_dir,"open_zs.npy"),open_zs)

    np.save(os.path.join(out_dir,"train_labs.npy"),train_labs)
    np.save(os.path.join(out_dir,"val_labs.npy"),val_labs)
    np.save(os.path.join(out_dir,"valopen_labs.npy"),valopen_labs)
    np.save(os.path.join(out_dir,"test_labs.npy"),test_labs)
    np.save(os.path.join(out_dir,"open_labs.npy"),open_labs)

    make_plot2d(embs_train, train_labs, "train embs", os.path.join(out_dir,"train.png"))
    make_plot2d(embs_eval_zs, eval_labs, "test/open embs (open is k+1 )", os.path.join(out_dir,"eval.png"))
    make_plot2d(embs_eval_zs, np.concatenate([np.array([0]*test_labs.shape[0]), np.array([1]*open_labs.shape[0])]), "test/open embs ", os.path.join(out_dir,"eval2.png"))

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)

    # Data loading code
    trainfile =  'train-set-v2.1.txt'
    valfile   =  'val-set-v2.1.txt'
    testfile  =  'test-set-v2.1.txt'
    data_path =  'mars_data'
    #openfile = os.path.join(args.data, 'val_outlier')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if args.marstype == 'hirise': normalize = transforms.Normalize(mean=[0.5], std=[0.225])

    print("loading training dataset")

    mars_dataset_map = {
        "original": MarsData,
        "artifact": MarsDataArtifactOpen,
        "solday"  :  MarsDataSolDay100,
        "solday50":  MarsDataSolDay50,
        "hirise"  :  MarsHiRiseEjectaSpiderOpen,
    }
    datasetclass = mars_dataset_map[args.marstype]

    train_dataset = datasetclass(
        data_path,
        "train",
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), True )

    val_dataset = datasetclass(data_path,'val', transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]), True )

    valopen_dataset = datasetclass(data_path,'val', transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]), False )

    test_dataset = datasetclass(data_path,'test', transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]), True )

    open_dataset = datasetclass(data_path,'test', transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]), False )



    num_classes = train_dataset.num_classes 
    selected_layer = CustomLayers[args.flayer]
    model.num_classes = num_classes
    model.fc = selected_layer(model.fc.weight.shape[1],num_classes)
    modes = selected_layer.get_modes()




    outdir_base = f'{args.loss}_{args.flayer}_{args.marstype}_{args.seed}'

    print(f"freezing the first {args.frozen_layers} layers")
    for i, child in enumerate(model.children()):
        if i < args.frozen_layers:
            for param in child.parameters():
                param.requires_grad = False

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    my_losses = Losses()
    criterion = my_losses.get_loss_dict()[args.loss]

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                #momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    print("loading val dataset")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    valopen_loader = torch.utils.data.DataLoader(
        valopen_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    open_loader = torch.utils.data.DataLoader(
                    open_dataset,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)


    latent_analysis(model, train_loader, val_loader, valopen_loader, test_loader, open_loader, outdir_base+"_pretrained")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

    
    latent_analysis(model, train_loader, val_loader,valopen_loader, test_loader, open_loader, outdir_base+"_fullytrained")

        
        
    '''save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)'''


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, target) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            if type(output) is tuple:
                output = output[0]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    return
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
