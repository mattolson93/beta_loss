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

from utils import CustomLayers, save_openset_all, Losses, calc_auroc

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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
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
parser.add_argument('--alpha', default='0.0', type=float)
parser.add_argument('--beta', default='0.0', type=float)
parser.add_argument('--inlier_class', default='0', type=int)

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
    def __init__(self, data_path, data_split, transform, inlier, inlier_class):
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
                new_labels.append(1 if label == inlier_class else 0)
            self.labels = new_labels
            self.num_inliers = np.array(new_labels).sum()
        else:
            artifact_ind = self.labels == 2
            self.images = self.images[artifact_ind]
            self.labels = np.array([self.num_classes] * len(self.images))
            self.num_inliers = 0

        #import pdb; pdb.set_trace()
        self.tf = transform

    def get_num_inliers(self):
        return self.num_inliers
    
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
    data_path =  './data/mars_data'
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
    openset = args.marstype != "original"

    train_dataset = datasetclass(
        data_path,
        "train",
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), True, args.inlier_class )

    val_dataset = datasetclass(data_path,'val', transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]), True, args.inlier_class )

    test_dataset = datasetclass(data_path,'test', transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]), True, args.inlier_class )

    open_dataset = datasetclass(data_path,'test', transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]), False, args.inlier_class )

    train_inliers, train_length = train_dataset.get_num_inliers(), len(train_dataset)
    val_inliers, val_length = val_dataset.get_num_inliers(), len(val_dataset)
    test_inliers, test_length  = test_dataset.get_num_inliers(), len(test_dataset)
    open_length = len(open_dataset)


    num_classes = 1
    selected_layer = CustomLayers[args.flayer]
    model.fc = selected_layer(model.fc.weight.shape[1],num_classes)
    modes = selected_layer.get_modes()

    file_base = f'{args.alpha}_{args.beta}_{args.marstype}_{args.inlier_class}_{args.lr}'
    out_file_epoch = os.path.join("betaresults","full", file_base+ f'_{args.seed}.csv')
    out_file_last  = os.path.join("betaresults","last", file_base + '.csv')
    out_file_val   = os.path.join("betaresults","val" , file_base + '.csv')

    print(f"freezing the first {args.frozen_layers} layers")
    for i, child in enumerate(model.children()):
        if i < args.frozen_layers:
            for param in child.parameters():
                param.requires_grad = False

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
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
    mylosses = losses()
    criterion = mylosses.getset_alpha_beta(args.alpha, args.beta)

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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    open_loader = torch.utils.data.DataLoader(
        open_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
   
    if args.evaluate:
        exit("evaluate note yet impplemented")
        return

    def save_results(filee, outstring):
        with open(filee, "a") as writer:
            writer.write(outstring+"\n")

    save_results(out_file_epoch, "epoch,val_auc, val_inliers, val_length")
    save_results(out_file_last, "epoch,auc_test, auc_open, auc_both, train_inliers, train_length, test_inliers, test_length, open_length")
    save_results(out_file_val,  "epoch,auc_test, auc_open, auc_both, train_inliers, train_length, test_inliers, test_length, open_length")

    best_auc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)


        # evaluate on validation set
        #import pdb; pdb.set_trace()
        results, val_labs = testData_getinfo(model,val_loader)
        inlier_results  = results[val_labs == 1]
        outlier_results = results[val_labs != 1]

        val_auc =  calc_auroc(inlier_results, outlier_results)
        save_results(out_file_epoch, f"{epoch},{val_auc}, {val_inliers}, {val_length}")

        print("val auc ", val_auc)
        best_auc = max(val_auc, best_auc)
        if best_auc == val_auc: 
            best_val_model = copy.deepcopy(model.state_dict())
    
        if (epoch + 1) % 10 == 0:
            auc_test, auc_open, auc_both = get_auc_results(model, test_loader, open_loader)
            save_results(out_file_last, f"{epoch},{auc_test}, {auc_open}, {auc_both}, {train_inliers}, {train_length}, {test_inliers}, {test_length}, {open_length}")

            cur_model = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_val_model)
            auc_test, auc_open, auc_both = get_auc_results(model, test_loader, open_loader)
            model.load_state_dict(cur_model)
            save_results(out_file_val, f"{epoch},{auc_test}, {auc_open}, {auc_both}, {train_inliers}, {train_length}, {test_inliers}, {test_length}, {open_length}")


def get_auc_results(model, test_data, open_data):
    results, labs = testData_getinfo(model,test_data)
    
    inlier_results  = results[labs == 1]
    outlier_results = results[labs != 1]

    auc_test  = calc_auroc(inlier_results, outlier_results)

    openresults, _ = testData_getinfo(model,open_data)

    auc_open  = calc_auroc(inlier_results, openresults)
    auc_both = calc_auroc(inlier_results, np.concatenate([outlier_results,openresults]))

    return auc_test, auc_open, auc_both

        
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
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
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

        losses.update(loss.item(), images.size(0))

        # measure accuracy and record loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


@torch.no_grad()
def testData_getinfo(model, data_loader):
    model.eval()
    results = []
    labs = []

    for images, labels in tqdm(data_loader):
        model_output = model(images.cuda())
        results.extend(model_output.cpu().numpy())
        labs.extend(labels.numpy())


    return np.array(results).squeeze(),  np.array(labs).squeeze()  


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    return
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

eps = np.finfo(np.float32).eps 
one_m_eps = 1 - eps
logit_range = -np.log(eps)


from torch.cuda.amp import custom_bwd, custom_fwd
class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, max):
        return input.clamp(max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, max)

class losses:
    def __init__(self, do_softplus=False):
        self.softplus = nn.Softplus() #if do_softplus else lambda x: x

        self.do_sigmoid=True
        self.max_clip = torch.tensor(500, dtype=torch.float).cuda()


        self.last_layer = self.last_layer_wrapper
        
    def last_layer_wrapper(self, logits):
        logits =  torch.clamp(logits,min=-logit_range, max=logit_range)
        preds = torch.sigmoid(logits)
        return preds

    def log_loss(self, logits, labels, max_ratio=50):
        preds = self.last_layer(logits)
        
        
        #print(preds[0])
        infunc  = -torch.log(preds)
        outfunc = -torch.log(1.0-preds)

        
        inlier_loss  = (labels     * infunc ).mean(1)
        outlier_loss = ((1-labels) * outfunc).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss

    def getset_alpha_beta(self, alpha, beta):
        

        self.alpha = torch.tensor(alpha, requires_grad=False).cuda()
        self.beta  = torch.tensor(beta, requires_grad=False).cuda()

        if alpha == beta == -42:
            print("doing normal log-loss")

            return self.log_loss

        print(self.alpha, self.beta)
        return self.trapz_loss

    def L1_prime(self, q,a,b):
        return (torch.pow(torch.clamp(q,min=eps),(a-1))) * torch.pow(torch.clamp(1-q,min=eps),b)
    
    def L0_prime(self, q,a,b):
        return (torch.pow(torch.clamp(q,min=eps),a))     * torch.pow(torch.clamp(1-q,min=eps),(b-1))
        
    
    def trapz_loss(self, logits, labels, max_ratio=50):
        #import pdb; pdb.set_trace()

        probs = self.last_layer(logits)
        
        num_tzoids = 10000

        a = self.alpha
        b = self.beta
        
        
        preds = probs.unsqueeze(0).repeat(num_tzoids,1,1)
        
        
        lower_preds = preds * torch.linspace(eps,1.0,num_tzoids).unsqueeze(1).unsqueeze(1).cuda()
        upper_preds = 1- ((1-preds) * torch.linspace(1.0,eps,num_tzoids).unsqueeze(1).unsqueeze(1).cuda())

        
        infunc  = torch.trapz(self.L1_prime(upper_preds,a,b), upper_preds, dim=0)
        outfunc = torch.trapz(self.L0_prime(lower_preds,a,b), lower_preds, dim=0)


        use_logit_in  = labels     * (logits <  logit_range).int()
        use_logit_out = (1-labels) * (logits > -logit_range).int()



        inlier_loss  = dclamp(infunc * use_logit_in, self.max_clip)
        outlier_loss = dclamp(outfunc * use_logit_out, self.max_clip)


        self.inlier_loss = inlier_loss
        self.outlier_loss = outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss



if __name__ == '__main__':
    main()
