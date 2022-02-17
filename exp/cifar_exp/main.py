import os, sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

# config
from config.parser import cifar_parser

import numpy as np
import torch
use_cuda = torch.cuda.is_available()
import torch.nn as nn
import torch.optim as optim

# network
from networks.cifar_resnet import resnet20, resnet32, resnet56

# utils
from utils.datasetmanager import cifar10_dataloader
from utils.img_transform import cifar10_transform
from utils.pytorch_test import get_device, fix_random
from utils.write_op import create_muldir
from utils.meter_op import AverageMeter, accuracy


def test(model, criterion, loader):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        test_loss = losses.avg 
        test_acc = top1.avg
        return test_loss, test_acc

def train(model, criterion, optimizer, loader):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (losses.avg, top1.avg)

def str2bool(v):
    if isinstance(v, bool):
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = cifar_parser()
    parser.add_argument("--mode", default = 'flops_2D', help="prunning mode", type = str)
    parser.add_argument("--timelimit", default = 5000, help="timelimit for solving this", type = int)
    parser.add_argument("--baseseed", default=1, help="baseline seed", type=int)
    parser.add_argument("--lr",default=0.01,help="intial_lr",type=float)
    parser.add_argument("--decay",default=0.0005,help="weight_decay",type=float)
    parser.add_argument("--momentum",default=0.9,help="momentum",type=float)
    parser.add_argument("--epochs",default=200,help="epochs",type=int)
    parser.add_argument("--schedule",default=[60,120,160],help="decrease lr at these epochs.",nargs="+",type=int)
    parser.add_argument("--lr_decay",default=0.2,help="lr is multiplied by lr_decay on schedule.",nargs="+",type=float)
    parser.add_argument("--target_c",default=0.6,help="remaining ratio of target resource",type=float)
    parser.add_argument("--use_cplex", help='use_cplex', default=False, type=str2bool)
    parser.add_argument("--gamma",default=1.0,help="ratio of amplification in 4D",type=float)
    parser.add_argument("--arch",default="resnet20",help="the name of network",type=str)
    parser.add_argument("--rseed",default=0,help="random seed",type=int)

    args = parser.parse_args() # parameter required for model
    
    state = {k: v for k, v in args._get_kwargs()}

    fix_random(state['rseed'])
    
    LOADPATH = 'pretrained_networks/{}/network_{}.pt'.format(state['arch'], state['baseseed'])
    METADIR = 'meta/'
    create_muldir(METADIR)
    
    # Data Load
    transform_train, transform_test = cifar10_transform()
    trainloader, testloader = cifar10_dataloader(transform_train=transform_train, transform_test=transform_test, train_batch=state['nbatch'], test_batch=state['nbatch'], shuffle_train=True, shuffle_test=False)
    
    # init model
    if state['arch'] == "resnet20": 
        model = resnet20()
        sidx, eidx = (0, 19)
    elif state['arch'] == "resnet32":
        model = resnet32()
        sidx, eidx = (0, 31)
    elif state['arch'] == "resnet56": 
        model = resnet56()
        sidx, eidx = (0, 55)
    maskman = model.maskman

    # load model
    load_dict = torch.load(LOADPATH, map_location='cpu')
    model.load_from_state_dict(source_dict=load_dict)
    model = model.to(get_device())

    # cross entropy, optimizer
    optimizer = torch.optim.SGD(model.get_trainable_parameters(), state['lr'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Prune model    
    if state['mode']=='flops_2D':
        maskman.prun_flops_with_weight(
                start_idx=sidx,
                end_idx=eidx,
                flops_c=state['target_c'],
                use_cplex=state['use_cplex'],
                timelimit=state['timelimit'],
                path=METADIR,
                debug=False)
    elif state['mode']=='flops_4D':
        maskman.prun_flops_with_weight_4D(
                start_idx=sidx,
                end_idx=eidx,
                flops_c=state['target_c'],
                gamma=state['gamma'],
                use_cplex=state['use_cplex'],
                timelimit=state['timelimit'],
                path=METADIR,
                debug=False)
    elif state['mode']=='mem_2D':
        maskman.prun_mem_with_weight(
                start_idx=sidx,
                end_idx=eidx,
                mem_c=state['target_c'],
                use_cplex=state['use_cplex'],
                timelimit=state['timelimit'],
                path=METADIR,
                debug=False)
    elif state['mode']=='mem_4D':
        maskman.prun_mem_with_weight_4D(
                start_idx=sidx,
                end_idx=eidx,
                mem_c=state['target_c'],
                gamma=state['gamma'],
                use_cplex=state['use_cplex'],
                timelimit=state['timelimit'],
                path=METADIR,
                debug=False)
    
    print("Model pruning ends")
    content = maskman.evaluate_from_uvq_and_mask()
    
    # After pruning
    test_loss, test_acc = test(model=model, criterion=criterion, loader=testloader)
    print("   After Pruning")
    print("   test_loss : {}, test_acc : {}".format(test_loss, test_acc))

    best_acc = test_acc  # best test accuracy
    # Train and val
    for epoch in range(state['epochs']):
        if epoch in state['schedule']:
            state['lr'] *= state['lr_decay']
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']

        train_loss, train_acc = train(model=model, criterion=criterion, optimizer=optimizer, loader=trainloader)
        test_loss, test_acc = test(model=model, criterion=criterion, loader=testloader)

        print("Epoch({}/{}) lr : {}, train_loss : {}, test_loss : {}, train_acc : {}, test_acc : {}".format(epoch, state['epochs'], state['lr'], train_loss, test_loss, train_acc, test_acc))

        if test_acc>best_acc:
            best_acc = max(test_acc, best_acc)

if __name__=='__main__':
    main()
