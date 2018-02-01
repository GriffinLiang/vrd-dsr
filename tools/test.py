import os
import os.path as osp
import sys
import cPickle
import argparse
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.init

import _init_paths
from lib.nets.Vrd_Model import Vrd_Model
import lib.network as network
from lib.data_layers.vrd_data_layer import VrdDataLayer
from lib.model import test_pre_net, test_rel_net

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch VRD Training')
    parser.add_argument('--name', dest='name',
                        help='experiment name',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='ds_name',
                        help='dataset name',
                        default=None, type=str)    
    parser.add_argument('--model_type', dest='model_type',
                        help='model type: RANK_IM_ALL, RANK_IM, LOC',
                        default=None, type=str)
    parser.add_argument('--no_so', dest='use_so', action='store_false')
    parser.set_defaults(use_so=True)
    parser.add_argument('--no_obj', dest='use_obj', action='store_false')
    parser.set_defaults(use_obj=True)
    parser.add_argument('--no_obj_prior', dest='use_obj_prior', action='store_false')
    parser.set_defaults(use_obj_prior=True)        
    parser.add_argument('--loc_type', default=0, type=int)
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    global args
    args = parse_args()
    args.proposal = '../data/vrd/proposal.pkl'
    args.resume = '../models/%s/epoch_%d_checkpoint.pth.tar'%(args.name, args.epochs-1)
    print args
    print 'Evaluating...'
    # Data
    test_data_layer = VrdDataLayer(args.ds_name, 'test', model_type = args.model_type)
    args.num_relations = test_data_layer._num_relations
    args.num_classes = test_data_layer._num_classes
    # Model
    net = Vrd_Model(args)
    net.cuda()
    if osp.isfile(args.resume):
        print("=> loading model '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])
        headers = ["Epoch","Pre R@50", "ZS", "R@100", "ZS", "Rel R@50", "ZS", "R@100", "ZS"]
        res = []
        res.append((args.epochs-1,) + test_pre_net(net, args)+test_rel_net(net, args))
        print tabulate(res, headers)
    else:
        print "=> no model found at '{}'".format(args.resume)
