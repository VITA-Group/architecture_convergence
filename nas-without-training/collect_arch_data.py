import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scores import get_score_func
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from tqdm import trange
from statistics import mean
import time
from utils import add_dropout
from pdb import set_trace as bp


parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/ICML', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--kernel', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--activations', action='store_true')
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, ints = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), ints.detach()


OPS = ['none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'avg_pool_3x3', 'max_pool_3x3']
def edge2value(op_str):
    if 'skip' in op_str or 'pool' in op_str: return 0
    if 'conv' in op_str: return 1
    else: return -1


def find_all_paths(Aff, all_paths, curr_path=[], curr_pos=0, end_pos=5):
    if curr_pos == end_pos:
        all_paths.append(list(curr_path))
        return

    next_nodes = np.where(Aff[curr_pos, (curr_pos+1):] >= 0)[0] + curr_pos + 1
    # print(curr_pos, next_nodes)
    for node in next_nodes:
        curr_path.append(Aff[curr_pos, node])
        find_all_paths(Aff, all_paths, curr_path, node, end_pos)
        curr_path.pop(-1)
    return all_paths


def effective_depth_width(Aff):
    Aff0 = np.array(Aff)
    Aff0 = np.delete(Aff0, 1, 0)
    Aff0 = np.delete(Aff0, 1, 1)
    paths0 = find_all_paths(Aff0, [])
    paths1 = find_all_paths(Aff[1:, 1:], [])
    paths = paths0 + paths1
    depth = 0
    width = 0
    for path in paths:
        depth += np.sum(path)
        width += int(np.sum(path) > 0)
    if depth == 0: return 0, 0
    else:
        depth = depth / len(paths)
        return depth, width/depth


def get_depth_width(genotype):
    Aff = np.ones((7, 7)) * -1 # from x to
    np.fill_diagonal(Aff, 0)
    Aff[2:6, -1] = 0
    for edge_idx, (op, in_node) in enumerate(genotype):
        Aff[in_node, 2 + edge_idx//2] = edge2value(op)  # start from node #2, two edges per node
    depth, width = effective_depth_width(Aff)
    return depth, width * depth, width



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)

times     = []
chosen    = []
acc       = []
val_acc   = []
topscores = []
order_fn = np.nanargmax


if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'


arch_info = {}
for arch in tqdm(range(len(searchspace))):
    uid = searchspace[arch]

    geno = searchspace.get_network_config(uid)['genotype']
    depth, width, width_depth = get_depth_width(geno['normal'])
    depth_r, width_r, width_depth_r = get_depth_width(geno['normal'])
    arch_info[arch] = {
        'normal_depth': depth,
        'normal_width': width_depth,
        'reduce_depth': depth_r,
        'reduce_width': width_depth_r,
        'perf': searchspace.get_final_accuracy(uid, acc_type, False),
        'geno': geno,
        'normal_conv_count': str(geno['normal']).count('conv'),
        'reduce_conv_count': str(geno['reduce']).count('conv'),
    }
    # val_acc.append(searchspace.get_final_accuracy(uid, val_acc_type, args.trainval))

import json
with open('data.json', 'w') as f:
    json.dump(arch_info, f)