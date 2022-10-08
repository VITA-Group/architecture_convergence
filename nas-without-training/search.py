import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scores import get_score_func
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from statistics import mean, stdev
import time
from utils import add_dropout, init_network


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

parser.add_argument('--dag', action='store_true', help="pre-filtration by dag's depth and width.")

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


def effective_depth_width_201(arch):
    edges = arch.split('+')
    edges = [edge[1:-1].split("|") for edge in edges]
    edge1 = '+'.join([edges[0][0], edges[1][1], edges[2][2]])
    edge2 = '+'.join([edges[0][0], edges[2][1]])
    edge3 = '+'.join([edges[1][0], edges[2][2]])
    edge4 = '+'.join([edges[2][0]])
    depths = []
    width = 0
    for edge in [edge1, edge2, edge3, edge4]:
        if "none" in edge: continue
        if "nor_conv" in edge: width += 1
        depths.append(edge.count("nor_conv"))
    if len(depths) == 0:
        return 0, 0
    else:
        return np.mean(depths), width / np.mean(depths) if np.mean(depths) > 0 else 0


def pre_is_bad(genotype, depth_min, depth_max, width_min, width_max):
    if args.nasspace == "nds_darts":
        Aff = np.ones((7, 7)) * -1 # from x to
        np.fill_diagonal(Aff, 0)
        Aff[2:6, -1] = 0
        for edge_idx, (op, in_node) in enumerate(genotype['normal']):
            Aff[in_node, 2 + edge_idx//2] = edge2value(op)  # start from node #2, two edges per node
        depth, width = effective_depth_width(Aff)
    elif args.nasspace == "nasbench201":
        depth, width = effective_depth_width_201(genotype)
    else:
        return
    if depth < depth_min or depth > depth_max or width < width_min or width > width_max:
        # print(depth, width)
        return True # bad
    else:
        return False # not bad


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)

dset = args.dataset if not (args.trainval and args.dataset == 'cifar10') else 'cifar10-valid'

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


runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    start = time.time()
    indices = np.random.randint(0,len(searchspace), args.n_samples)
    scores = []
    depths_widths = []

    npstate = np.random.get_state()
    ranstate = random.getstate()
    torchstate = torch.random.get_rng_state()
    for arch in indices:
        try:
            uid = searchspace[arch]
            if args.nasspace == "nds_darts":
                _genotype = searchspace.get_network_config(uid)['genotype']
            elif args.nasspace == "nasbench201":
                _genotype = searchspace.api.get_net_config(uid, dset)['arch_str']

            if args.dag:
                # pre-filter by depth, width
                # ranges = [0.5, 1.6, 4, 12] # NDS space
                ranges = [0.99, 2.51, 0.99, 3.01] # 201 space *
                if pre_is_bad(_genotype, *ranges):
                    scores.append(-1)
                    depth, width = effective_depth_width_201(_genotype)
                    depths_widths.append([depth, width])
                    continue

            network = searchspace.get_network(uid)
            network.to(device)
            if args.dropout:
                add_dropout(network, args.sigma)
            if args.init != '':
                init_network(network, args.init)
            if 'hook_' in args.score:
                network.K = np.zeros((args.batch_size, args.batch_size))
                def counting_forward_hook(module, inp, out):
                    try:
                        if not module.visited_backwards:
                            return
                        if isinstance(inp, tuple):
                            inp = inp[0]
                        inp = inp.view(inp.size(0), -1)
                        x = (inp > 0).float()
                        K = x @ x.t()
                        K2 = (1.-x) @ (1.-x.t())
                        network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
                    except:
                        pass

                def counting_backward_hook(module, inp, out):
                    module.visited_backwards = True

                for name, module in network.named_modules():
                    if 'ReLU' in str(type(module)):
                        #hooks[name] = module.register_forward_hook(counting_hook)
                        module.register_forward_hook(counting_forward_hook)
                        module.register_backward_hook(counting_backward_hook)

            random.setstate(ranstate)
            np.random.set_state(npstate)
            torch.set_rng_state(torchstate)

            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)

            if args.kernel:
                s = get_score_func(args.score)(out, labels)
            elif 'hook_' in args.score:
                network(x2.to(device))
                s = get_score_func(args.score)(network.K, target)
            elif args.repeat < args.batch_size:
                s = get_score_func(args.score)(jacobs, labels, args.repeat)
            else:
                s = get_score_func(args.score)(jacobs, labels)

        except Exception as e:
            print(e)
            s = 0.

        scores.append(s)
        depth, width = effective_depth_width_201(_genotype)
        depths_widths.append([depth, width])

    #print(len(scores))
    #print(scores)
    #print(order_fn(scores))

    best_arch = indices[order_fn(scores)]
    uid = searchspace[best_arch]

    if args.nasspace == "nds_darts":
        print(searchspace.get_network_config(uid))
    elif args.nasspace == "nasbench201":
        print(searchspace.api.get_net_config(uid, dset)['arch_str'])

    topscores.append(scores[order_fn(scores)])
    chosen.append(best_arch)
    #acc.append(searchspace.get_accuracy(uid, acc_type, args.trainval))
    acc.append(searchspace.get_final_accuracy(uid, acc_type, False))

    if not args.dataset == 'cifar10' or args.trainval:
        val_acc.append(searchspace.get_final_accuracy(uid, val_acc_type, args.trainval))
    #    val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

    times.append(time.time()-start)
    if len(acc) < 2:
        runs.set_description(f"[{100. * sum(np.array(scores) > 0)/args.n_samples}%] acc: {mean(acc):.2f}% time:{mean(times):.2f}")
    else:
        runs.set_description(f"[{100. * sum(np.array(scores) > 0)/args.n_samples}%] acc: {mean(acc):.2f} ({stdev(acc):.2f}) % time:{mean(times):.2f}")

print(f"Final mean test accuracy: {np.mean(acc)}")
#if len(val_acc) > 1:
#    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

state = {'accs': acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

fname = f"{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{dset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
torch.save(state, fname)
