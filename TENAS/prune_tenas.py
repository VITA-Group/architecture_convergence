import os, sys, time, argparse
import math
import random
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, prepare_logger
from procedures import Linear_Region_Collector, get_ntk_n
from utils import get_model_infos
from log_utils import time_string
from models import get_cell_based_tiny_net, get_search_spaces  # , nas_super_nets
from nas_201_api import NASBench201API as API
from pdb import set_trace as bp


INF = 1000  # used to mark prunned operators


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model


def round_to(number, precision, eps=1e-8):
    # round to significant figure
    dtype = type(number)
    if number == 0:
        return number
    sign = number / abs(number)
    number = abs(number) + eps
    power = math.floor(math.log(number, 10)) + 1
    if dtype == int:
        return int(sign * round(number*10**(-power), precision) * 10**(power))
    else:
        return sign * round(number*10**(-power), precision) * 10**(power)


def find_all_paths_3d_prob(Aff, all_paths, curr_path=[], curr_prob=1., curr_pos=0, end_pos=5):
    """
    Aff: from * to * (-1, 0, 1)
    end_post is 5 not 6: we start from node0 and node1 individually
    """
    if curr_pos == end_pos:
        all_paths.append([list(curr_path), curr_prob])
        return

    next_nodes = np.where(Aff[curr_pos, (curr_pos+1):, 0] < 1)[0] + curr_pos + 1
    for node in next_nodes:
        for v_idx in range(1, 3):
            if Aff[curr_pos, node, v_idx] > 0:
                # curr_path.append(v_idx-1) # v_idx to edge value: 0: -1, 1: 0, 2: 1
                find_all_paths_3d_prob(Aff, all_paths, curr_path+[v_idx-1],
                                       curr_prob*Aff[curr_pos, node, v_idx],
                                       node, end_pos)
                # curr_path.pop(-1)
    return all_paths


def effective_depth_width_prob(Aff, space_name):
    depth_width_probs = []
    #############
    Aff0 = np.array(Aff)
    _paths1 = []
    if space_name == 'darts':
        Aff0 = np.delete(Aff0, 1, 0)
        Aff0 = np.delete(Aff0, 1, 1)
        _paths1 = find_all_paths_3d_prob(Aff[1:, 1:], [])
    # all possible unique end2end paths
    _paths0 = find_all_paths_3d_prob(Aff0, [], end_pos=3 if space_name=="nas-bench-201" else 5)
    _depth_group = 0; _width_group = 0 # for this current mask
    _probs0 = 0
    for _path, _prob in _paths0:
        _probs0 += _prob
    for _path, _prob in _paths0:
        assert np.sum(_path) <= 4
        _depth_group += (np.sum(_path) * _prob / _probs0 * len(_paths0))
        # _depth_group += (np.sum(_path) * _prob / _probs0)
        _width_group += ((_prob) if np.sum(_path) > 0 else 0)
    if space_name == 'darts':
        _probs1 = 0
        for _path, _prob in _paths1:
            _probs1 += _prob
        for _path, _prob in _paths1:
            assert np.sum(_path) <= 4
            _depth_group += (np.sum(_path) * _prob / _probs1 * len(_paths1))
            _width_group += ((_prob) if np.sum(_path) > 0 else 0)
    _depth_group = _depth_group / (len(_paths0) + len(_paths1))
    depth_all = _depth_group
    width_all = (_width_group / _depth_group)
    # return depth_all / num_groups, width_all / num_groups, np.array(depth_width_probs)
    return depth_all, width_all#, np.array(depth_width_probs)


def edge2value(op_str):
    if 'skip' in op_str or 'pool' in op_str: return 0
    if 'conv' in op_str: return 1
    else: return -1


def alpha2affinity(alpha, search_space, space_name):
    """
    Aff: from * to * (-1, 0, 1)
    """
    def set_edge_prob(Aff, from_node, to_node, _alpha):
        edge_prob = np.zeros(3)
        for op, _a in zip(search_space, _alpha):
            if _a >= 0: edge_prob[edge2value(op)+1] += 1
        if sum(edge_prob) > 0:
            edge_prob /= sum(edge_prob)
            Aff[from_node, to_node] = edge_prob
        else:
            Aff[from_node, to_node] = [1, 0, 0] # muted edge

    if space_name == 'nas-bench-201':
        Aff = np.zeros((4, 4, 3))
        Aff[:, :] = [1, 0, 0]
        set_edge_prob(Aff, 0, 1, alpha[0])
        set_edge_prob(Aff, 0, 2, alpha[1])
        set_edge_prob(Aff, 1, 2, alpha[2])
        set_edge_prob(Aff, 0, 3, alpha[3])
        set_edge_prob(Aff, 1, 3, alpha[4])
        set_edge_prob(Aff, 2, 3, alpha[5])
    elif space_name == 'darts':
        Aff = np.zeros((7, 7, 3))
        Aff[:, :] = [1, 0, 0]
        mute_pos = [[0, 1], [1, 0], [0, 6], [1, 6]]
        for pos in mute_pos:
            Aff[pos[0], pos[1]] = [1, 0, 0]
        for i in range(7): # no reverse directions
            for j in range(i):
                Aff[i, j] = [1, 0, 0]
        for i in range(7): # diagonal
            Aff[i, i] = [0, 1, 0]
        Aff[2:6, -1] = [0, 1, 0] # connect to output node
        set_edge_prob(Aff, 0, 2, alpha[0])
        set_edge_prob(Aff, 1, 2, alpha[1])
        set_edge_prob(Aff, 0, 3, alpha[2])
        set_edge_prob(Aff, 1, 3, alpha[3])
        set_edge_prob(Aff, 2, 3, alpha[4])
        set_edge_prob(Aff, 0, 4, alpha[5])
        set_edge_prob(Aff, 1, 4, alpha[6])
        set_edge_prob(Aff, 2, 4, alpha[7])
        set_edge_prob(Aff, 3, 4, alpha[8])
        set_edge_prob(Aff, 0, 5, alpha[9])
        set_edge_prob(Aff, 1, 5, alpha[10])
        set_edge_prob(Aff, 2, 5, alpha[11])
        set_edge_prob(Aff, 3, 5, alpha[12])
        set_edge_prob(Aff, 4, 5, alpha[13])
    return Aff


def prune_func_rank(xargs, arch_parameters, model_config, model_config_thin, loader, lrc_model, search_space, space_name, precision=10,
                    prune_number=1, pre_researve_per_edge=2): # TODO
    # arch_parameters now has three dim: cell_type, edge, op
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    init_model(network_origin, xargs.init)
    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    init_model(network_thin_origin, xargs.init)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)
    network_thin_origin.set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    prune_number = min(prune_number, alpha_active[0][0].sum()-1)  # adjust prune_number based on current remaining ops on each edge
    good_choices_skip_compute = [] # per edge
    pre_researve_per_edge = min(pre_researve_per_edge, alpha_active[0][0].sum().item() - prune_number - 1) # at least rank #prune_number out of #prune_number+1 (per edge) by NTK/LR
    pre_researve_per_edge = int(pre_researve_per_edge)
    if pre_researve_per_edge > 0:
        for idx_ct in range(len(arch_parameters)):
            # cell type (ct): normal or reduce
            for idx_edge in range(len(arch_parameters[idx_ct])):
                # edge
                depth_widths = []; choices = []
                if alpha_active[idx_ct][idx_edge].sum() == 1:
                    # only one op remaining
                    continue
                for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                    # op
                    if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                        # this edge-op not pruned yet
                        _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                        _arch_param[idx_ct][idx_edge, idx_op] = -INF
                        Aff = alpha2affinity(_arch_param[idx_ct], search_space, space_name)
                        _depth, _width = effective_depth_width_prob(Aff, xargs.search_space_name)
                        depth_widths.append([_depth, _width])  # change of ntk
                        choices.append((idx_ct, idx_edge, idx_op))  # change of ntk
                depth_widths = np.array(depth_widths)
                centroid = depth_widths.mean(0, keepdims=True)
                distances = ((depth_widths - centroid) ** 2).sum(1)
                good_choices = np.take(choices, np.argsort(distances)[:pre_researve_per_edge], axis=0).tolist() # ascending sort: keep choices near the centroid
                good_choices_skip_compute += good_choices
    # print(good_choices_skip_compute)

    ntk_all = []  # (ntk, (edge_idx, op_idx))
    regions_all = []  # (regions, (edge_idx, op_idx))
    choice2regions = {}  # (edge_idx, op_idx): regions
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_edge in range(len(arch_parameters[idx_ct])):
            # edge
            if alpha_active[idx_ct][idx_edge].sum() == 1:
                # only one op remaining
                continue
            for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                # op
                if [idx_ct, idx_edge, idx_op] in good_choices_skip_compute:
                    # print("skip:", (idx_ct, idx_edge, idx_op))
                    pbar.update(1)
                    continue
                if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                    # this edge-op not pruned yet
                    _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                    _arch_param[idx_ct][idx_edge, idx_op] = -INF
                    # ##### get ntk (score) ########
                    network = get_cell_based_tiny_net(model_config).cuda().train()
                    network.set_alphas(_arch_param)
                    ntk_delta = []
                    repeat = xargs.repeat
                    for _ in range(repeat):
                        # random reinit
                        init_model(network_origin, xargs.init+"_fanout" if xargs.init.startswith('kaiming') else xargs.init)  # for backward
                        # make sure network_origin and network are identical
                        for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                            param.data.copy_(param_ori.data)
                        network.set_alphas(_arch_param)
                        # NTK cond TODO #########
                        ntk_origin, ntk = get_ntk_n(loader, [network_origin, network], recalbn=0, train_mode=True, num_batch=1)
                        # ####################
                        ntk_delta.append(round((ntk_origin - ntk) / ntk_origin, precision))  # higher the more likely to be prunned
                    ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])  # change of ntk
                    network.zero_grad()
                    network_origin.zero_grad()
                    #############################
                    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                    network_thin_origin.set_alphas(arch_parameters)
                    network_thin_origin.train()
                    network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                    network_thin.set_alphas(_arch_param)
                    network_thin.train()
                    with torch.no_grad():
                        _linear_regions = []
                        repeat = xargs.repeat
                        for _ in range(repeat):
                            # random reinit
                            init_model(network_thin_origin, xargs.init+"_fanin" if xargs.init.startswith('kaiming') else xargs.init)  # for forward
                            # make sure network_thin and network_thin_origin are identical
                            for param_ori, param in zip(network_thin_origin.parameters(), network_thin.parameters()):
                                param.data.copy_(param_ori.data)
                            network_thin.set_alphas(_arch_param)
                            #####
                            lrc_model.reinit(models=[network_thin_origin, network_thin], seed=xargs.rand_seed)
                            _lr, _lr_2 = lrc_model.forward_batch_sample()
                            _linear_regions.append(round((_lr - _lr_2) / _lr, precision))  # change of #Regions, lower the more likely to be prunned
                            lrc_model.clear()
                        linear_regions = np.mean(_linear_regions)
                        regions_all.append([linear_regions, (idx_ct, idx_edge, idx_op)])
                        choice2regions[(idx_ct, idx_edge, idx_op)] = linear_regions
                    #############################
                    torch.cuda.empty_cache()
                    del network_thin
                    del network_thin_origin
                    pbar.update(1)
    ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
    # print("NTK conds:", ntk_all)
    rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]
    for idx, data in enumerate(ntk_all):
        if idx == 0:
            rankings[data[1]] = [idx]
        else:
            if data[0] == ntk_all[idx-1][0]:
                # same ntk as previous
                rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] ]
            else:
                rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] + 1 ]
    regions_all = sorted(regions_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
    # print("#Regions:", regions_all)
    for idx, data in enumerate(regions_all):
        if idx == 0:
            rankings[data[1]].append(idx)
        else:
            if data[0] == regions_all[idx-1][0]:
                # same #Regions as previous
                rankings[data[1]].append(rankings[regions_all[idx-1][1]][1])
            else:
                rankings[data[1]].append(rankings[regions_all[idx-1][1]][1]+1)
    rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    # ascending by sum of two rankings
    rankings_sum = sorted(rankings_list, key=lambda tup: sum(tup[1]), reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    edge2choice = {}  # (cell_idx, edge_idx): list of (cell_idx, edge_idx, op_idx) of length prune_number
    for (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank] in rankings_sum:
        if (cell_idx, edge_idx) not in edge2choice:
            edge2choice[(cell_idx, edge_idx)] = [(cell_idx, edge_idx, op_idx)]
        elif len(edge2choice[(cell_idx, edge_idx)]) < prune_number:
            edge2choice[(cell_idx, edge_idx)].append((cell_idx, edge_idx, op_idx))
    choices_edges = list(edge2choice.values())
    # print("Final Ranking:", rankings_sum)
    # print("Pruning Choices:", choices_edges)
    for choices in choices_edges:
        for (cell_idx, edge_idx, op_idx) in choices:
            arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF

    return arch_parameters, choices_edges


def prune_func_rank_group(xargs, arch_parameters, model_config, model_config_thin, loader, lrc_model, search_space, space_name, edge_groups=[(0, 2), (2, 5), (5, 9), (9, 14)],
                          num_per_group=2, precision=10, pre_researve_per_edge=1): # pre_researve_per_edge can only be 1 or 0
    # arch_parameters now has three dim: cell_type, edge, op
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    init_model(network_origin, xargs.init)
    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    init_model(network_thin_origin, xargs.init)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)
    network_thin_origin.set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    ntk_all = []  # (ntk, (edge_idx, op_idx))
    regions_all = []  # (regions, (edge_idx, op_idx))
    choice2regions = {}  # (edge_idx, op_idx): regions
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    assert edge_groups[-1][1] == len(arch_parameters[0])
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_group in range(len(edge_groups)):
            edge_group = edge_groups[idx_group]
            # print("Pruning cell %s group %s.........."%("normal" if idx_ct == 0 else "reduction", str(edge_group)))
            if edge_group[1] - edge_group[0] <= num_per_group:
                # this group already meets the num_per_group requirement
                pbar.update(1)
                continue

            good_choices_skip_compute = []
            if pre_researve_per_edge > 0:
                depth_widths = []; choices = []
                for idx_edge in range(edge_group[0], edge_group[1]):
                    # edge
                    for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                        # op, actually only one is alive for prune_gropu
                        if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                            # this edge-op not pruned yet
                            _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                            _arch_param[idx_ct][idx_edge, idx_op] = -INF
                            Aff = alpha2affinity(_arch_param[idx_ct], search_space, space_name)
                            _depth, _width = effective_depth_width_prob(Aff, xargs.search_space_name)
                            depth_widths.append([_depth, _width])  # change of ntk
                            choices.append((idx_edge, idx_op))  # change of ntk
                depth_widths = np.array(depth_widths)
                centroid = depth_widths.mean(0, keepdims=True)
                distances = ((depth_widths - centroid) ** 2).sum(1)
                good_choices = np.take(choices, np.argsort(distances)[:pre_researve_per_edge], axis=0).tolist() # ascending sort: keep choices near the centroid
                good_choices_skip_compute += good_choices

            for idx_edge in range(edge_group[0], edge_group[1]):
                # edge
                for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                    # op
                    if [idx_edge, idx_op] in good_choices_skip_compute:
                        pbar.update(1)
                        continue
                    if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                        # this edge-op not pruned yet
                        _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                        _arch_param[idx_ct][idx_edge, idx_op] = -INF
                        # ##### get ntk (score) ########
                        network = get_cell_based_tiny_net(model_config).cuda().train()
                        network.set_alphas(_arch_param)
                        ntk_delta = []
                        repeat = xargs.repeat
                        for _ in range(repeat):
                            # random reinit
                            init_model(network_origin, xargs.init+"_fanout" if xargs.init.startswith('kaiming') else xargs.init)  # for backward
                            # make sure network_origin and network are identical
                            for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                                param.data.copy_(param_ori.data)
                            network.set_alphas(_arch_param)
                            # NTK cond TODO #########
                            ntk_origin, ntk = get_ntk_n(loader, [network_origin, network], recalbn=0, train_mode=True, num_batch=1)
                            # ####################
                            ntk_delta.append(round((ntk_origin - ntk) / ntk_origin, precision))
                        ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])  # change of ntk
                        network.zero_grad()
                        network_origin.zero_grad()
                        #############################
                        network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                        network_thin_origin.set_alphas(arch_parameters)
                        network_thin_origin.train()
                        network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                        network_thin.set_alphas(_arch_param)
                        network_thin.train()
                        with torch.no_grad():
                            _linear_regions = []
                            repeat = xargs.repeat
                            for _ in range(repeat):
                                # random reinit
                                init_model(network_thin_origin, xargs.init+"_fanin" if xargs.init.startswith('kaiming') else xargs.init)  # for forward
                                # make sure network_thin and network_thin_origin are identical
                                for param_ori, param in zip(network_thin_origin.parameters(), network_thin.parameters()):
                                    param.data.copy_(param_ori.data)
                                network_thin.set_alphas(_arch_param)
                                #####
                                lrc_model.reinit(models=[network_thin_origin, network_thin], seed=xargs.rand_seed)
                                _lr, _lr_2 = lrc_model.forward_batch_sample()
                                _linear_regions.append(round((_lr - _lr_2) / _lr, precision))  # change of #Regions
                                lrc_model.clear()
                            linear_regions = np.mean(_linear_regions)
                            regions_all.append([linear_regions, (idx_ct, idx_edge, idx_op)])
                            choice2regions[(idx_ct, idx_edge, idx_op)] = linear_regions
                        #############################
                        torch.cuda.empty_cache()
                        del network_thin
                        del network_thin_origin
                        pbar.update(1)
            # stop and prune this edge group
            ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
            # print("NTK conds:", ntk_all)
            rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]
            for idx, data in enumerate(ntk_all):
                if idx == 0:
                    rankings[data[1]] = [idx]
                else:
                    if data[0] == ntk_all[idx-1][0]:
                        # same ntk as previous
                        rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] ]
                    else:
                        rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] + 1 ]
            regions_all = sorted(regions_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
            # print("#Regions:", regions_all)
            for idx, data in enumerate(regions_all):
                if idx == 0:
                    rankings[data[1]].append(idx)
                else:
                    if data[0] == regions_all[idx-1][0]:
                        # same #Regions as previous
                        rankings[data[1]].append(rankings[regions_all[idx-1][1]][1])
                    else:
                        rankings[data[1]].append(rankings[regions_all[idx-1][1]][1]+1)
            rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
            # ascending by sum of two rankings
            rankings_sum = sorted(rankings_list, key=lambda tup: sum(tup[1]), reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
            choices = [item[0] for item in rankings_sum[:-(num_per_group-pre_researve_per_edge)]] # TODO
            # print("Final Ranking:", rankings_sum)
            # print("Pruning Choices:", choices)
            for (cell_idx, edge_idx, op_idx) in choices:
                arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF
            # reinit
            ntk_all = []  # (ntk, (edge_idx, op_idx))
            regions_all = []  # (regions, (edge_idx, op_idx))
            choice2regions = {}  # (edge_idx, op_idx): regions

    return arch_parameters


def is_single_path(network):
    arch_parameters = network.get_alphas()
    edge_active = torch.cat([(nn.functional.softmax(alpha, 1) > 0.01).float().sum(1) for alpha in arch_parameters], dim=0)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    return True


def main(xargs):
    PID = os.getpid()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)

    ##### config & logging #####
    config = edict()
    config.class_num = class_num
    config.xshape = xshape
    config.batch_size = xargs.batch_size
    xargs.save_dir = xargs.save_dir + \
        "/repeat%d-prunNum%d-prec%d-%s-batch%d"%(
                xargs.repeat, xargs.prune_number, xargs.precision, xargs.init, config["batch_size"]) + \
        "/{:}/seed{:}".format(xargs.timestamp, xargs.rand_seed)
    config.save_dir = xargs.save_dir
    logger = prepare_logger(xargs)
    ###############

    if xargs.dataset != 'imagenet-1k':
        search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/', config.batch_size, xargs.workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=xargs.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

    search_space = get_search_spaces('cell', xargs.search_space_name)
    if xargs.search_space_name == 'nas-bench-201':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                   'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                  })
    elif xargs.search_space_name == 'darts':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                              'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                              'super_type': xargs.super_type,
                              'steps': 4,
                              'multiplier': 4,
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 2, 'use_stem': False, 'stem_multiplier': 1,
                                   'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                   'super_type': xargs.super_type,
                                   'steps': 4,
                                   'multiplier': 4,
                                  })
    network = get_cell_based_tiny_net(model_config)
    logger.log('model-config : {:}'.format(model_config))
    arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, :] = 0

    # TODO Linear_Region_Collector
    lrc_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3, dataset=xargs.dataset, data_path=xargs.data_path, seed=xargs.rand_seed)

    # ### all params trainable (except train_bn) #########################
    flop, param = get_model_infos(network, xshape)
    logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
    if xargs.arch_nas_dataset is None or xargs.search_space_name == 'darts':
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

    network = network.cuda()

    genotypes = {}; genotypes['arch'] = {-1: network.genotype()}

    arch_parameters_history = []
    arch_parameters_history_npy = []
    start_time = time.time()
    epoch = -1

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
    arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
    np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
    while not is_single_path(network):
        epoch += 1
        torch.cuda.empty_cache()
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(xargs.save_dir.split("/")[-6:])))

        arch_parameters, op_pruned = prune_func_rank(xargs, arch_parameters, model_config, model_config_thin, train_loader, lrc_model, search_space, xargs.search_space_name,
                                                     precision=xargs.precision,
                                                     prune_number=xargs.prune_number
                                                     )
        # rebuild supernet
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)

        arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
        genotypes['arch'][epoch] = network.genotype()

        logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    if xargs.search_space_name == 'darts':
        print("===>>> Prune Edge Groups...")
        arch_parameters = prune_func_rank_group(xargs, arch_parameters, model_config, model_config_thin, train_loader, lrc_model, search_space, xargs.search_space_name,
                                                edge_groups=[(0, 2), (2, 5), (5, 9), (9, 14)], num_per_group=2,
                                                precision=xargs.precision,
                                                )
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)
        arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)

    logger.log('<<<--->>> End: {:}'.format(network.genotype()))
    logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    end_time = time.time()
    logger.log('\n' + '-'*100)
    logger.log("Time spent: %d s"%(end_time - start_time))
    # check the performance from the architecture dataset
    if api is not None:
        logger.log('{:}'.format(api.query_by_arch(genotypes['arch'][epoch])))

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("TENAS")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
    parser.add_argument('--search_space_name', type=str, default='nas-bench-201',  help='space of operator candidates: nas-bench-201 or darts.')
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for ntk')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset', type=str, help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')
    parser.add_argument('--prune_number', type=int, default=1, help='number of operator to prune on each edge per round')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK and Regions')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--init', default='kaiming_uniform', help='use gaussian init')
    parser.add_argument('--super_type', type=str, default='basic',  help='type of supernet: basic or nasnet-super')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
