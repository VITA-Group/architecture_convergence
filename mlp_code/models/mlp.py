import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, bn=False):
        super(Block, self).__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._bn = bn
        layer = []
        layer.append(nn.Linear(in_dim, out_dim))
        if bn: layer.append(nn.BatchNorm1d(out_dim))
        layer.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*layer)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    def __init__(self, dag, in_dim, width, out_dim=10, bn=False):
        super(MLP, self).__init__()
        self._dag = dag
        self._in_dim = in_dim
        self._widths = width
        self._out_dim = out_dim
        self._bn = bn
        self._stem = Block(in_dim, width, bn)
        self._to_from_layers = nn.ModuleList()
        # list of to_node (list of in_node). 0: broken; 1: skip-connect; 2: linear or conv
        # e.g. "2_02_002" => [[2], [0, 2], [0, 0, 2]]
        dag = [[int(edge) for edge in node] for node in dag.split('_')]
        self._dag = dag
        for _to in range(len(dag)):
            self._to_from_layers.append(nn.ModuleList())
            for _from in range(len(dag[_to])):
                if dag[_to][_from] == 2:
                    self._to_from_layers[-1].append(Block(width, width, bn))
                elif dag[_to][_from] == 1:
                    self._to_from_layers[-1].append(nn.Identity())
                else:
                    self._to_from_layers[-1].append(nn.Module())
        self._readout = nn.Linear(width, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_all=False):
        x = torch.flatten(x, 1)
        x = self._stem(x)
        nodes = [x]
        for _to in range(len(self._dag)):
            _node = []
            for _from in range(len(self._dag[_to])):
                if self._dag[_to][_from] > 0:
                    _node.append(self._to_from_layers[_to][_from](nodes[_from]))
            nodes.append(sum(_node))
        output = self._readout(nodes[-1])
        return output
        # if return_all:
        #     return nodes
        # else:
        #     return nodes[-1]