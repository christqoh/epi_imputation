import torch.nn as nn


def create_net(layers: int,
               in_size: int,
               out_size: int,
               hidden_size: int = -1,
               dropout: float = 0.0,
               bias: bool = True,
               in_bias: bool = True,
               activation=nn.ReLU,
               activation_final=None) -> nn.Sequential:

    net = []
    for i in range(layers-1):
        if i == 0 and in_bias == False:
            net.append(linear_xavier_init(in_size, hidden_size, bias=in_bias))
        else:
            net.append(linear_xavier_init(in_size, hidden_size, bias=bias))
        net.append(activation())
        net.append(nn.Dropout(p=dropout))
        in_size = hidden_size

    # add last layer
    if layers == 1:
        hidden_size = in_size

    net.append(linear_xavier_init(hidden_size, out_size, bias=bias))

    if activation_final is not None:
        net.append(activation_final)

    return nn.Sequential(*net)


def conv_xavier_init(in_channels, out_channels, kernel_size: int = 3, stride=1, padding=0, **kwargs):
    _conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, **kwargs)
    nn.init.xavier_uniform_(_conv.weight)
    if _conv.bias is not None:
        nn.init.zeros_(_conv.bias.data)
    return _conv


def linear_xavier_init(in_size, out_size, bias: bool = True):
    _li = nn.Linear(in_size, out_size, bias=bias)
    nn.init.kaiming_normal_(_li.weight)
    if bias:
        nn.init.zeros_(_li.bias)
    return _li


def gru_kaiming_init(in_size, out_size):
    _gru = nn.GRU(in_size, out_size)
    nn.init.kaiming_normal_(_gru.weight)
    return _gru
