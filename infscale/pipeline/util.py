import torch
import torch.distributed as dist


def list2csvcell(l):
    if len(l) == 0:
        return "0"

    s = str(l[0])
    for i in range(1, len(l)):
        s += "-" + str(l[i])

    return s


def send_tensor(index, t, dst, tag):
    """The generic function to send a tensor with its index over Torch Distributed primitives"""
    index = torch.tensor(index, dtype=torch.int)
    t_shape = list(t.shape)
    t_dim = torch.tensor(len(t_shape), dtype=torch.int)
    t_shape = torch.tensor(t_shape, dtype=torch.int)
    dist.send(t_dim, dst, tag=tag)
    dist.send(t_shape, dst, tag=tag)
    dist.send(t, dst, tag=tag)
    dist.send(index, dst, tag=tag)


def recv_tensor(src, tag):
    """The generic function to receive a tensor with its index over Torch Distributed primitives"""
    t_dim = torch.tensor(1, dtype=torch.int)
    dist.recv(t_dim, src, tag=tag)
    t_shape = torch.zeros(t_dim.item(), dtype=torch.int)
    dist.recv(t_shape, src, tag=tag)
    t_shape = list(t_shape.numpy())
    t = torch.zeros(t_shape)
    dist.recv(t, src, tag=tag)
    index = torch.tensor(0, dtype=torch.int)
    dist.recv(index, src, tag=tag)

    return index.item(), t
