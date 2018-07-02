import torch
import torch.nn as nn
import torch.optim as optim


# =============================================
# For training
# =============================================
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def init_model(net: nn.Module, configs):
    if not configs.resume_model:
        return
    else:
        print("Resuming model from %s" % configs.mckpt_file)
        m = torch.load(configs.mckpt_file)['model']
        net.load_state_dict(m)


def init_optimizer(optimizer: optim.Optimizer, configs):
    if not configs.resume_optim:
        return
    else:
        print("Resuming optimizer from %s" % configs.ockpt_file)
        o = torch.load(configs.ockpt_file)['optim']
        optimizer.load_state_dict(o)


def build_optimizer(params, configs):
    """
    Args:
        ` optim: required by all, "rmsprop|adagrad|sgd|sgdm|sgdmom|adam"
        ` learning_rate: required by all
        ` optim_alpha: required by rmsprop & sgdm & sgdmom & adam
        ` optim_beta: required by adam
        ` optim_epsilon: required by rmsprop & adam
        ` weight_decay: required by all
        `
    """
    if configs.optim == 'rmsprop':
        return optim.RMSprop(params, configs.learning_rate, configs.optim_alpha, configs.optim_epsilon,
                             weight_decay=configs.weight_decay)
    elif configs.optim == 'adagrad':
        return optim.Adagrad(params, configs.learning_rate, weight_decay=configs.weight_decay)
    elif configs.optim == 'sgd':
        return optim.SGD(params, configs.learning_rate, weight_decay=configs.weight_decay)
    elif configs.optim == 'sgdm':
        return optim.SGD(params, configs.learning_rate, configs.optim_alpha, weight_decay=configs.weight_decay)
    elif configs.optim == 'sgdmom':
        return optim.SGD(params, configs.learning_rate, configs.optim_alpha, weight_decay=configs.weight_decay,
                         nesterov=True)
    elif configs.optim == 'adam':
        return optim.Adam(params, configs.learning_rate, (configs.optim_alpha, configs.optim_beta),
                          configs.optim_epsilon, weight_decay=configs.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(configs.optim))