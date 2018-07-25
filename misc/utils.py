import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

def nCk_inds(n, k):
    samples = [i for i in range(n)]
    random.shuffle(samples)
    return samples[:k]


class VisAttention(object):
    def __init__(self, nway, nsupport, nquery, nsupport_sample, nquery_sample, width=14, height=10):
        plt.ion()
        assert nsupport_sample <= nsupport
        assert nquery_sample <= nquery
        self.fig, self.ax = plt.subplots(nway, (nsupport_sample + nquery_sample) * 2, sharex=True, sharey=True)
        self.nway = nway
        self.nsupport = nsupport
        self.nquery = nquery
        self.s_sample = nsupport_sample
        self.q_sample = nquery_sample
        self.fig.set_size_inches(width, height)
        self.rows = []
        self.labels = []

    def update(self, images, attention):
        """

        :param images: B x C x W x H
        :param attention: B x 1 x w' x h'
        :return:
        """
        samples_per_cls = self.nsupport + self.nquery
        assert images.size()[0] == self.nway * samples_per_cls
        n_image = images.size()[0]
        if images.shape[2:] == attention.shape[2:]:
            spatial_attention = attention
        else:
            spatial_attention = nn.functional.upsample(attention, size=(images.size()[2], images.size()[3]),
                                                       mode='bilinear')
        spatial_attention = spatial_attention.numpy()
        spatial_attention = 128 * (spatial_attention - spatial_attention.min()) / \
                            (spatial_attention.max() - spatial_attention.min() + 1e-8)
        spatial_attention = np.transpose(np.tile(spatial_attention.astype(np.uint8), (3, 1, 1)), (0, 2, 3, 1))

        images = images.numpy()
        images = 255 * (images - images.min()) / (images.max() - images.min() + 1e-8)
        images = np.transpose(images.astype(np.uint8), (0, 2, 3, 1))

        q_samples = nCk_inds(self.nquery, self.q_sample)
        s_samples = nCk_inds(self.nsupport, self.s_sample)

        rows = self.rows = []
        labels = self.labels = []
        for i in range(self.nway):
            rows.append([])
            labels.append([])
            for j in s_samples:
                ind = i*self.nsupport + j
                rows[i].append(images[ind])
                rows[i].append(spatial_attention[ind])
                labels[i].append('support')
                labels[i].append('Attn')

            for j in q_samples:
                ind = self.nway * self.nsupport + i*self.nquery + j
                rows[i].append(images[ind])
                rows[i].append(spatial_attention[ind])
                labels[i].append('query')
                labels[i].append('Attn')

    def show(self):
        for r, row in enumerate(self.rows):
            for c, col in enumerate(row):
                self.ax[r][c].imshow(col)
                self.ax[r][c].set_title(self.labels[r][c])
        self.fig.show()
        self.fig.canvas.draw()
        plt.pause(0.001)




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