import sys
import os
import pdb
import json
import numpy as np
import math
import tqdm
import pprint
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from misc.register import get_hparams
from misc.utils import build_optimizer
from misc.utils import clip_gradient
from misc.lr_schedulers import get_lr_scheduler
from misc.logger import create_logger

from lib.modules import ProtonetDisentangle
from lib.modules import ProtoConvNet
from lib.modules import ProtoDisentangle
from dataloader.MiniImageNet import MiniImageNetDataset


net = None
dataloader = None
optimizer = None
lr_scheduler = None
logger = None
saver = None
writer = None
epochs = 0


def train(episodes, configs):
    global net, dataloader, optimizer, lr_scheduler, logger, writer, epochs
    net = net.train()

    training = configs.training
    data = dataloader.get_episode(training.nway, training.nshot, training.nquery)

    loss_val, state = net(data)
    ind_norm_max, den_norm_max = state['ind_norm_max'], state['den_norm_max']
    ind_norm_min, den_norm_min = state['ind_norm_min'], state['den_norm_min']
    if (episodes-1) % 50 == 0:
        print('[Train](%d-way, %d-shot) Iteration [%d]: ind_norm_max: %.4f, '
              'ind_norm_min: %.4f, den_norm_max: %.4f, den_norm_min: %.4f' %(training.nway, training.nshot,
                                                                             episodes, ind_norm_max, ind_norm_min,
                                                                             den_norm_max, den_norm_min))
        print('[Train](%d-way, %d-shot) Iteration [%d]: loss: %.4f, acc: %.2f%%, ce_loss: %.4f, recon_loss: %.4f' % (
            training.nway, training.nshot, episodes, state['loss'], state['acc']*100, state['ce_loss'],
            state['recon_loss']))

        logger.info('[Train](%d-way, %d-shot) Iteration [%d]: ind_norm_max: %.4f, '
                    'ind_norm_min: %.4f, den_norm_max: %.4f, den_norm_min: %.4f' %
                    (training.nway, training.nshot, episodes, ind_norm_max, ind_norm_min, den_norm_max, den_norm_min))
        logger.info('[Train](%d-way, %d-shot) Iteration [%d]: loss: %.4f, '
                    'acc: %.2f%%, ce_loss: %.4f, recon_loss: %.4f' % (training.nway, training.nshot, episodes,
                                                                      state['loss'], state['acc'] * 100,
                                                                      state['ce_loss'], state['recon_loss']))

    writer.add_scalar('train_loss', state['loss'], episodes)
    writer.add_scalar('train_acc', state['acc'], episodes)
    lr_scheduler(optimizer, episodes)
    optimizer.zero_grad()
    loss_val.backward()
    clip_gradient(optimizer, configs.optimizer.grad_clip)
    optimizer.step()


def test(split, configs):
    global net, dataloader, optimizer, lr_scheduler, logger, writer, epochs
    net = net.eval()

    testing = configs.testing

    accs = []
    loss = []
    for episode in tqdm.tqdm(range(testing.n_episodes), desc="Epoch %d, %s" % (epochs, split)):
        data = dataloader.get_episode(testing.nway, testing.nshot, testing.nquery, split)
        _, state = net(data)
        ind_norm_max, den_norm_max = state['ind_norm_max'], state['den_norm_max']
        ind_norm_min, den_norm_min = state['ind_norm_min'], state['den_norm_min']

        logger.info('[%s](%d-way, %d-shot) Iteration [%d]: ind_norm_max: %.4f, '
                    'den_norm_max: %.4f, ind_norm_min: %.4f, den_norm_min: %.4f' % (split, testing.nway,
                                                                                    testing.nshot, episode,
                                                                                    ind_norm_max, den_norm_max,
                                                                                    ind_norm_min, den_norm_min))

        logger.info('[%s](%d-way, %d-shot) Iteration [%d]: loss: %.4f, acc: %.2f%%, '
                    'ce_loss: %.4f, recon_loss: %.4f' % (split, testing.nway, testing.nshot,
                                                         episode, state['loss'], state['acc'] * 100,
                                                         state['ce_loss'], state['recon_loss']))
        accs.append(state['acc'])
        loss.append(state['loss'])

    mean_acc = sum(accs)/len(accs)
    std = np.std(accs) * 1.96 / math.sqrt(len(accs))
    writer.add_scalar('%s_acc' % split, mean_acc, epochs)
    writer.add_scalar('%s_loss' % split, sum(loss)/len(loss), epochs)
    print('[%s] Epoch[%d], acc: %.2f +/- %.2f' % (split, epochs, mean_acc * 100, std * 100))
    logger.info('[%s] Epoch[%d], acc: %.2f +/- %.2f' % (split, epochs, mean_acc * 100, std * 100))
    return mean_acc


def extract_features(split, configs):
    global net, dataloader, logger, writer, epochs
    net = net.eval()
    ind_feats = []
    den_feats = []
    feats = []
    images = []
    labels = []
    batch_size = 128
    wrapped = False
    while not wrapped:
        data = dataloader.get_batch(batch_size, split)
        labels.extend(data['label'])
        zs = net.extract(data)
        feats.append(zs['z'])
        ind_feats.append(zs['ind_z'])
        den_feats.append(zs['den_z'])
        wrapped = data['wrapped']
        images.extend(data['images'])
    return torch.cat(feats, 0), torch.cat(ind_feats, 0), torch.cat(den_feats, 0), images, labels


def build_model(configs):
    model = configs.model
    encoder = ProtoConvNet(model.x_dim, model.hid_dim, model.z_dim)
    disentangle = ProtoDisentangle(1600, 512, 1024)
    return ProtonetDisentangle(encoder, disentangle)


def build_dataset(configs):
    d = configs.dataloader
    return MiniImageNetDataset(d.csv_dir, d.split, d.image_dir,
                               d.shuffle, d.num_threads, d.transform,
                               d.transformed_images, d.imname_index_file)


def init_model(model: nn.Module, configs):
    resume = configs.ckpt.resume
    resume_model = configs.ckpt.resume_model
    model_path = configs.ckpt.load_model_path
    if resume and resume_model:
        print('Loading model from %s.' % model_path)
        state_dict = torch.load(model_path)['model']
        model.load_state_dict(state_dict)
    return model


def init_optim(optim, configs):
    resume = configs.ckpt.resume
    resume_optim = configs.ckpt.resume_optim
    optim_path = configs.ckpt.load_optim_path
    if resume and resume_optim:
        print('Loading optimizer from %s.' % optim_path)
        state_dict = torch.load(optim_path)['optim']
        optim.load_state_dict(state_dict)
    return optim


def init_configs(configs):
    resume = configs.ckpt.resume
    resume_config = configs.ckpt.resume_config
    config_path = configs.ckpt.load_config_path
    ckpt = configs.ckpt
    do_test = configs.do_test
    if resume and resume_config:
        print('Loading configs from %s.' % config_path)
        cfgs = torch.load(config_path)['configs']
        configs = cfgs
        configs.ckpt = ckpt
        configs.do_test = do_test
    return configs


def check_dir(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'ckpts')):
        os.mkdir(os.path.join(log_dir, 'ckpts'))
    if not os.path.exists(os.path.join(log_dir, 'test')):
        os.mkdir(os.path.join(log_dir, 'test'))


def main(configs, args):
    global net, dataloader, optimizer, lr_scheduler, writer, epochs, logger
    best_acc = 0

    torch.manual_seed(6666)
    configs = init_configs(configs)
    net = build_model(configs)
    net = init_model(net, configs)
    net = net.cuda().train()
    print(net)

    if args.debug:
        configs.log_dir = os.path.join('debug', configs.log_dir)
        configs.ckpt.save_config_path = os.path.join('debug', configs.ckpt.save_config_path)
        configs.ckpt.save_model_path = os.path.join('debug', configs.ckpt.save_model_path)
        configs.ckpt.save_optim_path = os.path.join('debug', configs.ckpt.save_optim_path)

    check_dir(configs.log_dir)
    if not configs.do_test:
        config_path = configs.ckpt.save_config_path
        torch.save({'configs': configs}, os.path.join(config_path, 'configs.pth'))

    logger = create_logger(configs.log_dir, configs.cfg_name)
    writer = SummaryWriter(configs.log_dir)

    for name, param in net.named_parameters():
        print('%s required grad is %s' % (name, param.requires_grad))

    dataloader = build_dataset(configs)
    optimizer = build_optimizer(net.parameters(), configs.optimizer)
    optimizer = init_optim(optimizer, configs)
    lr_scheduler = get_lr_scheduler(configs.training)

    max_iterations = configs.training.max_episodes
    test_every_iterations = configs.testing.test_every_episodes
    for iteration in range(1, max_iterations + 1):
        try:
            if iteration % test_every_iterations == 0 or configs.do_test or (args.debug and args.debug_test):
                epochs += 1
                acc = test('test', configs)
                optim_path = configs.ckpt.save_optim_path
                model_path = configs.ckpt.save_model_path
                z, ind_z, den_z, images, labels = extract_features('test', configs)
                if not configs.do_test:
                    torch.save({'model': net.state_dict()}, os.path.join(model_path, 'model_%d.pth' % iteration))
                    torch.save({'optim': optimizer.state_dict()}, os.path.join(optim_path, 'optim_%d.pth' % iteration))
                    torch.save({'z': z.numpy(),
                                'ind_z': ind_z.numpy(),
                                'den_z': den_z.numpy(),
                                'labels': labels,
                                'images': images},
                               os.path.join(model_path, 'results_%d.pth' % iteration))
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({'model': net.state_dict()}, os.path.join(model_path, 'model_best.pth'))
                        torch.save({'optim': optimizer.state_dict()}, os.path.join(optim_path, 'optim_best.pth'))
                if configs.do_test or (args.debug and args.debug_test):
                    return
            train(iteration, configs)
        except KeyboardInterrupt:
            import ipdb
            ipdb.set_trace()


if __name__ == '__main__':
    import argparse
    from experiments import *
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, required=True, help='Which set of hparams to use?')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_test', action='store_true')
    args = parser.parse_args()
    print('Loading hparams:', args.hparams)
    configs = get_hparams(args.hparams)().configs
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(configs)
    print('Do debug is', args.debug)
    print('Do debug test is', args.debug_test)
    main(configs, args)
