import sys
import os
import pdb
import json
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

from lib.modules import Protonet
from lib.modules import ProtoConvNet
from dataloader import MiniImageNetDataset


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

    logger.info('[Train](%d-way, %d-shot) Iteration [%d]: loss: %.4f, acc: %.2f%%' % (
        training.nway, training.nshot, episodes, state['loss'], state['acc']*100))

    writer.add_scalar('train_loss', state['loss'], episodes)
    writer.add_scalar('train_acc', state['acc'], episodes)
    lr_scheduler(optimizer, episodes)
    optimizer.zero_grad()
    loss_val.backward()
    clip_gradient(optimizer, configs.optimizer.grad_clip)
    optimizer.step()


def test(split, configs):
    global net, dataloader, optimizer, lr_scheduler, logger, writer, epochs
    net = net.train()

    testing = configs.training

    accs = []
    loss = []
    for episode in tqdm(range(testing.n_episodes), desc="Epoch %d, %s" % (epochs, split)):
        data = dataloader.get_episode(testing.nway, testing.nshot, testing.nquery, split)
        _, state = net(data)
        logger.info('[%s](%d-way, %d-shot) Iteration [%d]: loss: %.4f, acc: %.2f%%' % (
            split, testing.nway, testing.nshot, episode, state['loss'], state['acc'] * 100))
        accs.append(state['acc'])
        loss.append(state['loss'])

    mean_acc = sum(accs)/len(accs)
    writer.add_scalar('%s_loss' % split, mean_acc, epochs)
    writer.add_scalar('%s_acc' % split, sum(loss)/len(loss), epochs)

    return mean_acc


def build_model(configs):
    model = configs.model
    encoder = ProtoConvNet(model.x_dim, model.hid_dim, model.z_dim)
    return Protonet(encoder)


def build_dataset(configs):
    d = configs.dataloader
    return MiniImageNetDataset(d.csv_dir, d.split, d.image_dir, d.shuffle, d.num_threads, d.transform)


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


def main(configs, args):
    global net, dataloader, optimizer, lr_scheduler, writer, epochs, logger
    best_acc = 0

    torch.manual_seed(6666)
    configs = init_configs(configs)
    net = build_model(configs)
    net = init_model(net, configs)
    net = net.cuda().train()
    print(net)

    if not configs.do_test:
        config_path = configs.ckpt.save_config_path
        torch.save({'configs': configs}, os.path.join(config_path, 'configs.pth'))
    if args.debug:
        configs.log_dir = os.path.join('debug', configs.log_dir)

    writer = SummaryWriter(configs.log_dir)
    logger = create_logger(configs.log_dir)

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
                if not configs.do_test:
                    torch.save({'model': net.state_dict()}, os.path.join(model_path, 'model_%d.pth' % iteration))
                    torch.save({'optim': optimizer.state_dict()}, os.path.join(optim_path, 'optim_%d.pth' % iteration))
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
