import sys
import os
import pdb
import json
import math
import tqdm
import numpy as np
import PIL.Image as im
import pprint
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from misc.register import get_hparams
from misc.utils import build_optimizer
from misc.utils import clip_gradient, VisAttention
from misc.lr_schedulers import get_lr_scheduler
from misc.logger import create_logger

#from lib.modules import Protonet
from lib.modules import *
from dataloader.MiniImageNetMAML import MiniImageNetMAMLDataset


net = None
dataloader = None
optimizer = None
lr_scheduler = None
logger = None
saver = None
writer = None
plotter = None
epochs = 0


def pretrain(episodes, configs):
    global net, dataloader, optimizer, lr_scheduler, logger, writer, epochs, plotter
    net = net.train()
    training = configs.training
    data = dataloader.get_episode_MAML(training.nway, training.nshot, training.nquery, training.nmeta)
    optimizer.zero_grad()
    loss_val, state = net(data, training.nway, training.nshot, training.nquery, training.nmeta, 0, 0.0)

    if episodes % 50 == 0:
        logger.info('[PreTrain](%d-way, %d-shot) Iteration [%d]: preloss: %.4f, preacc: %.2f%%, postloss: %.4f, postacc: %.2f%% ' % (
            training.nway, training.nshot, episodes, state['first_loss'], state['first_acc']*100, state['loss'], state['acc']*100))
        print('[PreTrain](%d-way, %d-shot) Iteration [%d]: preloss: %.4f, preacc: %.2f%%, postloss: %.4f, postacc: %.2f%% ' % (
            training.nway, training.nshot, episodes, state['first_loss'], state['first_acc']*100, state['loss'], state['acc']*100))


    writer.add_scalar('train_loss', state['loss'], episodes)
    writer.add_scalar('train_acc', state['acc'], episodes)
    lr_scheduler(optimizer, episodes)
    #loss_val.backward()
    clip_gradient(optimizer, configs.optimizer.grad_clip)
    optimizer.step()

def train(episodes, configs):
    global net, dataloader, optimizer, lr_scheduler, logger, writer, epochs, plotter
    net = net.train()
    training = configs.training
    data = dataloader.get_episode_MAML(training.nway, training.nshot, training.nquery, training.nmeta)
    optimizer.zero_grad()
    loss_val, state = net(data, training.nway, training.nshot, training.nquery, training.nmeta, training.num_updates, training.inner_lr)

    if episodes % 50 == 0:
        logger.info('[Train](%d-way, %d-shot) Iteration [%d]:  preloss: %.4f, preacc: %.2f%%, postloss: %.4f, postacc: %.2f%% ' % (
            training.nway, training.nshot, episodes, state['first_loss'], state['first_acc']*100, state['loss'], state['acc']*100))
        print('[Train](%d-way, %d-shot) Iteration [%d]:  preloss: %.4f, preacc: %.2f%%, postloss: %.4f, postacc: %.2f%% ' % (
            training.nway, training.nshot, episodes, state['first_loss'], state['first_acc']*100, state['loss'], state['acc']*100))


    writer.add_scalar('train_loss', state['loss'], episodes)
    writer.add_scalar('train_acc', state['acc'], episodes)
    for i in range(training.num_updates):
        writer.add_scalar('train_inner_loop_acc_step_%i' % i, state['inner_loop_acc'][i], episodes)
        writer.add_scalar('train_inner_loop_loss_step_%i' % i, state['inner_loop_loss'][i], episodes)

    lr_scheduler(optimizer, episodes)

    #loss_val.backward()
    clip_gradient(optimizer, configs.optimizer.grad_clip)
    optimizer.step()


def test(split, configs):
    global net, dataloader, logger, writer, epochs, plotter
    net = net.eval()

    testing = configs.testing

    accs = []
    loss = []
    inner_loop_accs = [[] for _ in range(testing.num_updates)]
    inner_loop_loss = [[] for _ in range(testing.num_updates)]
    for episode in tqdm.tqdm(range(testing.n_episodes), desc="Epoch %d, %s" % (epochs, split)):
        data = dataloader.get_episode_MAML(testing.nway, testing.nshot, testing.nquery, testing.nmeta, split)
        # TODO: fix
        _, state = net(data, testing.nway, testing.nshot, testing.nquery, testing.nmeta, testing.num_updates, testing.inner_lr)

        if configs.misc.visualize and episode == (testing.n_episodes - 1):
            images = data['data'].detach().clone().cpu()  # B 3 84 84
            spatial_attention = net.encoder.padded_attn.detach().clone().cpu()  # B 1 84 84
            #spatial_attention = net.encoder.spatial_output.detach().clone().cpu()  # B 1 5 5
            plotter.update(images, spatial_attention)
            plotter.show()

        logger.info('[%s](%d-way, %d-shot) Iteration [%d]:  preloss: %.4f, preacc: %.2f%%, postloss: %.4f, postacc: %.2f%% ' % (
            split, testing.nway, testing.nshot, episode, state['first_loss'], state['first_acc']*100, state['loss'], state['acc']*100))
        # print('[%s](%d-way, %d-shot) Iteration [%d]: loss: %.4f, acc: %.2f%%' % (
        #     split, testing.nway, testing.nshot, episode, state['loss'], state['acc'] * 100))
        accs.append(state['acc'])
        loss.append(state['loss'])
        for i in range(testing.num_updates):
            inner_loop_accs[i].append(state['inner_loop_acc'][i])
            inner_loop_loss[i].append(state['inner_loop_loss'][i])

    mean_acc = sum(accs)/len(accs)
    std = np.std(accs) * 1.96 / math.sqrt(len(accs))
    il_accs = np.array(inner_loop_accs).mean(axis=1)
    il_loss = np.array(inner_loop_loss).mean(axis=1)
    writer.add_scalar('%s_acc' % split, mean_acc, epochs)
    writer.add_scalar('%s_loss' % split, sum(loss)/len(loss), epochs)
    for i in range(testing.num_updates):
        writer.add_scalar('%s_acc_%i' % (split, i), il_accs[i], epochs)
        writer.add_scalar('%s_loss_%i' % (split, i), il_loss[i], epochs)

    print('[%s] Epoch[%d], acc: %.2f +/- %.2f' % (split, epochs, mean_acc*100, std*100))
    logger.info('[%s] Epoch[%d], acc: %.2f +/- %.2f' % (split, epochs, mean_acc*100, std*100))
    return mean_acc


def extract_features(split, configs):
    global net, dataloader, logger, writer, epochs
    net = net.eval()
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
        wrapped = data['wrapped']
        images.extend(data['images'])

    return torch.cat(feats, 0), images, labels


def build_model(configs):
    model = configs.model
    encoder = ProtoConvNetMAML(model.x_dim, model.hid_dim, model.z_dim)
    return ProtonetMAML(encoder)

#def build_model(configs):
#    model = configs.model
#    encoder = ProtoConvNetBPFeatureAttn(model.x_dim, model.hid_dim, model.z_dim, attn_iters=10, distance='euc')
    #encoder = ProtoConvNetwithGGNNAttn(model.x_dim, model.hid_dim, model.z_dim, use_relu=False, n_steps=1)
    #encoder = ProtoConvNetwithAttn(model.x_dim, model.hid_dim, model.z_dim, attn_module=BiLSTMAttention, hid_dim_conv=128, hid_dim_lstm=256)
#    return Protonet(encoder)


def build_dataset(configs):
    d = configs.dataloader
    return MiniImageNetMAMLDataset(d.csv_dir, d.split, d.image_dir,
                               d.shuffle, d.num_threads, d.transform,
                               d.transformed_images, d.imname_index_file, d.separate_metasample, d.metasample_holdout)


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
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'ckpts')):
        os.mkdir(os.path.join(log_dir, 'ckpts'))
    if not os.path.exists(os.path.join(log_dir, 'test')):
        os.mkdir(os.path.join(log_dir, 'test'))


def main(configs, args):
    global net, dataloader, optimizer, lr_scheduler, writer, epochs, logger, plotter
    best_acc = 0

    torch.manual_seed(6666)
    configs = init_configs(configs)
    net = build_model(configs)
    net = init_model(net, configs)
    net = net.cuda().train()
    print(net)
    if configs.misc.visualize:
        plotter = VisAttention(configs.testing.nway, configs.testing.nshot, configs.testing.nquery, 5, 3)
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

    pretrain_iterations = configs.training.pretrain_episodes
    max_iterations = configs.training.max_episodes
    test_every_iterations = configs.testing.test_every_episodes

    itr = 0

    for iteration in range(1, max_iterations + 1):
        try:
            if iteration % test_every_iterations == 0 or configs.do_test or (args.debug and args.debug_test):
                epochs += 1
                acc = test('test', configs)
                optim_path = configs.ckpt.save_optim_path
                model_path = configs.ckpt.save_model_path
                # z, images, labels = extract_features('test', configs)
                if not configs.do_test:
                    torch.save({'model': net.state_dict()}, os.path.join(model_path, 'model_%d.pth' % iteration))
                    torch.save({'optim': optimizer.state_dict()}, os.path.join(optim_path, 'optim_%d.pth' % iteration))
                    # torch.save({'z': z.numpy(),
                    #             'labels': labels,
                    #             'images': images},
                    #            os.path.join(model_path, 'results_%d.pth' % iteration))
                    if acc > best_acc:
                        best_acc = acc
                        torch.save({'model': net.state_dict()}, os.path.join(model_path, 'model_best.pth'))
                        torch.save({'optim': optimizer.state_dict()}, os.path.join(optim_path, 'optim_best.pth'))
                if configs.do_test or (args.debug and args.debug_test):
                    return

            if iteration < pretrain_iterations:
                pretrain(iteration, configs)
            else:
                train(iteration, configs)
            itr += 1
        except KeyboardInterrupt:
            import ipdb
            ipdb.set_trace()
            #pass



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