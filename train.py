import pdb
import os
import json
import pprint
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from misc.register import get_hparams
from misc.utils import build_optimizer
from misc.utils import clip_gradient
from misc.lr_schedulers import get_lr_scheduler


CHECK = False
net = None
dataloader = None
optimizer = None
lr_scheduler = None
logger = None
saver = None
writer = None


class Object(object):
    pass


def train(iterations, configs):
    global net, dataloader, optimizer, lr_scheduler, logger, CHECK, writer
    net = net.train()
    cfg = Object()
    cfg.batch_size = configs.dataloader.batch_size

    data = dataloader.get_batch('train', cfg)
    trees = data['trees']
    fc_feats = data['fc_feats'].cuda()
    att_feats = data['att_feats'].cuda()

    preds, gts = net(trees, fc_feats, att_feats)
    sytx_pred_a, sytx_pred_f, sytx_pred_o, capt_pred_o = preds
    sytx_gt_a, sytx_gt_f, sytx_gt_o = torch.cuda.FloatTensor(gts[0]), torch.cuda.FloatTensor(gts[1]), torch.cuda.LongTensor(gts[2])
    capt_gt_o = torch.cuda.LongTensor(gts[3])

    sytx_topo_a_loss = nn.BCELoss()(sytx_pred_a.squeeze(), sytx_gt_a)
    sytx_topo_f_loss = nn.BCELoss()(sytx_pred_f.squeeze(), sytx_gt_f)
    sytx_o_loss = nn.CrossEntropyLoss()(sytx_pred_o, sytx_gt_o)

    capt_o_loss = nn.CrossEntropyLoss()(capt_pred_o, capt_gt_o)
              
    sytx_loss = (sytx_topo_a_loss + sytx_topo_f_loss) + sytx_o_loss
    capt_loss = capt_o_loss
    loss = 0.02*sytx_loss + capt_loss

    print('Iteration [%d]: loss: %.5f; sytx_loss:%.5f, capt_loss:%.5f; sytx_a:%.5f, '
          'sytx_f:%.5f, sytx_o:%.5f;  capt_o:%.5f' % (iterations,
                                                      loss.cpu().item(),
                                                      sytx_loss.cpu().item(),
                                                      capt_loss.cpu().item(),
                                                      sytx_topo_a_loss.cpu().item(),
                                                      sytx_topo_f_loss.cpu().item(),
                                                      sytx_o_loss.cpu().item(),
                                                      capt_o_loss.cpu().item()))
    writer.add_scalar('loss/loss', loss.cpu().item(), iterations)
    writer.add_scalar('loss/sytx_loss', sytx_loss.cpu().item(), iterations)
    writer.add_scalar('loss/capt_loss', capt_loss.cpu().item(), iterations)
    writer.add_scalar('loss/sytx_topo_a', sytx_topo_a_loss.cpu().item(), iterations)
    writer.add_scalar('loss/sytx_topo_f', sytx_topo_f_loss.cpu().item(), iterations)
    writer.add_scalar('loss/sytx_o', sytx_o_loss.cpu().item(), iterations)
    writer.add_scalar('loss/capt_o', capt_o_loss.cpu().item(), iterations)

    lr_scheduler(optimizer, iterations)
    optimizer.zero_grad()
    loss.backward()
    clip_gradient(optimizer, configs.optimizer.grad_clip)
    optimizer.step()

    if data['wrapped']:
        dataloader.shuffle('train')


def test(split, configs):
    global net, dataloader, writer
    net = net.eval()

    cfgs = Object()
    cfgs.batch_size = 1

    gen_trees = []
    im_infos = []
    gt_trees = []
    captions = []

    wrapped = False
    count = 0
    while not wrapped:
        data = dataloader.get_batch(split, cfgs)
        trees = data['trees']
        fc_feats = data['fc_feats'].cuda()
        att_feats = data['att_feats'].cuda()
        info = data['image_infos']
        wrapped = data['wrapped']
        st = net.tree_sample(fc_feats,
                             att_feats,
                             configs.testing.max_depth,
                             configs.testing.max_num_children)

        gt_trees.append(trees)
        gen_trees.append(st)
        im_infos.append(info)
        image_id = info[0]['cocoid']

        captions.append({'image_id': image_id, 'caption': caption})
        count += 1
        if count % 100 == 0:
            print('Test [%d] Done.' % count)
            # break

    return gen_trees, im_infos, gt_trees, captions


def build_model(configs):
    cfgs = {
        'vocab': configs.vocab,
        'syntax_core': configs.syntax_core,
        'caption_core': configs.caption_core,
        'mem_dim': configs.lm.mem_dim,
        'drop_lm': configs.lm.drop_lm,
        'att_hid_dim': configs.lm.att_hid_dim,
        'fc_feat_dim': configs.lm.fc_feat_dim,
        'att_feat_dim': configs.lm.att_feat_dim,
        'link_leaves': configs.dataloader.link_leaves
    }
    return DRNNNetFast(edict(cfgs))


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
    annFile = '/u/cqwang/Tools/coco-caption/annotations/captions_val2014.json'
    global net, dataloader, optimizer, lr_scheduler, CHECK, writer
    torch.manual_seed(6666)
    configs = init_configs(configs)
    net = build_model(configs)
    net = init_model(net, configs)
    net = net.cuda()
    coco = COCO(annFile)
    print(net)

    if not configs.do_test:
        config_path = configs.ckpt.save_config_path
        torch.save({'configs': configs}, os.path.join(config_path, 'configs.pth'))
    if args.debug:
        configs.log_dir = os.path.join('debug', configs.log_dir)

    writer = SummaryWriter(configs.log_dir)

    for name, param in net.named_parameters():
        print('%s required grad is %s' % (name, param.requires_grad))
    dataloader = COCOTreeDataloader(configs.dataloader)
    optimizer = build_optimizer(net.parameters(), configs.optimizer)
    optimizer = init_optim(optimizer, configs)
    lr_scheduler = get_lr_scheduler(configs.training)

    max_iterations = configs.training.max_iterations
    test_every_iterations = configs.testing.test_every_iterations
    for iteration in range(1, max_iterations+1):
        try:
            if iteration % test_every_iterations == 0 or configs.do_test or (args.debug and args.debug_test):
                results = test('test', configs)
                optim_path = configs.ckpt.save_optim_path
                model_path = configs.ckpt.save_model_path
                results_path = configs.testing.res_save_path
                if not configs.do_test:
                    torch.save({'model': net.state_dict()}, os.path.join(model_path, 'model_%d.pth' % iteration))
                    torch.save({'optim': optimizer.state_dict()}, os.path.join(optim_path, 'optim_%d.pth' % iteration))
                torch.save({'results': results}, os.path.join(results_path, 'results_%d.pth' % iteration))
                results = torch.load(os.path.join(results_path, 'results_%d.pth' % iteration))['results']
                resFile = os.path.join(results_path, 'captions_%d.json' % iteration)
                with open(resFile, 'w') as f:
                    json.dump(results[-1], f)
                cocoRes = coco.loadRes(resFile)
                cocoEval = COCOEvalCap(coco, cocoRes)
                cocoEval.params['image_id'] = cocoRes.getImgIds()
                cocoEval.evaluate()
                for metric, score in cocoEval.eval.items():
                    print('%s: %.3f' % (metric, score))
                    writer.add_scalar('test/%s' % metric, score, iteration // test_every_iterations)
                if configs.do_test or (args.debug and args.debug_test):
                    return
            train(iteration, configs)
        except KeyboardInterrupt:
            CHECK = True


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
