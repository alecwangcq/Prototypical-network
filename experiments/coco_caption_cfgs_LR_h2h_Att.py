import os
import json
import socket
from datetime import datetime
from misc.hparams import HParams
from misc.register import register
from misc.vocab import Vocab
from easydict import EasyDict as edict


@register('coco_tree_syntax_caption_configs_LR_h2h_Att')
def coco_tree_syntax_caption_configs_LR_h2h_Att():
    comment = '-COCO-ADAM-words-level-32-att-dropout075-topolabel0.1-h2h-Att'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
    # setup vocab
    coco_caption_json = './data/coco_vocab.json'
    with open(coco_caption_json, 'r') as f:
        coco_caption_words = json.load(f)['words']
    coco_reserved = ('<EOS>', '<SOS>', '<UNK>')
    coco_syntax_words = ['NP', 'NN', 'DT', 'PP', 'IN', 'VP', '.', 'JJ', 'S', 'VBG',
                         'NNS', 'VBZ', 'CC', 'VBP', 'CD', 'TO', 'ADVP', 'VBN', 'RP',
                         'PRT', 'RB', 'PRP$', 'SBAR', 'VBD', 'ADJP', 'PRP', 'VB',
                         'WHNP', 'WDT', 'EX', 'WP', 'JJR', 'NNP', 'LST', 'LS', 'FRAG',
                         'MD', 'WRB', 'WHADVP', 'NX', 'UCP', 'QP', 'SYM', 'FW', 'PDT']

    __vocab = edict()
    __vocab.syntax_vocab = Vocab(coco_syntax_words)
    __vocab.caption_vocab = Vocab(coco_caption_words, coco_reserved)

    # setup language model
    __lang_model = edict()
    __lang_model.mem_dim = 512
    __lang_model.drop_lm = 0.75
    __lang_model.att_hid_dim = 512
    __lang_model.fc_feat_dim = 2048
    __lang_model.att_feat_dim = 2048

    # setup syntax core
    __syntax_emb_dim = 512
    __caption_emb_dim = 512

    __syntax_core = edict()
    __syntax_core.use_vis_fc_mlp = True
    __syntax_core.use_att_fc_mlp = False
    __syntax_core.mem_dim = __lang_model.mem_dim
    __syntax_core.embed_dim = __syntax_emb_dim
    __syntax_core.scale = 0.1
    __syntax_core.in_dim = __syntax_core.embed_dim + __caption_emb_dim + __lang_model.mem_dim + \
                           __syntax_core.mem_dim*__syntax_core.use_vis_fc_mlp + __lang_model.mem_dim
    __syntax_core.rnn_type = 'gru'
    __syntax_core.prediction_hid_dim = 512
    __syntax_core.vocab_size = len(__vocab.syntax_vocab)

    # setup caption core
    __caption_core = edict()
    __caption_core.use_vis_fc_mlp = True
    __caption_core.use_att_fc_mlp = True
    __caption_core.mem_dim = __lang_model.mem_dim
    __caption_core.embed_dim = __caption_emb_dim
    __caption_core.scale = 0.1
    __caption_core.in_dim = __caption_core.embed_dim + __syntax_core.embed_dim + \
                            __caption_core.mem_dim*(__caption_core.use_vis_fc_mlp + __caption_core.use_att_fc_mlp)
    __caption_core.vocab_size = len(__vocab.caption_vocab)

    # setup dataloader
    __dataloader = edict()
    __dataloader.batch_size = 32
    __dataloader.coco_data_file = './data/dataset_coco.json'
    __dataloader.coco_fc_feat_dir = './data/cocotalk_fc'
    __dataloader.coco_att_feat_dir = './data/cocotalk_att'
    __dataloader.coco_tree_file = './data/coco_trees.json'
    __dataloader.link_leaves = True

    # setup optimizer
    __optimizer = edict()
    __optimizer.optim = 'adam'
    __optimizer.learning_rate = 1e-4
    __optimizer.optim_alpha = 0.9
    __optimizer.optim_beta = 0.999
    __optimizer.optim_epsilon = 1e-8
    __optimizer.weight_decay = 5e-4
    __optimizer.grad_clip = 0.1

    # setup training
    __training = edict()
    __training.max_iterations = 1000000
    __training.learning_rate_decay_start = 20000
    __training.learning_rate_decay_every = 12000
    __training.learning_rate_decay_rate = 0.5

    # setup loss
    __criterion = edict()
    __criterion.topo_lambda = 5

    # for testing (sampling trees)
    __testing = edict()
    __testing.test_every_iterations = 2500
    __testing.max_depth = 10
    __testing.max_num_children = 5
    __testing.res_save_path = os.path.join(log_dir, 'test')

    # setup checkpointing
    __ckpt = edict()
    __ckpt.resume = False
    __ckpt.resume_model = True
    __ckpt.resume_optim = False
    __ckpt.resume_config = True
    __ckpt.load_model_path = 'runs/May25_21-47-37_dgx1_coco_ADAM_Dropout/ckpts/model_35000.pth'
    __ckpt.load_optim_path = None
    __ckpt.load_config_path = 'runs/May25_21-47-37_dgx1_coco_ADAM_Dropout/ckpts/configs.pth'
    __ckpt.save_model_path = os.path.join(log_dir, 'ckpts')
    __ckpt.save_optim_path = os.path.join(log_dir, 'ckpts')
    __ckpt.save_config_path = os.path.join(log_dir, 'ckpts')
    __ckpt.logging_path = None

    # setup misc
    __misc = edict()
    __misc.seed = 6666

    # setup config
    __config = edict()
    __config.vocab = __vocab
    __config.lm = __lang_model
    __config.syntax_core = __syntax_core
    __config.caption_core = __caption_core
    __config.dataloader = __dataloader
    __config.optimizer = __optimizer
    __config.training = __training
    __config.criterion = __criterion
    __config.testing = __testing
    __config.ckpt = __ckpt
    __config.misc = __misc
    __config.log_dir = log_dir
    __config.do_test = False

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'ckpts')):
        os.mkdir(os.path.join(log_dir, 'ckpts'))
    if not os.path.exists(os.path.join(log_dir, 'test')):
        os.mkdir(os.path.join(log_dir, 'test'))

    return HParams(configs=__config)
