import os
import socket
import torchvision.transforms as transforms
from datetime import datetime
from misc.hparams import HParams
from misc.register import register
from easydict import EasyDict as edict


@register('protonet_5way_5shot_disentangle')
def protonet_5way_5shot_disentangle():
    comment = '-protonet-5way-5shot_disentangle'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)

    # setup model
    __model = edict()
    __model.x_dim = 3
    __model.hid_dim = 64
    __model.z_dim = 64

    # setup dataloader
    __dataloader = edict()
    __dataloader.csv_dir = 'data/splits/mini_imagenet_split'
    __dataloader.split = 'Ravi'
    __dataloader.image_dir = 'data/raw/mini-imagenet'
    __dataloader.shuffle = True
    __dataloader.num_threads = 4
    __dataloader.transform = transforms.Compose([transforms.Resize((84, 84)),
                                                 transforms.ToTensor()])
    __dataloader.transformed_images = 'data/raw/mini-imagenet.h5'
    __dataloader.imname_index_file = 'data/raw/mini-imagenet-indices.json'

    # setup optimizer
    __optimizer = edict()
    __optimizer.optim = 'adam'
    __optimizer.learning_rate = 1e-3
    __optimizer.optim_alpha = 0.9
    __optimizer.optim_beta = 0.999
    __optimizer.optim_epsilon = 1e-8
    __optimizer.weight_decay = 0
    __optimizer.grad_clip = 1000  # this means no clip

    # setup training
    __training = edict()
    __training.max_episodes = 1000000
    __training.learning_rate_decay_start = 0
    __training.learning_rate_decay_every = 2000
    __training.learning_rate_decay_rate = 0.5
    __training.nway = 20
    __training.nshot = 5
    __training.nquery = 15

    # setup loss
    __criterion = edict()

    # for testing (sampling trees)
    __testing = edict()
    __testing.test_every_episodes = 1000
    __testing.nway = 5
    __testing.nshot = 5
    __testing.nquery = 15
    __testing.n_episodes = 600

    # setup checkpointing
    __ckpt = edict()
    __ckpt.resume = False
    __ckpt.resume_model = True
    __ckpt.resume_optim = False
    __ckpt.resume_config = True
    __ckpt.load_model_path = None
    __ckpt.load_optim_path = None
    __ckpt.load_config_path = None
    __ckpt.save_model_path = os.path.join(log_dir, 'ckpts')
    __ckpt.save_optim_path = os.path.join(log_dir, 'ckpts')
    __ckpt.save_config_path = os.path.join(log_dir, 'ckpts')
    __ckpt.logging_path = None

    # setup misc
    __misc = edict()
    __misc.seed = 6666

    # setup config
    __config = edict()
    __config.model = __model
    __config.dataloader = __dataloader
    __config.optimizer = __optimizer
    __config.training = __training
    __config.criterion = __criterion
    __config.testing = __testing
    __config.ckpt = __ckpt
    __config.misc = __misc
    __config.log_dir = log_dir
    __config.do_test = False
    __config.cfg_name = 'protonet-5way-5shot'

    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)
    # if not os.path.exists(os.path.join(log_dir, 'ckpts')):
    #     os.mkdir(os.path.join(log_dir, 'ckpts'))
    # if not os.path.exists(os.path.join(log_dir, 'test')):
    #     os.mkdir(os.path.join(log_dir, 'test'))

    return HParams(configs=__config)
