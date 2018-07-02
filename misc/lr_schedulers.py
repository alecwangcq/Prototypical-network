class LRScheduler(object):
    def __init__(self, start_at, interval, decay_rate):
        self.start_at = start_at
        self.interval = interval
        self.decay_rate = decay_rate

    def __call__(self, optimizer, iteration):
        start_at = self.start_at
        interval = self.interval
        decay_rate = self.decay_rate
        if (start_at >= 0) \
                and (iteration >= start_at) \
                and (iteration + 1)%interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
                print('[%d]Decay lr to %f' % (iteration, param_group['lr']))


def get_lr_scheduler(configs):
    start_at = configs.learning_rate_decay_start
    interval = configs.learning_rate_decay_every
    decay_rate = configs.learning_rate_decay_rate
    return LRScheduler(start_at, interval, decay_rate)


