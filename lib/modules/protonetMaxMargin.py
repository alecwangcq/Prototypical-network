import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    from base import euclidean_dist
    from base import ProtoConvNet
    from base import ProtoRadiusPredictor
else:
    from .base import euclidean_dist
    from .base import ProtoConvNet
    from .base import ProtoRadiusPredictor


class ProtonetMaxMargin(nn.Module):
    def __init__(self, encoder, radiusPredictor):
        super(ProtonetMaxMargin, self).__init__()
        self.encoder = encoder
        self.radiusPredictor = radiusPredictor
        self.radius_decay = 0

    def forward(self, data):
        x = data['data'].cuda()
        z = self.encoder(x)  # [nway*(nshot + nquery), z_dim]
        z_dim = z.size(-1)
        n_class = data['nway']
        n_support = data['nshot']
        n_query = data['nquery']

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().cuda()
        zs = z[:n_class*n_support]
        zq = z[n_class*n_support:]
        z_proto = zs.view(n_class, n_support, z_dim).mean(1)

        dists = torch.sqrt(euclidean_dist(zq, z_proto) + 1e-6)  # (nq*nc) * nc
        radiuses = self.radiusPredictor(z_proto).view(-1, 1)

        # constrain each point with the corresponding ball
        rexpand = -radiuses.expand(n_class, n_class*n_query).contiguous().view(n_class*n_query, n_class)
        xpmiu_loss = nn.ReLU()(-4 + rexpand + dists)
        xpmiu_loss = xpmiu_loss.view(n_class, n_query, -1).gather(2, target_inds).squeeze().view(-1).mean()

        # constrain the l2 norm of radius of each ball. Uniform prior.
        radius_loss = torch.pow(radiuses, 2).mean()

        # constrain the distances between the center of each ball
        miumiu_dists = torch.sqrt(euclidean_dist(z_proto, z_proto) + 1e-6)  # get distance between each center
        r_add_r = radiuses + radiuses.view(1, -1)
        r_sub_r = (radiuses - radiuses.view(1, -1))
        mask = (r_sub_r > 0).float()
        r_max_r = (mask * radiuses + (1 - mask) * radiuses.t()) * 2
        rr_matrix = r_add_r + r_max_r
        miumiu_loss = nn.ReLU()(16 + rr_matrix - miumiu_dists) * (1-torch.eye(n_class).cuda())
        miumiu_loss = miumiu_loss.view(-1).sum()/(n_class*(n_class-1))
        loss = xpmiu_loss + self.radius_decay*radius_loss + miumiu_loss

        _, y_hat = dists.view(n_class, n_query, -1).min(2)
        acc = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss, {'loss': loss.data.cpu().item(),
                      'acc': acc.data.cpu().item(),
                      'radius': [float(_.item()) for _ in list(radiuses.cpu().data)]}


def check_grad(net):
    for name, param in net.named_parameters():
        if param.grad is not None:
            output = '%-40s: %.5f,\t%.5f,\t%.5f'
            print(output % (name, param.grad.norm().item(), param.max().item(), param.min().item()))


def test(net, loader):
    accs = []
    net = net.eval()
    for idx in range(100):
        episode = loader.get_episode(5, 1, 15, 'test')
        loss_val, state = net(episode)
        radiuses = state['radius']
        print('[Test][%d]: loss: %.4f, acc: %.2f%%, max_r: %.2f, min_r: %.2f' % (idx,
                                                                           state['loss'],
                                                                           state['acc'] * 100,
                                                                           max(radiuses),
                                                                           min(radiuses)))
        accs.append(state['acc'])
    print('Mean acc is: %.2f%%' % (sum(accs)/len(accs)*100))


if __name__ == '__main__':
    import sys
    sys.path.append('/u/cqwang/Projects/few-shot/codes/protonets-variants/code')
    import torchvision.transforms as transforms
    from dataloader.MiniImageNet import MiniImageNetDataset
    from torch.optim import Adam, SGD
    from misc.utils import clip_gradient
    csv_dir = '/u/cqwang/Projects/few-shot/codes/protonets-variants/code/data/splits/mini_imagenet_split'
    split = 'Ravi'
    image_dir = '/u/cqwang/Projects/few-shot/codes/protonets-variants/code/data/raw/mini-imagenet'
    shuffle = True
    num_threads = 4
    transform = transforms.Compose([transforms.Resize((84, 84)),
                                    transforms.ToTensor()])

    mini_loader = MiniImageNetDataset(csv_dir, split,
                                      image_dir, shuffle,
                                      num_threads, transform)
    x_dim = 3
    hid_dim = 64
    z_dim = 64
    in_dim = 1600

    encoder = ProtoConvNet(x_dim, hid_dim, z_dim)
    radiusPredictor = ProtoRadiusPredictor(in_dim)
    protoNet = ProtonetMaxMargin(encoder, radiusPredictor).cuda().train()
    optim = Adam(protoNet.parameters(), lr=0.001)
    for idx in range(10000):
        if (idx + 1) % 1000 == 0:
            test(protoNet, mini_loader)
        if (idx + 1) % 20 == 0:
            check_grad(protoNet)
        protoNet = protoNet.train()
        optim.zero_grad()
        data = mini_loader.get_episode(5, 1, 15)
        loss_val, state = protoNet(data)
        loss_val.backward()
        clip_gradient(optim, 0.25)
        optim.step()
        radiuses = state['radius']
        print('[%d]: loss: %.4f, acc: %.2f%%, max_r: %.2f, min_r: %.2f' % (idx,
                                                                           state['loss'],
                                                                           state['acc']*100,
                                                                           max(radiuses),
                                                                           min(radiuses)))



