import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    from base import euclidean_dist
    from base import ProtoConvNet
else:
    from .base import euclidean_dist
    from .base import ProtoConvNet


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder

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

        dists = euclidean_dist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {'loss': loss_val.data.cpu().item(),
                          'acc': acc_val.data.cpu().item()}


if __name__ == '__main__':
    import sys
    sys.path.append('/u/cqwang/Projects/few-shot/codes/protonets-variants/code')
    import torchvision.transforms as transforms
    from dataloader.MiniImageNet import MiniImageNetDataset
    from torch.optim import Adam
    csv_dir = 'data/splits/mini_imagenet_split'
    split = 'Ravi'
    image_dir = 'data/raw/mini-imagenet'
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

    encoder = ProtoConvNet(x_dim, hid_dim, z_dim)
    protoNet = Protonet(encoder).cuda()
    optim = Adam(protoNet.parameters(), lr=0.001)
    for idx in range(70000):
        optim.zero_grad()
        data = mini_loader.get_episode(5, 1, 15)
        loss_val, state = protoNet(data)
        loss_val.backward()
        optim.step()
        print('[%d]: loss: %.4f, acc: %.4f%%' % (idx, state['loss'], state['acc']*100))



