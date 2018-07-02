import torch
import torch.nn as nn
import torch.nn.functional as F
import random

if __name__ == '__main__':
    from base import euclidean_dist
    from base import ProtoConvNet
    from base import ProtoDisentangle
else:
    from .base import euclidean_dist
    from .base import ProtoConvNet
    from .base import ProtoDisentangle


def make_data_paris(z, ind_z, den_z, n_s, n_q, n_c):
    ind_parts = []
    den_parts = []
    targets = []

    z_s = z[:n_c*n_s]
    z_q = z[n_c*n_s:]
    ind_z_s = ind_z[:n_c*n_s]
    ind_z_q = ind_z[n_c*n_s:]
    den_z_s = den_z[:n_c*n_s]
    den_z_q = den_z[n_c*n_s:]

    # gen for support
    for c in range(n_c):
        for s in range(n_s):
            ind_idx = n_s*c + s
            ind_parts.append(ind_z_s[ind_idx:ind_idx+1, :])
            den_idx = random.randint(0, n_q-1) + n_q*c
            den_parts.append(den_z_q[den_idx:den_idx+1, :])
            targets.append(z_q[den_idx:den_idx+1, :])

    # gen for query
    for c in range(n_c):
        for q in range(n_q):
            ind_idx = n_q*c + q
            ind_parts.append(ind_z_q[ind_idx:ind_idx + 1, :])
            den_idx = random.randint(0, n_s-1) + n_s*c
            den_parts.append(den_z_s[den_idx:den_idx + 1, :])
            targets.append(z_s[den_idx:den_idx + 1, :])

    ind_parts = torch.cat(ind_parts, 0)
    den_parts = torch.cat(den_parts, 0)
    ind_den = torch.cat([ind_parts, den_parts], 1)
    targets = torch.cat(targets, 0)

    return ind_den, targets


class ProtonetDisentangle(nn.Module):
    def __init__(self, encoder, disentangle):
        super(ProtonetDisentangle, self).__init__()
        self.encoder = encoder
        self.disentangle = disentangle

    def forward(self, data):
        x = data['data'].cuda()
        z = self.encoder(x)  # [nway*(nshot + nquery), z_dim]

        n_class = data['nway']
        n_support = data['nshot']
        n_query = data['nquery']

        ind_z, den_z = self.disentangle.encoding(z)
        ro_z, ro_t = torch.cat([ind_z, den_z], 1), z
        rx_z, rx_t = make_data_paris(z, ind_z, den_z, n_support, n_query, n_class)
        r_z, t_z = torch.cat([ro_z, rx_z], 0), torch.cat([ro_t, rx_t], 0)
        dec_z = self.disentangle.decoding(r_z)
        recon_loss = nn.MSELoss()(dec_z, t_z.detach())
        ind_norm_max = ind_z.norm(2, 1).max().cpu().data.item()
        den_norm_max = den_z.norm(2, 1).max().cpu().data.item()
        ind_norm_min = ind_z.norm(2, 1).min().cpu().data.item()
        den_norm_min = den_z.norm(2, 1).min().cpu().data.item()

        z = ind_z
        z_dim = z.size(-1)
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().cuda()
        zs = z[:n_class*n_support]
        zq = z[n_class*n_support:]
        z_proto = zs.view(n_class, n_support, z_dim).mean(1)

        dists = euclidean_dist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        ce_loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        loss_val = ce_loss + recon_loss
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {'loss': loss_val.data.cpu().item(),
                          'recon_loss': recon_loss.cpu().item(),
                          'ce_loss': ce_loss.cpu().item(),
                          'ind_norm_min': ind_norm_min,
                          'den_norm_min': den_norm_min,
                          'ind_norm_max': ind_norm_max,
                          'den_norm_max': den_norm_max,
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
    disentangler = ProtoDisentangle(1600, 300, 600)
    protoNet = ProtonetDisentangle(encoder, disentangler).cuda()
    optim = Adam(protoNet.parameters(), lr=0.001)
    for idx in range(70000):
        optim.zero_grad()
        data = mini_loader.get_episode(5, 1, 15)
        loss_val, state = protoNet(data)
        loss_val.backward()
        optim.step()
        print('[%d]: loss: %.4f, acc: %.4f%%, ce_loss: %.4f, recon_loss: %.4f.' % (idx,
                                                                                  state['loss'],
                                                                                  state['acc']*100,
                                                                                  state['ce_loss'],
                                                                                  state['recon_loss']))



