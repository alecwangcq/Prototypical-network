import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

if __name__ == '__main__':
    from base import euclidean_dist
else:
    from .base import euclidean_dist


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_block(in_channels, out_channels, **kwargs):
    track_running_stats = kwargs.get('track_running_stats', True)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, affine=False, track_running_stats=track_running_stats),
        nn.ReLU(),
        nn.MaxPool2d(2))

class ProtoConvNetMAML(nn.Module):
    def __init__(self, x_dim, hid_dim, z_dim):
        super(ProtoConvNetMAML, self).__init__()
        self.net = nn.Sequential(conv_block(x_dim, hid_dim, track_running_stats=False),
                                 conv_block(hid_dim, hid_dim, track_running_stats=False),
                                 conv_block(hid_dim, hid_dim, track_running_stats=False),
                                 conv_block(hid_dim, z_dim, track_running_stats=False),
                                 Flatten())

    def forward(self, x, *args):
        return self.net(x), None


class ProtonetMAML(nn.Module):
    def __init__(self, encoder, **kwargs):
        super(ProtonetMAML, self).__init__()
        self.encoder = encoder

    def inference(self, z, nway, nsupport, nquery):
        z_dim = z.size(-1)
        target_inds = torch.arange(0, nway).view(nway, 1, 1).expand(nway, nquery, 1).long().cuda()
        zs = z[:nway * nsupport]
        zq = z[nway * nsupport:]
        z_proto = zs.view(nway, nsupport, z_dim).mean(1)

        dists = euclidean_dist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(nway, nquery, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze(-1)).float().mean()
        return loss_val, acc_val

    def forward(self, data, nway=20, nsupport=4, nquery=1, nmeta=15, num_updates=1, inner_lr=0.1):
        x = data['data']
        assert x.shape[0] == nway*(nsupport + nquery + nmeta)
        inner_loop_data = x[: nway * (nsupport + nquery)].cuda()
        outer_query = x[nway * (nsupport + nquery):]
        outer_support = []
        for cls in range(nway):
            outer_support.extend([x[ind].unsqueeze(0) for ind in range(cls * nsupport, (cls + 1) * nsupport)])
            outer_support.extend(
                [x[nway * nsupport + ind].unsqueeze(0) for ind in range(cls * nquery, (cls + 1) * nquery)])
        outer_support = torch.cat(outer_support, dim=0)
        outer_data = torch.cat([outer_support, outer_query], dim=0).cuda()

        inner_loop_encoder = copy.deepcopy(self.encoder)
        if not inner_loop_encoder.training:
            inner_loop_encoder.train()
        #inner_loop_optimizer = optim.Adam(inner_loop_encoder.parameters(), inner_lr, (0.9, 0.999), 1e-8, weight_decay=0)
        inner_loop_optimizer = optim.SGD(inner_loop_encoder.parameters(), inner_lr)
        with torch.no_grad():
            inner_loop_optimizer.zero_grad()
            z, _ = inner_loop_encoder(outer_data)
            loss, acc = self.inference(z, nway, nsupport=nsupport + nquery, nquery=nmeta)
            first_loss = loss.data.cpu().item()
            first_acc = acc.data.cpu().item()

        inner_loop_loss = []
        inner_loop_acc = []
        for i in range(num_updates):
            inner_loop_optimizer.zero_grad()
            z, _ = inner_loop_encoder(inner_loop_data)
            loss, acc = self.inference(z, nway, nsupport, nquery)
            inner_loop_loss.append(loss.data.cpu().item())
            inner_loop_acc.append(acc.data.cpu().item())
            loss.backward()
            inner_loop_optimizer.step()
        inner_loop_optimizer.zero_grad()
        z, _ = inner_loop_encoder(outer_data)
        loss, acc = self.inference(z, nway, nsupport=nsupport+nquery, nquery=nmeta)
        final_loss = loss.data.cpu().item()
        final_acc = acc.data.cpu().item()
        grads = torch.autograd.grad(loss, inner_loop_encoder.parameters(), only_inputs=True)
        for ind, p in enumerate(self.parameters()):
            if p.grad is not None:
                p.grad.data = grads[ind].data
            else:
                p.grad = grads[ind].data

        return loss, {'loss': final_loss,
                      'acc': final_acc,
                      'first_loss': first_loss,
                      'first_acc': first_acc,
                      'nway': nway,
                      'nquery': nquery,
                      'nsupport': nsupport,
                      'z': z.cpu().data,
                      'encoder_state': None,
                      'inner_loop_loss': inner_loop_loss,
                      'inner_loop_acc': inner_loop_acc,
                      }



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



