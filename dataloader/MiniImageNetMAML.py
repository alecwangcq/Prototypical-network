import tqdm
import torch
import csv
import os
import os.path as osp
import random
import json
import h5py
import time

from collections import defaultdict

if __name__ == '__main__':
    from MiniImageNet import MiniImageNetDataset, TransformedImageLoader, h5load
    from base import MultiProcessImageLoader
else:
    from .base import MultiProcessImageLoader
    from .MiniImageNet import MiniImageNetDataset, TransformedImageLoader, h5load


def shuffle_slice(a, start, stop):
    assert stop <= len(a)
    i = start
    while (i < stop-1):
        idx = random.randrange(i, stop)
        a[i], a[idx] = a[idx], a[i]
        i += 1


class MiniImageNetMAMLDataset(MiniImageNetDataset):

    def __init__(self,
                 csv_dir,
                 split,
                 image_dir,
                 shuffle=True,
                 num_threads=2,
                 transform=None,
                 transformed_images=None,
                 imname_index_file=None,
                 separate_metasample={'train':True, 'test':False, 'val':False},
                 metasample_holdout=200,
                 ):

        super(MiniImageNetMAMLDataset, self).__init__(csv_dir, split, image_dir, shuffle, num_threads, transform,
                                                      transformed_images, imname_index_file)

        assert len(self.split_names) == len(separate_metasample)
        self.separate_metasample = separate_metasample
        #for i in range(len(separate_metasample)):
        #    self.separate_metasample.update({self.split_names[i]: separate_metasample[i]})
        self.metasample_holdout = metasample_holdout

        self.cat_ixs_meta = dict()      # holds current read index for the meta-update images

        for sp in self.split_names:
            if self.separate_metasample[sp]:
                self.cat_ixs_meta[sp] = dict()
            else:
                self.cat_ixs_meta[sp] = self.cat_ixs[sp]            # make the meta index refer to the normal index
            if self.separate_metasample[sp]:
                for cat in self.cat_to_files[sp].keys():
                    self.cat_ixs_meta[sp][cat] = len(self.cat_to_files[sp][cat]) - self.metasample_holdout

    def shuffle(self, split, cat):
        # shuffle all samples of given split and category
        if 'separate_metasample' in self.__dict__:
            if self.separate_metasample[split]:
                # shuffle separately
                self.shuffle_sq(split, cat)
                self.shuffle_meta(split, cat)
        else:
            random.shuffle(self.cat_to_files[split][cat])

    def shuffle_sq(self, split, cat):
        #print('sq shuffle', split, cat)
        num_files = len(self.cat_to_files[split][cat])
        shuffle_slice(self.cat_to_files[split][cat], 0, num_files - self.metasample_holdout)

    def shuffle_meta(self,split, cat):
        #print('meta shuffle', split, cat)
        num_files = len(self.cat_to_files[split][cat])
        shuffle_slice(self.cat_to_files[split][cat], num_files - self.metasample_holdout, num_files)

    def update_sq_iterator(self, split, cat):
        ix = self.cat_ixs[split][cat]
        ix += 1
        maxlen = len(self.cat_to_files[split][cat])
        if self.separate_metasample[split]:
            maxlen -= self.metasample_holdout
        if ix == maxlen:
            self.cat_ixs[split][cat] = 0
            if self.do_shuffle:
                if self.separate_metasample[split]:
                    self.shuffle_sq(split, cat)
                else:
                    self.shuffle(split, cat)
        else:
            self.cat_ixs[split][cat] = ix

    def update_meta_iterator(self, split, cat):
        ix = self.cat_ixs_meta[split][cat]
        ix += 1

        if ix == len(self.cat_to_files[split][cat]):
            if self.separate_metasample[split]:
                self.cat_ixs_meta[split][cat] = len(self.cat_to_files[split][cat]) - self.metasample_holdout
                if self.do_shuffle:
                    self.shuffle_meta(split, cat)
            else:
                self.cat_ixs_meta[split][cat] = 0
                if self.do_shuffle:
                    self.shuffle(split, cat)
        else:
            self.cat_ixs_meta[split][cat] = ix

    def fetch_images_MAML(self, split, cat, num, metanum):
        image_files = []
        for _ in range(num):
            ix = self.cat_ixs[split][cat]
            file = self.cat_to_files[split][cat][ix]
            image_files.append(file)
            self.update_sq_iterator(split, cat)
        for _ in range(metanum):
            ix = self.cat_ixs_meta[split][cat]
            file = self.cat_to_files[split][cat][ix]
            image_files.append(file)
            self.update_meta_iterator(split, cat)
        return image_files

    def get_episode_MAML(self, nway=5, nshot=4, nquery=1, nmeta=15, split='train', transform=None):
        categories = self.sample_categories(split, nway)
        images = dict()
        support_data = []
        support_lbls = []
        support_cats = []
        query_data = []
        query_lbls = []
        query_cats = []
        metasample_data = []
        metasample_lbls = []
        metasample_cats = []
        for lbl, cat in enumerate(categories):
            images[cat] = self.fetch_images_MAML(split, cat, nshot+nquery, nmeta)
            supports = images[cat][:nshot]
            queries = images[cat][nshot:nshot+nquery]
            metasamples = images[cat][nshot+nquery:]

            support_data.extend(supports)
            query_data.extend(queries)
            metasample_data.extend(metasamples)

            support_lbls.extend([lbl] * nshot)
            query_lbls.extend([lbl] * nquery)
            metasample_lbls.extend([lbl] * nmeta)

            support_cats.extend([cat] * nshot)
            query_cats.extend([cat] * nquery)
            metasample_cats.extend([cat] * nmeta)

        data_files = support_data + query_data + metasample_data
        data = self.load_images(data_files, transform)
        lbls = support_lbls + query_lbls + metasample_lbls
        cats = support_cats + query_cats + metasample_cats

        return {'data': data,
                'label': torch.LongTensor(lbls),
                'file': data_files,
                'category': cats,
                'nway': nway,
                'nshot': nshot,
                'nquery': nquery,
                'nmeta': nmeta,
                'split': split}


if __name__ == '__main__':
    import torchvision.transforms as transforms
    csv_dir = 'data/splits/mini_imagenet_split'
    split = 'Ravi'
    image_dir = 'data/raw/mini-imagenet'
    shuffle = True
    num_threads = 4
    transform = transforms.Compose([transforms.Resize((84, 84)),
                                    transforms.ToTensor()])
    separate_metasample = {'train': True, 'test': True, 'val': False}
    mini_loader = MiniImageNetMAMLDataset(csv_dir, split,
                                      image_dir, shuffle,
                                      num_threads, transform, separate_metasample=separate_metasample)
    for idx in tqdm.tqdm(range(100)):
        data = mini_loader.get_episode_MAML(20, 5, 15, 5, split='test')

        #data = mini_loader.get_episode_MAML(5, 4, 1, 15, split='test')
        #import ipdb
        #ipdb.set_trace()
    for idx in tqdm.tqdm(range(1000)):
        data = mini_loader.get_episode_MAML(5, 4, 1, 15, split='test')



