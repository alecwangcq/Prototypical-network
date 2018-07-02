import tqdm
import torch
import csv
import os
import os.path as osp
import random

from collections import defaultdict

if __name__ == '__main__':
    from base import MultiProcessImageLoader
else:
    from .base import MultiProcessImageLoader


class MiniImageNetDataset(object):

    def __init__(self, csv_dir, split, image_dir, shuffle=True, num_threads=2, transform=None):
        files = os.listdir(osp.join(csv_dir, split))
        self.do_shuffle = shuffle
        self.image_dir = image_dir
        self.split_names = [f.split('.')[0] for f in files]
        self.split_csvs = dict()
        self.cat_to_files = dict()
        self.cat_ixs = dict()
        self.image_loader = MultiProcessImageLoader(num_threads, transform)
        for sp in self.split_names:
            csv_file = osp.join(csv_dir, split, sp+".csv")
            self.split_csvs[sp] = self.load_csv_data(csv_file)
            self.cat_to_files[sp] = self.make_categories_to_files(self.split_csvs[sp])
            self.cat_ixs[sp] = dict()
            for cat in self.cat_to_files[sp].keys():
                self.cat_ixs[sp][cat] = 0
        self.print_stats()

        for sp in self.split_names:
            for cat in self.cat_to_files[sp].keys():
                self.shuffle(sp, cat)

    def print_stats(self):
        print('Shuffle   = %s' % self.do_shuffle)
        print('image_dir = %s' % self.image_dir)
        print('split_names = ', self.split_names)
        for sp in self.split_names:
            counts = defaultdict(lambda: 0)
            csvs = self.split_csvs[sp]
            print('In [%-8s], there are %s samples.' % (sp, len(csvs)))
            print('In [%-8s], there are %s categories.' % (sp, len(self.cat_ixs[sp].keys())))
            for cat in self.cat_ixs[sp].keys():
                counts[cat] = len(self.cat_to_files[sp][cat])
            res = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            print('In [%-8s], %s: %d, %s: %d.' % (sp, res[0][0], res[0][1], res[-1][0], res[-1][1]))

    @staticmethod
    def make_categories_to_files(tuples):
        cat_to_file = defaultdict(list)
        for f, c in tuples:
            cat_to_file[c].append(f)
        return cat_to_file

    def load_csv_data(self, csv_file, ignore_head=True):
        print("Loading csv data from %s" % csv_file)
        start_ix = 1 if ignore_head else 0
        with open(csv_file, 'r') as f:
            data = list(csv.reader(f))[start_ix:]
            print('Before removing empty: %d' % len(data))
        data = [_ for _ in data if len(_) > 0]
        print('After removing empty: %d' % len(data))
        return data

    def shuffle(self, split, cat):
        random.shuffle(self.cat_to_files[split][cat])

    def reset_iterator(self, split, cat):
        self.cat_ixs[split][cat] = 0
        if self.do_shuffle:
            self.shuffle(split, cat)

    def update_iterator(self, split, cat):
        ix = self.cat_ixs[split][cat]
        ix += 1
        if ix == len(self.cat_to_files[split][cat]):
            self.reset_iterator(split, cat)
        else:
            self.cat_ixs[split][cat] = ix

    def fetch_images(self, split, cat, num):
        image_files = []
        for _ in range(num):
            ix = self.cat_ixs[split][cat]
            file = self.cat_to_files[split][cat][ix]
            image_files.append(file)
            self.update_iterator(split, cat)
        return image_files

    def sample_categories(self, split, nway) -> list:
        categories = list(self.cat_ixs[split].keys())
        random.shuffle(categories)
        return categories[:nway]

    def load_images(self, image_files, transform) -> list:
        image_files = [osp.join(self.image_dir, _) for _ in image_files]
        images = self.image_loader.load_images(image_files, transform)
        images, file_names = images[0], images[1]
        assert all(image_files[_] == file_names[_] for _ in range(len(image_files)))
        return images

    def get_episode(self, nway=5, nshot=1, nquery=20, split='train', transform=None):
        categories = self.sample_categories(split, nway)
        images = dict()
        support_data = []
        support_lbls = []
        support_cats = []
        query_data = []
        query_lbls = []
        query_cats = []
        for lbl, cat in enumerate(categories):
            images[cat] = self.fetch_images(split, cat, nshot+nquery)
            supports = images[cat][:nshot]
            queries = images[cat][nshot:]
            support_data.extend(supports)
            query_data.extend(queries)
            support_lbls.extend([lbl] * nshot)
            query_lbls.extend([lbl] * nquery)
            support_cats.extend([cat] * nshot)
            query_cats.extend([cat] * nquery)

        data_files = support_data + query_data
        data = self.load_images(data_files, transform)
        lbls = support_lbls + query_lbls
        cats = support_cats + query_cats

        return {'data': data,
                'label': torch.LongTensor(lbls),
                'file': data_files,
                'category': cats,
                'nway': nway,
                'nshot': nshot,
                'nquery': nquery,
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

    mini_loader = MiniImageNetDataset(csv_dir, split,
                                      image_dir, shuffle,
                                      num_threads, transform)
    for idx in tqdm.tqdm(range(1000)):
        data = mini_loader.get_episode(5, 5, 15)
        import ipdb
        ipdb.set_trace()



