import ipdb
import torch
import torch.utils.data as data

import torchvision.transforms as transforms

import PIL.Image as Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
        except:
            img = transforms.ToPILImage()(torch.zeros(3, 96, 96))
        return img.convert('RGB')


class MultiProcessImageLoader:

    class _Pool(data.Dataset):
        def __init__(self, image_files, transform=None):
            self.files = image_files
            self.transform = transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            file = self.files[index]
            image = pil_loader(file)
            if self.transform is not None:
                image = self.transform(image)
            return image, file

    def __init__(self, num_threads, transform):
        self.num_threads = num_threads
        self.transform = transform

    def load_images(self, files, transform=None):
        """ Load Images
        Args:
            files: list
            transform: object
        Return:
            a list of images.
        """
        if transform is None:
            transform = self.transform
        loader = data.DataLoader(dataset=self._Pool(files, transform),
                                 batch_size=len(files),
                                 shuffle=False,
                                 num_workers=self.num_threads)
        images = next(loader.__iter__())
        return images


def convert_dict(k, v):
    return {k: v}


class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k, v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data


class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


if __name__ == '__main__':
    print('Testing')
    image_files = ['n1313361300001299.jpg', 'n1313361300001297.jpg',
                   'n1313361300001267.jpg', 'n1313361300001242.jpg',
                   'n1313361300001241.jpg', 'n1313361300001270.jpg',
                   'n1313361300001276.jpg', 'n1313361300001192.jpg']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    normalize])
    loader = MultiProcessImageLoader(4, transform)
    # ipdb.set_trace()
