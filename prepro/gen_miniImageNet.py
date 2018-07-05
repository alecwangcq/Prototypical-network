import json
import h5py
import csv
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms

from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
        except:
            print(f)
            img = transforms.ToPILImage()(torch.zeros(3, 84, 84))
        return img.convert('RGB')


def load_csv_data(csv_file, ignore_head=True):
    print("Loading csv data from %s" % csv_file)
    start_ix = 1 if ignore_head else 0
    with open(csv_file, 'r') as f:
        data = list(csv.reader(f))[start_ix:]
        print('Before removing empty: %d' % len(data))
    data = [_ for _ in data if len(_) > 0]
    print('After removing empty: %d' % len(data))
    return data


def build_image_name_to_h5id(image_files):
    return {name: idx for idx, name in enumerate(image_files)}


def load_image(image_file, transform=None):
    image = pil_loader(image_file)
    if transform is not None:
        image = transform(image)
    return image


def get_transform():
    transform = transforms.Compose([transforms.Resize((84, 84)),
                                    transforms.ToTensor()])
    return transform


def main(args):
    transform = transforms.Compose([transforms.Resize((84, 84)),
                                    transforms.ToTensor()])
    image_dir = args.data_dir
    output_file = args.output_file
    train_csv = [_[0] for _ in load_csv_data(args.train_csv)]
    val_csv = [_[0] for _ in load_csv_data(args.val_csv)]
    test_csv = [_[0] for _ in load_csv_data(args.test_csv)]
    print("There are %s in train, %s in val, %s in test." % (len(train_csv), len(val_csv), len(test_csv)))
    csvs = train_csv + val_csv + test_csv

    im_to_h5ids = build_image_name_to_h5id(csvs)

    h5file = h5py.File(output_file, 'w')
    output_pool = h5file.create_dataset('images', (len(csvs), 3, 84, 84), dtype=np.float32)
    with open(args.output_idx, 'w') as f:
        json.dump(im_to_h5ids, f)
    for idx, csv in enumerate(csvs):
        file = osp.join(image_dir, csv)
        h5id = im_to_h5ids[csv]
        assert h5id == idx
        image = load_image(file, transform).numpy()
        # import ipdb
        # ipdb.set_trace()
        output_pool[idx] = image
        if idx % 100 == 0:
            print('[%s/%s] Done.' % (idx, len(csvs)))

    print('All done.')
    h5file.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw/mini-imagenet')
    parser.add_argument('--output_file', type=str, default='data/raw/mini-imagenet.h5')
    parser.add_argument('--output_idx', type=str, default='data/raw/mini-imagenet-indices.json')
    parser.add_argument('--train_csv', type=str, default='data/splits/mini_imagenet_split/Ravi/train.csv')
    parser.add_argument('--val_csv', type=str, default='data/splits/mini_imagenet_split/Ravi/val.csv')
    parser.add_argument('--test_csv', type=str, default='data/splits/mini_imagenet_split/Ravi/test.csv')
    args =parser.parse_args()
    main(args)




