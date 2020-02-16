import os
import torch
import random
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms
from utils import show_landmarks, rotate
import cv2
from PIL import Image

scale = 160
width, height = 162, 162


color_jit = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, train=True):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.resize = transforms.Resize(output_size)
        self.train = train

    def __call__(self, sample):
        img = sample['image']
        if self.train:
            sample['landmarks']
        w, h = img.size[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        if self.train:
            img = color_jit(img)
        img = self.resize(img)
        
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if self.train:
            landmarks = landmarks * [new_w / w, new_h / h]
            return {'image': img, 'landmarks': landmarks}
        else:
            return {'image': img}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # for the size of images are different, so random crop
        # should include all the landmark
        image, landmarks = sample['image'], sample['landmarks']

        w, h  = image.size
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image.crop((left, top, left+new_w, top+new_h))
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        if 'landmarks' in sample:
            landmarks = sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = transforms.ToTensor()(image)

        if 'landmarks' not in sample:
            return {'image': image}
        return {'image': image,
                'landmarks': (torch.from_numpy(landmarks)).flatten()}


video = r"D:\BaiduNetdiskDownload\关键点视频\gather"
landmarks_v = pickle.load(open(os.path.join(video, 'landmark.pkl'), 'rb'))
video_dir = glob.glob(video+r"\*.png")[0:-1: 20]


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if not test:
            self.landmarks_frame = pickle.load(open(os.path.join(root_dir, 'landmark.pkl'), 'rb'))
            self.landmarks_frame = dict(self.landmarks_frame.items(), **landmarks_v)
            
            self.root_dir = glob.glob(root_dir+r"\*.png") + video_dir
        else:
            self.root_dir = glob.glob(root_dir+r"\*.jpg")
        # for r in self.root_dir:
        #     Image.open(r).convert('RGB').save(r)
        if not test:
            self.transform = transforms.Compose([
            Rescale((width, height)),
            RandomCrop((scale, scale)),
            RandomRotate(10),
            ToTensor(),
            Normalize([ 0.485, 0.456, 0.406 ],
                            [ 0.229, 0.224, 0.225])
        ])
        else:
            self.transform = transforms.Compose([
                Rescale((scale, scale),  train=False),
                ToTensor(),
                Normalize([ 0.485, 0.456, 0.406 ],
                                [ 0.229, 0.224, 0.225 ])
            ])

        self.test = test

    def __len__(self):
        if not self.test:
            return len(self.root_dir)
        else:
            return len(self.root_dir)

    def __getitem__(self, idx):
        img_name = self.root_dir[idx]
        image = Image.open(img_name)
        index = img_name.split("\\")[-1].split('.')[0]

        if not self.test:
            if "_" in index:
                landmarks = self.landmarks_frame[index]
            else:
                landmarks = self.landmarks_frame[int(index)]

            landmarks = landmarks.astype('float').reshape(-1, 2)
            sample = {'image': image, 'landmarks': landmarks}
        else:
            sample = {'image': image, 'landmarks': np.random.rand(68, 2)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip(object):
    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']
        if random.random()<0.5:
            image = cv2.flip(image,0)
            landmarks[:,0] = image.shape[1]-landmarks[:,0]
        return {'image': image, 'landmarks': landmarks}


class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
        self.norm = transforms.Normalize(mean=[0.5], std=[0.5])
    def __call__(self, sample):
        import copy
        image = sample['image']
        sample['real_img'] = copy.deepcopy(image)
        # for t, m, s in zip(image, self.mean, self.std):
        #     t.sub_(m).div_(s)
        # image = image.float() - 127.
        # image = image.float() / 127.
        # sample['image'] = self.norm(image)
        return sample


class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle
    
    def __call__(self, data):
        angle = random.randint(0, self.angle)
        if random.random() < 0.5:
            angle = -angle
        image, landmarks = Rotate_aug(data['image'], angle, data['landmarks'])
        data['image'] = image
        data['landmarks'] = landmarks
        return data

def Rotate_aug(src,angle,label=None,center=None,scale=1.0):
    w, h = src.size
    center = (w / 2., h / 2.)
    image = src.rotate(angle, False, False, center)
    # image.show()
    for i in range(label.shape[0]):
        x, y = rotate(angle, label[i, 0], label[i, 1], center[0], center[1])
        label[i, 0] = x
        label[i, 1] = y
    
    return image, label


test_transform = transforms.Compose([
            Rescale((scale, scale), train=False),
            ToTensor(),
            Normalize([ 0.485, 0.456, 0.406 ],
                            [ 0.229, 0.224, 0.225 ])
        ])