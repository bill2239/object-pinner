from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
class PhoneDataset(Dataset):
    def __init__(self, label_path, root_dir, transform=None):
        """
        Args:
            label_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_arr = [line.strip("\n").split(" ") for line in open(label_path,'r').readlines()]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_arr)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.label_arr[idx][0])
        image = io.imread(img_name)
        #print image.shape
        # raw_input()
        location =[float(self.label_arr[idx][1]),float(self.label_arr[idx][2])] 
        location= np.asarray(location)
        sample = {'image': image, 'location': location}

        if self.transform:
            sample = self.transform(sample)
        # print sample['image'],sample['location']
        
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, location = sample['image'], sample['location']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return {'image': img, 'location': location}




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, location = sample['image'], sample['location']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        location= np.asarray(location)

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'location': torch.from_numpy(location)}