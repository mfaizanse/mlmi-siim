import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils.mask_functions import rle2mask
import pydicom
import numpy as np

class SIMMDataset(Dataset):
    """SIMM dataset."""

    def __init__(self, root_dir, split, transform=None, preload_data=False):
        """
        Args:
            dicomPaths (Array<string>): Array of DICOM file Paths.
            mask_csv_file (string): csv file with encoded masks (rle).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.im_height = 1024
        self.im_width = 1024
        self.im_chan = 1

        ## Read masks file
        mask_csv_file = root_dir + '/train-rle.csv' 
        self.encodedMasks = pd.read_csv(mask_csv_file, names=['ImageId', 'EncodedPixels'], index_col='ImageId')

        ## Read dataset file names
        dsFile = root_dir + '/simm_DS_' + split + '.csv'
        dsFileData = pd.read_csv(dsFile)
        self.dicomPaths = dsFileData['path'].tolist()

        self.transform = transform

    def __len__(self):
        return len(self.dicomPaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dPath = self.dicomPaths[idx]
        dicom = pydicom.dcmread('.' + dPath)
        
#         image = np.zeros((1, im_height, im_width, im_chan), dtype=np.uint8)
#         image = np.expand_dims(dicom.pixel_array, axis=2)
        image = np.array(dicom.pixel_array)
        
        # get mask (in rle) from csv
        landmarks = np.zeros((self.im_height, self.im_width), dtype=np.bool)
        
        fileId = dPath.split('/')[-1][:-4]
        rle = self.encodedMasks.loc[fileId, 'EncodedPixels']
        try:
            if type(rle) == str: # if single rle
                decodedRle = rle2mask(rle, self.im_height, self.im_width)
#                 landmarks = np.expand_dims(decodedRle, axis=2)
                landmarks = decodedRle
            else: # if multiple rle
                for x in rle:
                    decodedRle = rle2mask(x, self.im_height, self.im_width)
                    landmarks = landmarks + decodedRle
#                     landmarks = landmarks + np.expand_dims(decodedRle, axis=2)
        except Exception as e:
            print(e)
            
        #### TODO - IMPORTANT::: CHECK THIS  
        ## QUESTION: SHOULD WE TRANSPOSE THE MASK IN THE GETITEM FUNCTION 
        ## BECAUSE WHEN PLOTING THE GRAPHS WE HAVE TO TRANSPOSE IT.
        landmarks = landmarks.T

        # for some images, we have multiple masks, so we are adding the masks
        # which results in some pixels to > 1
        landmarks = (landmarks >= 1).astype('float64')
            
        sample = {'image': image, 'mask': landmarks}

        if self.transform:
            sample = self.transform(sample)

        img = np.expand_dims(sample['image'], axis=0)
        mk = np.expand_dims(sample['mask'], axis=0)

        return img, mk