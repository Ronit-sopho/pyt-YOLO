"""
Define a custom 'Dataset', which will be loaded by
torch.utils.data.Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, utils
from matplotlib import pyplot as plt
import os
import matplotlib.patches as mpatches


class FaceDataset(Dataset):

    def __init__(self, annot_file, image_dir, transform=None, resize=None):
        """
        """
        self.annotations = []
        self.image_dir = image_dir
        self.transform = transform
        self.new_shape = resize
        with open(annot_file,'r') as f:
            for line in f:
                self.annotations.append(line.rstrip())

    def _resize_(self, img, labels):
        """
        Currently only supports tuples, not functional for single integer input of shape

        PIL .size returns width and height, unlike cv2/np shape function
        """
        w,h = img.size
        H,W = self.new_shape[0], self.new_shape[1]
        img = img.resize((H,W))
        for l in labels:
            coords = l[1:].reshape(2,2)
            new_coords = coords*(np.array([W/w, H/h]).reshape(1,2))
            l[1:] = new_coords.flatten()

        return img, labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        annot = self.annotations[idx].split()
        img_name = os.path.join(self.image_dir, annot[0]+'.jpg')
        image = Image.open(img_name)
        image = image.convert('RGB')

        # Since arrays with different dimensions cannot be stacked together
        # along a new dimension, define a fixed size array with zeros
        # E.g- Array of (1,5) cannot be stacked with (3,5) along a new dimension of batch
        # Choose a maximum of 15 boxes per image
        MAX_SIZE = 15
        BOXES = np.zeros((MAX_SIZE,5))
        boxes = np.array([np.array(list(map(float, box.split(',')))) for box in annot[1:]], dtype=np.int32)


        if self.new_shape is not None:
            image, boxes = self._resize_(image, boxes)

        if self.transform:
            image = self.transform(image)

        BOXES[:boxes.shape[0]] = boxes
        sample = {'image':np.asarray(image), 'label':BOXES}


        return sample


# --------------Test the custom dataset by plotting some images-------------------------
#---------------Uncomment to plot sample data from the dataset--------------------------
# 
# annotFile = '../data/annotations.txt'
# imageDir = '../data/images/originalPics'
#
# trans = transforms.Compose([transforms.ToTensor()])
# dataset= FaceDataset(annotFile, imageDir, transform=None, resize=(416,416))
# dataHolder = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
# plt.style.use('dark_background')
# fig = plt.figure()
#
# for i,data in enumerate(dataHolder):
#     bs = data['image'].shape[0]
#     # print('Batch size: {}'.format(bs))
#     imgs = data['image']
#     labels = data['label']
#     for j in range(bs):
#         ax = plt.subplot(3, bs, i*bs+(j+1))
#         plt.tight_layout()
#         boxes = labels[j]
#         for box in boxes:
#             rect = mpatches.Rectangle((box[1]-box[3]//2, box[2]-box[4]//2), box[3], box[4], edgecolor='r', fill=False)
#             ax.add_patch(rect)
#         img  = imgs[j]
#         ax.imshow(img)
#         ax.set_title('Sample {}'.format(i*bs+(j+1)))
#         ax.axis('off')
#
#     if i == 2:
#         plt.show()
#         break
#----------------------------------------------
