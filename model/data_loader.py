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
        boxes = np.array([np.array(list(map(float, box.split(',')))) for box in annot[1:]], dtype=np.int32)

        if self.transform:
            image = self.transform(image)

        if self.new_shape is not None:
            image, boxes = self._resize_(image, boxes)

        return (image, boxes)


# --------------Test the custom dataset by plotting some images-------------------------
#---------------Uncomment to plot sample data from the dataset--------------------------

# annotFile = '../data/annotations.txt'
# imageDir = '../data/images/originalPics'
#
# # trans = transforms.Compose([transforms.Resize((416,416))])
# dataHolder = FaceDataset(annotFile, imageDir, transform=None, resize=(416,416))
#
# fig = plt.figure()
# for i in range(len(dataHolder)):
#     sample = dataHolder[i]
#     img = sample[0]
#     boxes = sample[1][0]
#     # print(i, sample['image'].shape, sample['boxes'])
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     rect = mpatches.Rectangle((boxes[1]-boxes[3]//2, boxes[2]-boxes[4]//2), boxes[3], boxes[4], edgecolor='r', fill=False)
#     ax.imshow(img)
#     ax.add_patch(rect)
#     ax.set_title('Sample {}'.format(i))
#     ax.axis('off')
#
#     if i == 3:
#         plt.show()
#         break
#----------------------------------------------
