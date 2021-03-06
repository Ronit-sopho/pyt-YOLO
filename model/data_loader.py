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
import os, time
import matplotlib.patches as mpatches

MAX_BOXES = 30

def letterbox_image(org_img, size):
    """
    Resize image while keeping the aspect ratio same
    """
    img_w, img_h = org_img.size
    w, h = size
    scaling_factor = min(w/img_w, h/img_h)
    new_w = int(scaling_factor*img_w)
    new_h = int(scaling_factor*img_h)

    org_img = org_img.resize((new_w, new_h), Image.LINEAR)
    resized_img = Image.new('RGB', size, (128,128,128))
    resized_img.paste(org_img, ((w-new_w)//2, (h-new_h)//2))
    return resized_img

def YoloLabel(labels, input_shape, anchors, num_classes, batch_size):
    """
    Current implementation is really slow :(

    labels: (batch_size, MAX_BOXES, 5)
    input_shape: (net_h, net_w)
    anchors: np.array([])
    num_classes: int

    Generate ground truth labels from bounding boxes of the images
     y1,   y2,    y3 are the GT labels for respective grid sizes of
    h//8, h//16, h//32
    Reminder----> check for correctness once again
    """
    def _iou_(box1, boxes2):
        b1 = box1
        ious = []
        for b2 in boxes2:
            xmin = max(b1[0],b2[0])
            xmax = min(b1[2],b2[2])
            ymin = max(b1[1],b2[1])
            ymax = min(b1[3],b2[3])
            intersection_area = (ymax-ymin)*(xmax-xmin)
            b1_area = (b1[3]-b1[1])*(b1[2]-b1[0])
            b2_area = (b2[3]-b2[1])*(b2[2]-b2[0])
            ious.append(intersection_area/(b1_area+b2_area-intersection_area))
        return ious

    net_h, net_w = input_shape
    masks = np.array([[0,1,2],[3,4,5],[6,7,8]])
    batch_size=batch_size   # Reminder -------> Read it from data
    y1 = torch.zeros(batch_size,net_h//8,net_w//8,3,(num_classes+4+1))
    y2 = torch.zeros(batch_size,net_h//16,net_w//16,3,(num_classes+4+1))
    y3 = torch.zeros(batch_size,net_h//32,net_w//32,3,(num_classes+4+1))

    xx1,yy1 = net_w//8, net_h//8
    xx2,yy2 = net_w//16, net_h//16
    xx3,yy3 = net_w//32, net_h//32

    grids = {0:(xx1,yy1),1:(xx2,yy2),2:(xx3,yy3)}
    y_truth = {0:y1,1:y2,2:y3}

    # Continue with loops for now, use broadcasting later on
    for i in range(len(labels)):
        boxes = labels[i]
        pos = []
        for box in boxes:
            cls,xb,yb,w,h = box
            x,y=0,0
            if w*h!=0:
                anchor_boxes = [(x-anchors[i]//2,y-anchors[i+1]//2,x+anchors[i]//2,y+anchors[i+1]//2) for i in range(0,len(anchors)//2,2)]
                IOUs = _iou_((x-w//2,y-h//2,x+w//2,y+h//2), anchor_boxes)
                pos = np.argwhere(masks==np.argmax(IOUs))
                r = int(xb/net_w * grids[pos[0,0]][0])
                c = int(yb/net_h * grids[pos[0,0]][1])
                if r==grids[pos[0,0]][0]:
                    r-=1
                if c==grids[pos[0,0]][1]:
                    c-=1
                y_truth[pos[0,0]][i][r][c][pos[0,1]][0:4] = torch.tensor([xb/net_w, yb/net_h, w/net_w, h/net_h])
                y_truth[pos[0,0]][i][r][c][pos[0,1]][4] = 1
                y_truth[pos[0,0]][i][r][c][pos[0,1]][5+int(cls)] = 1

    return y_truth


class FaceDataset(Dataset):
    """
    Returns a data dictionary:
    data = {
        'image': torch tensor of size equal to "resize"
        'label': (MAX_BOXES,5) #5==id,x,y,w,h
        }
    """

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
        # Choose a maximum of 30 boxes per image
        BOXES = np.zeros((MAX_BOXES,5))
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
# imageDir = '../data/training_data/images/originalPics'
# anchors = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]

# trans = transforms.Compose([transforms.ToTensor()])
# start = time.time()
# dataset= FaceDataset(annotFile, imageDir, transform=None, resize=(416,416))
# dataHolder = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
# end = time.time()
# print('Time taken for loading dataset: ',end-start)
# plt.style.use('dark_background')
# fig = plt.figure()

# for i,data in enumerate(dataHolder):
#     bs = data['image'].shape[0]
#     # print('Batch size: {}'.format(bs))
#     imgs = data['image']
#     labels = data['label']
#     start = time.time()
#     yoloBoxes = YoloLabel(labels, (416,416), anchors, 1, 32)
#     end = time.time()
#     print('Time taken for conversion: ', end-start)
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
#     if i == 0:
#         plt.show()
#         break
#----------------------------------------------
