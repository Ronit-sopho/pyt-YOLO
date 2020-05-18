"""
This file implements:
    Network Architecture
    Loss function
    Evaluation metric
"""
# Use the simple rule:
# --- For operations that do not involve trainable parameters
# --- use torch.nn.functional module, as for rest use nn

import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class _residualBlocks_(nn.Module):
    """
    Each Residual Block in Darknet53 consists of:

    (N,C,H,W)
        x--------->conv---->bn---->lrelu------>conv---->bn---->lrelu---->add=======output
         \          c1                          c2                       /
          \     (N,c1,H,W)                  (N,c2,H,W)                  /
           \___________________________________________________________/

    Define the layers here in __init__.
    Connections are defined in forward

    /* Using 'constant padding' from : https://github.com/pjreddie/darknet/issues/950  */
    ----------padding = (kernel_size-1)//2


    """
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        """
        Implementation from:
        https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
        """
        super(_residualBlocks_, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2, affine=True, eps=1e-05, momentum=0.1)
        self.Lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.Lrelu=nn.LeakyReLU(inplace=True)        # Not sure if a new definition here is required

    def forward(self, x):
        """
        Add comments
        """
        residual=x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.Lrelu(out)
        out+=residual
        return out


class Darknet53FeatureExtrator(nn.Module):
    """
    Add comments
    """
    def __init__(self, resBlock, netsize, in_channels, out_channels):
        """
        Its just a bunch of residual blocks stacked together
        """
        super(Darknet53FeatureExtrator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.Lrelu = nn.LeakyReLU(inplace=True)
        self.block1 = self._netBlock_(resBlock, 1, 32, 64)
        self.block2 = self._netBlock_(resBlock, 2, 64, 128)
        self.block3 = self._netBlock_(resBlock, 8, 128, 256)
        self.block4 = self._netBlock_(resBlock, 8, 256, 512)
        self.block5 = self._netBlock_(resBlock, 4, 512, 1024)


    def _netBlock_(self, resBlock, numBlocks, in_channels, out_channels):
        """
        First downsample the input to make it compatible with Add
        """
        conv_down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        bn_down = nn.BatchNorm2d(out_channels)
        Lrelu_down = nn.LeakyReLU(inplace=True)
        downsample = nn.Sequential(conv_down,bn_down,Lrelu_down)

        layers = [downsample]
        in_channels = out_channels
        for i in range(numBlocks):
            layers.append(resBlock(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        returns 3 different outputs, which are fed into different parts of the final layers
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Lrelu(out)
        out = self.block1(out)
        out = self.block2(out)
        sc1 = self.block3(out)   #------> Scale 1 output
        sc2 = self.block4(sc1)   #------> Scale 2 output
        sc3 = self.block5(sc2)   #------> Scale 3 output
        return sc1, sc2, sc3


class YOLOv3Net(nn.Module):
    """
    Yolo uses residual blocks(a lot of them)
    It generates output at 3 scales.
    Network consists of skip connections, concatenations and upsampling
    """
    def __init__(self, num_classes, num_anchors):
        super(YOLOv3Net, self).__init__()
        self.features = Darknet53FeatureExtrator(_residualBlocks_, (416,416), 3, 32)
        self.route1, self.mapping1 = self.lastBlocks(384, 128, num_classes, num_anchors)
        self.route2, self.mapping2 = self.lastBlocks(768, 256, num_classes, num_anchors)
        self.route3, self.mapping3 = self.lastBlocks(1024, 512, num_classes, num_anchors)
        self.conv_match1 = nn.Conv2d(256,128, kernel_size=1, stride=1, padding=0)
        self.conv_match2 = nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0)
        self.upscale = nn.Upsample(scale_factor=2)


    def lastBlocks(self, in_channels, out_channels, num_classes, num_anchors):
        """
        The final layers in YOLO after feature extractors
        consists of 7 total convolutinal layers.
        Output from the 5th conv layer is fed into different part of the network.
        """
        conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
        bn1 = nn.BatchNorm2d(out_channels)
        Lrelu = nn.LeakyReLU(inplace=True)
        conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1)
        bn2 = nn.BatchNorm2d(out_channels*2)
        # self.Lrelu = nn.LeakyReLU(inplace=True)  Defined for completeness
        conv_out = nn.Conv2d(out_channels*2, num_anchors//3*(5+num_classes), kernel_size=1, stride=1, padding=0)

        lastBlock = [
                conv_in, bn1, Lrelu,
                conv2, bn2, Lrelu,
                conv1, bn1, Lrelu,
                conv2, bn2, Lrelu,
                conv1, bn1, Lrelu,
                ]
        lastLayer = [
                conv2, bn2, Lrelu,
                conv_out
                ]

        return nn.Sequential(*lastBlock), nn.Sequential(*lastLayer)

    def forward(self, x):
        """
        Final 3 outputs will be of the shape:
        out1 = N, num_anchors*(5+num_classes), h//8, w//8
        out2 = N, num_anchors*(5+num_classes), h//16, w//16
        out3 = N, num_anchors*(5+num_classes), h//32, w//32
        where,
        N:batch_size
        h,w: height and width of net respectively

        """
        sc1, sc2, sc3 = self.features(x)
        route3 = self.route3(sc3)
        out3 = self.mapping3(route3)

        match2 = self.conv_match2(route3)
        up2 = self.upscale(match2)
        cat2 = torch.cat((up2,sc2),dim=1)
        route2 = self.route2(cat2)
        out2 = self.mapping2(route2)

        match1 = self.conv_match1(route2)
        up1 = self.upscale(match1)
        cat1 = torch.cat((up1,sc1),dim=1)
        route1 = self.route1(cat1)
        out1 = self.mapping1(route1)

        return {0:out1.permute([0,2,3,1]), 1:out2.permute([0,2,3,1]), 2:out3.permute([0,2,3,1])}

#--------------- Uncomment to check implementation for errors---------------
# model = YOLOv3Net(2,9)
# input = torch.randn((1,3,416,416))
# input = torch.zeros((1,3,416,416))
# y_out = model(input)
# print(y_out[0].shape,y_out[1].shape,y_out[2].shape)
#---------------------------------------------------------------------------

class Yolov3Loss(nn.Module):
    def __init__(self, anchors, num_classes, input_size):
        """
        Yolo divides an image into grid cells.
        Network output at different scales requires different grid sizes
        """
        self.anchors = np.array(anchors)
        self.num_anchors = len(anchors)
        self.n_scales = self.num_anchors//6
        self.h, self.w = input_size
        self.mask = [[0,1,2], [3,4,5], [6,7,8]]
        self.scaling_factors = [8,16,32]
        self.grid_sizes = [self.h//self.scaling_factors[x] for x in range(self.n_scales)]
        self.ignore_threshold = 0.5
        self.truth_threshold = 0.7

    def processYoloOutput(self, yolo_output, mask_index):
        """
        This description is wrong change is later
        yolo_output: [y1, y2, y3]

        Might have to use sigmoid later

        y1: (b, h//8, w//8, 3 * 5+num_classes)
        y2: (b, h//16, w//16, 3 * 5+num_classes)
        y3: (b, h//32, w//32, 3 * 5+num_classes)

        """
        h,w = self.h, self.w
        mask = self.mask[mask_index]
        b,gy,gx,_ = yolo_output.shape
        # out = torch.zeros(yolo_output.shape)
        anchors = self.anchors.reshape(-1,2)
        # out = out.view(b,gy,gx,3,-1)
        out = yolo_output.view(b,gy,gx,3,-1)
        # print("out shape: ", out.shape)
        # print(out)
        xx, yy = np.meshgrid(np.arange(gx), np.arange(gy))
        xx = torch.from_numpy(xx).to(device)
        # xx = xx*(w/gx)
        # yy = yy*(h/gy)
        yy = torch.from_numpy(yy).to(device)
        xx = xx.view(1,xx.shape[0],xx.shape[1],1)
        yy = yy.view(1,yy.shape[0],yy.shape[1],1)
        xx = xx.repeat(b,1,1,out.shape[-2])
        yy = yy.repeat(b,1,1,out.shape[-2])
        # print("xx shape: ", xx.shape)
        # print(out.dtype)
        # out[...,0]+=torch.tensor(xx, dtype=out.dtype)
        out[...,4] = torch.sigmoid(out[...,4])
        # print(out[...,4])
        out[...,5:] = torch.sigmoid(out[...,5:])

        out[...,0] = torch.sigmoid(out[...,0])
        out[...,0]+=xx.float()
        out[...,0]/=gx
        # out[...,1]+=torch.tensor(yy, dtype=out.dtype)
        out[...,1] = torch.sigmoid(out[...,1])
        out[...,1]+=yy.float()
        out[...,1]/=gy

        for i,prior in enumerate(anchors[mask]):
            out[...,i,2] = torch.exp(out[...,i,2])*prior[0]/w
            out[...,i,3] = torch.exp(out[...,i,3])*prior[1]/h

        # out = out.to('cpu')
        # out = out.detach().numpy()
        # for i,prior in enumerate(anchors[mask]):
        #     out[...,i,2] = np.exp(out[...,i,2])*prior[0]/w
        #     out[...,i,3] = np.exp(out[...,i,3])*prior[1]/h
        # out = torch.from_numpy(out).to(device)
        # print("out dtype: ", out[1,1,1,1,:])
        # print('out grad find', out.requires_grad)
        return out

    def loss(self, y_out, y_truth):
        """
        """
        def boxIOU(mat1, mat2):
            """
            mat1 : (b, gy, gx, 3, num_gt_boxes, 4)
            mat2 : (b, gy, gx, 3, 4+1+num_classes)

            return mat.shape = (b,gy,gx,3,num_gt_boxes)
            """
            mat2 = mat2.unsqueeze(4) # Add dimension at axis 4
            # print("Mat 1 shape:", mat1[0,0,0,0,:])
            # print("Mat 2 shape: ",mat2[0,0,0,0,:])

            xtl1 = mat1[...,:,0] - mat1[...,:,2]/2
            ytl1 = mat1[...,:,1] - mat1[...,:,3]/2
            xbr1 = mat1[...,:,0] + mat1[...,:,2]/2
            ybr1 = mat1[...,:,1] + mat1[...,:,3]/2

            xtl2 = mat2[...,:,0] - mat2[...,:,2]/2
            ytl2 = mat2[...,:,1] - mat2[...,:,3]/2
            xbr2 = mat2[...,:,0] + mat2[...,:,2]/2
            ybr2 = mat2[...,:,1] + mat2[...,:,3]/2

            # print("diff: ", mat2[...,:,0] - mat2[...,:,2]/2)

            # print("mat2 vals: ", mat2[...,:,0])
            # print("mat2 vals: ", mat2[...,:,2])
            # print("xtl2 type: ", xtl2)
            # print("xbr2 type: ", xbr2[0,0,0,:,:])
            union_areas = (xbr1-xtl1)*(ybr1-ytl1) + (xbr2-xtl2)*(ybr2-ytl2)
            # print("Union_areas shape: ", union_areas)

            xmin_mask = xtl1[...,:]>xtl2[...,:]
            ymin_mask = ytl1[...,:]>ytl2[...,:]
            xmax_mask = xbr1[...,:]<xbr2[...,:]
            ymax_mask = ybr1[...,:]<ybr2[...,:]

            xtl1[xmin_mask==0] = xtl1[xmin_mask==0] - (xtl1[...,:]-xtl2[...,:])[xmin_mask==0]
            ytl1[ymin_mask==0] = ytl1[ymin_mask==0] - (ytl1[...,:]-ytl2[...,:])[ymin_mask==0]
            xbr1[xmax_mask==0] = xbr1[xmax_mask==0] - (xbr1[...,:]-xbr2[...,:])[xmax_mask==0]
            ybr1[ymax_mask==0] = ybr1[ymax_mask==0] - (ybr1[...,:]-ybr2[...,:])[ymax_mask==0]

            intersection_areas = (xbr1-xtl1)*(ybr1-ytl1)
            # print(intersection_areas.shape)

            return intersection_areas/(union_areas-intersection_areas)


        # y1, y2, y3 = [y_out[i] for i in range(3)]
        # y1_t, y2_t, y3_t = [y_truth[i] for i in range(3)]
        loss = 0

        for i in range(len(y_out)):

            y_out_ = self.processYoloOutput(y_out[i], i).requires_grad_(True) # y_out_ shape = (b,gy,gx,3,5+num_cls)
            # print('yout grad: ', y_out_.requires_grad)
            b,gy,gx,_,_ = y_out_.shape
            y_truth_ = y_truth[i].to(device)
            # print(y_truth_.device)
            # print('This is i', i)
            # print('Tu tu ru...', y_truth_.shape)
            # take ground truth boxes of every anchor boxes in that scale
            # this slicing wont work with torch :(
            # Sry, it will work :)
            gt = []
            for k in range(3):
                gt.append(y_truth[k][y_truth[k][...,:,4]==1][...,:4])
                # gtboxes = y_truth_[y_truth_[...,:,4]==1][...,:4] #gtboxes shape = (num_gt_boxes, 4)
            gtboxes = torch.cat(gt)
            gtboxes = gtboxes.to(device)
            # print(gtboxes.device)
            gtboxes_ = gtboxes.view(1,1,1,1,gtboxes.shape[0],gtboxes.shape[1]) #shape now = (1,1,1,1,num_gt_boxes,4)
            gtboxes_ = gtboxes_.repeat(b,gy,gx,3,1,1) #shape now = (b,gy,gx,3,num_gt_boxes,4)
            biou = boxIOU(gtboxes_, y_out_) # biou shape : (b,gy, gx, 3, num_gt_boxes)
            assign_mask = torch.zeros(b,gy,gx,3).to(device)
            # print("BIOU shape: ", biou.shape)
            for j in range(gtboxes.shape[0]):
                # print(biou[...,j].max())
                assign_mask[biou[...,j]==biou[...,j].max()]=1


            values, indices = biou.max(4) # Get maximum along the num_gt_box dimensions
            ignore_mask = values>self.ignore_threshold
            truth_mask = values>self.truth_threshold
            object_mask = y_truth_[...,4]
            box_loss_scale = 2-y_truth_[...,2]*y_truth_[...,3] 

            #This could be wrong, will check later
            # print('ytruth shape: ', y_truth_[...,:2].shape)
            # print('yout shape: ', y_out_[...,:2].shape)
            # print('assign shape: ', assign_mask.shape)
            # print('boxloss shape: ', box_loss_scale.shape)
            # print(assign_mask.dtype)
            assign_mask = assign_mask
            box_loss_scale = box_loss_scale
            y_truth_ = y_truth_
            y_out_ = y_out_
            object_mask = object_mask
        # print(y_out_[...,4])
            xy_loss = ((y_truth_[...,:2]- y_out_[...,:2])**2)*assign_mask.unsqueeze(-1)*box_loss_scale.unsqueeze(-1)
            wh_loss = ((y_truth_[...,2:4] - y_out_[...,2:4])**2)*assign_mask.unsqueeze(-1)*box_loss_scale.unsqueeze(-1)*0.5
            objectness_loss = object_mask*F.binary_cross_entropy_with_logits(y_out_[...,4], object_mask) + (1-assign_mask)*(1-object_mask)*F.binary_cross_entropy_with_logits(y_out_[...,4], object_mask)
            cls_loss = object_mask*F.binary_cross_entropy_with_logits(y_out_[...,5:], y_truth_[...,5:])

            xy_loss = xy_loss.sum()/b
            wh_loss = wh_loss.sum()/b
            objectness_loss = objectness_loss.sum()/b
            cls_loss = cls_loss.sum()/b

            loss+=xy_loss+wh_loss+objectness_loss+cls_loss
        # print('loss grad check:',loss, loss.requires_grad)

        return loss