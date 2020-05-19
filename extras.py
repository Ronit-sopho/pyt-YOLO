"""
Utility functions for generating annotaions, anchors, converting weights

Also handling data
"""

import numpy as np
import os, pprint
from sklearn.cluster import KMeans
from collections import OrderedDict
import torch.nn as nn
import torch

from experiments.helpers import trackOutput, Concatenate
from model.netcfg import modelCfg
import argparse

def cfg_to_dict(path_cfg):
    """
    Dictionary keys should be in the same oder as the input cfg file
    Use OrderedDict
    """

    cfg_dict = OrderedDict()
    counts = {}
    with open(path_cfg, 'r') as f:
        for line in f:
            l = line.rstrip()
            if l.startswith('#') or len(l)==0:
                continue
            if l.startswith('['):
                heading = l[1:-1]
                c = counts.get(heading, 0)
                if c==0:
                    counts[heading]=1
                else:
                    counts[heading]+=1
                cfg_dict[heading+'_'+str(counts[heading])] = {}
                
            else:
                k, v = l.split('=')
                cfg_dict[heading+'_'+str(counts[heading])][k.strip()] = v.strip()
    return cfg_dict


class FDDBDataset():
    """
    Convert Annotations of FDDB face dataset to be compatible with YOLO
    ====> Convert elliptical bounding boxes to rectangular
    ====> Generate single annotation file
    """

    def __init__(self):
        self.annot_dir = './data/training_data/FDDB-folds'

    def readEllipsefiles(self):
        self.annot_write =  open('./data/annotations.txt','a')
        self.annot_dir_list = sorted(os.listdir(self.annot_dir))
        self.annot_dir_ellipse_list = [f for f in self.annot_dir_list if "ellipseList" in f]
        for file in self.annot_dir_ellipse_list:
            f_struct = ['path', 'n', 'nlines']
            ptr = 0
            with open(os.path.join(self.annot_dir,file), 'r') as f:
                for line in f:
                    if ptr==0:
                        holder = ""
                        holder+=line.rstrip()
                        ptr=1
                    elif ptr==1:
                        count=0
                        n = int(line)
                        ptr=2
                    elif ptr==2:
                        l = self.ellipse2rect(line.rstrip())
                        if count<n-1:
                            holder = holder + " " + l
                            count+=1
                        elif count==n-1:
                            ptr=0
                            holder = holder + " " + l + '\n'
                            self.annot_write.write(holder)
        self.annot_write.close()

    def ellipse2rect(self, components):
        """
        For conversion refer to:
        https://math.stackexchange.com/questions/91132/how-to-get-the-limits-of-rotated-ellipse

        return string in yolo annotation format ===> class_id, centre_x, centre_y, width, height
        """
        # The last one in labels contains additional space
        major_axis, minor_axis, angle, centre_x, centre_y,_,_ = components.split(" ")
        major_axis = float(major_axis)
        minor_axis = float(minor_axis)
        angle = float(angle)

        width = 2 * (np.sqrt(np.square(major_axis*np.cos(angle))+np.square(minor_axis*np.sin(angle))))
        height = 2 * (np.sqrt(np.square(minor_axis*np.cos(angle))+np.square(major_axis*np.sin(angle))))
        return ",".join([str(0), centre_x, centre_y, str(width), str(height)])



class KnnAnchors():

    def __init__(self, path_annots, num_anchors):
        self.annot_file = path_annots
        self.num_anchors = num_anchors

    def read_annots(self):
        boxes_wh = []
        with open(self.annot_file, 'r') as f:
            for line in f:
                annotation = line.rstrip().split()
                for box in annotation[1:]:
                    boxes_wh.append(np.array(list(map(float, box.split(',')))[-2:]))
        return np.array(boxes_wh)

    def kmeans_clustering(self):
        boxes_wh = self.read_annots()
        kmeans = KMeans(n_clusters=self.num_anchors, random_state=0).fit(boxes_wh)
        return kmeans.cluster_centers_


class ConvertWeights():
    """
    Checkout --> https://github.com/pjreddie/darknet/blob/master/src/parser.c

    Check load_weights_upto(net, filename, start, cutoff) function for more insight.

    Following things are written while saving the weight file:
    1) major - 1 sizeof(int)
    2) minor - 1 sizeof(int)
    3) revision - 1 sizeof(int)
    4) seen - 1 either sizeof(int) or sizeof(size_t)
    """

    def __init__(self, path_to_weights, path_to_cfg):

        from model.net import YOLOv3Net

        self.path_to_weights = path_to_weights
        self.data_buffer = open(self.path_to_weights, 'rb')
        self.cfg_dict = cfg_to_dict(path_to_cfg)
        # pp = pprint.PrettyPrinter(indent=2)
        # pp.pprint(self.cfg_dict)

    def save_weights(self):
        """
        Model needs to be redefined from cfg file
        Since number of classes differ for pretrained weights

        For BatchNormalizatin Pytorch does not use shifts ==> beta and gamma

        Bias is not added in Conv2D layers in Darknet if BatchNorm is used
        https://github.com/pjreddie/darknet/blob/master/src/convolutional_layer.c#L445
        """

        major = np.frombuffer(self.data_buffer.read(4), dtype='int32', count=1)
        minor = np.frombuffer(self.data_buffer.read(4), dtype='int32', count=1)
        revision = np.frombuffer(self.data_buffer.read(4), dtype='int32', count=1)

        if (major*10+minor)>=2 and major<1000 and minor<1000:
            seen = np.frombuffer(self.data_buffer.read(8), dtype='int64', count=1)
        else:
            seen = np.frombuffer(self.data_buffer.read(4), dtype='int32', count=1)

        print('Weight file headers # major :{} # minor :{} # revision :{} # seen :{}'.format(major, minor, revision, seen))
        transpose = (major>1000) or (minor>1000)

        # Collect layers from cfg file and initialize model with weights
        model_layers = nn.ModuleList()
        prev_layer_shape = (None,None,3) # Start with input layer (416,416,3)
        count=0
        yolo_output = []

        # Iterate through ordered dict to load layer weights and biases
        for key, value in self.cfg_dict.items():

            layer = key
            print('Reading {} layer...'.format(layer))

            if layer.startswith('convolutional'):
                # Read parameters
                activation = value['activation']
                batch_normalize = int(value.get('batch_normalize',0))
                filters = int(value['filters'])
                pad = int(value['pad'])
                size = int(value['size'])
                stride = int(value['stride'])

                num = prev_layer_shape[-1]*filters*size*size # Assuming no groups
                print('.....Reading biases 4*{}'.format(filters))
                # Read biases
                biases = np.frombuffer(self.data_buffer.read(4*filters), dtype=np.float32, count=filters) # beta for BN

                if batch_normalize:
                    # Read the following ==> scales, rolling mean, rolling variance in that order
                    scales = np.frombuffer(self.data_buffer.read(4*filters), dtype=np.float32, count=filters) # gamma
                    rolling_mean = np.frombuffer(self.data_buffer.read(4*filters), dtype=np.float32, count=filters) # running_mean
                    rolling_variance = np.frombuffer(self.data_buffer.read(4*filters), dtype=np.float32, count=filters) # running_var
                
                print('.....Reading weights {}'.format(num))
                weights = np.frombuffer(self.data_buffer.read(4*num), dtype=np.float32, count=num)
                weights = weights.reshape(filters, prev_layer_shape[-1], size, size) # This is how weight are serialized in darknet
                # weights = np.transpose(weights, [1,0,2,3]) # Convert to pytorch format ==> (out_channels, in_channels//num_groups, size, size)
                

                if batch_normalize:
                    layer_conv = nn.Conv2d(prev_layer_shape[-1], filters, kernel_size=size, stride=stride, padding=(size-1)//2, bias=False)
                    layer_conv.weight.data = torch.Tensor(weights)
                    layer_bn = nn.BatchNorm2d(filters, affine=True, eps=0.0001, momentum=0.03)
                    layer_bn.weight.data = torch.Tensor(scales)
                    layer_bn.bias.data = torch.Tensor(biases)
                    layer_bn.running_mean = torch.Tensor(rolling_mean)
                    layer_bn.running_var = torch.Tensor(rolling_variance)
                    layers = [layer_conv, layer_bn]
                else:
                    layer_conv = nn.Conv2d(prev_layer_shape[-1], filters, kernel_size=size, stride=stride, padding=(size-1)//2, bias=True)
                    layer_conv.weight.data = torch.Tensor(weights)
                    layer_conv.bias.data = torch.Tensor(biases)
                    layers = [layer_conv]

                if activation=='leaky':
                    layers.append(nn.LeakyReLU(negative_slope=0.1,inplace=True))
                elif activation=='linear':
                    pass

                prev_layer_shape = (None, None, filters)
                model_layers.extend([nn.Sequential(*layers)])
                print(weights.shape)

            elif layer.startswith('shortcut'):
                # Read parameters
                activation = value['activation']
                from_ = int(value['from'])
                model_layers.extend([trackOutput(from_, -1)])

            elif layer.startswith('route'):
                # Read parameters
                route_layers = value['layers'].split(',')
                if len(route_layers)==1:
                    prev_layer_shape = (None, None, model_layers[int(route_layers[0])][0].weight.size()[0])
                    model_layers.extend([trackOutput(int(route_layers[0]))])
                else:
                    prev_layer_shape = (None, None, prev_layer_shape[-1] + model_layers[int(route_layers[1])+1][0].weight.size()[1])
                    model_layers.extend([Concatenate(int(route_layers[0]), int(route_layers[1]))])

            elif layer.startswith('upsample'):
                # Read parameters
                scaling = int(value['stride'])
                model_layers.extend([nn.Upsample(scale_factor=scaling)])

            elif layer.startswith('yolo'):
                # Read parameters - no need
                yolo_output.append(len(model_layers)-1)
                model_layers.extend([None])

            elif layer.startswith('net'):
                pass

            else:
                print('Unknown section :( ,  check cfg file!')
                break

        leftovers = np.frombuffer(self.data_buffer.read())
        print('Leftover weights : {}'.format(len(leftovers)/4))
        # print(model_layers)

        YOLOv3model = modelCfg(model_layers, yolo_output)
        print('Saving model and weights...')
        torch.save(YOLOv3model, './logs/yolov3.pt')



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_anchors", help="generate bounding box priors using k-means clustering", action="store_true")
    parser.add_argument("--gen_annotations", help="generate yolo annotations from training data lables", action="store_true", nargs='+')
    parser.add_argument("--convert", help="convert darknet weights to pytorch format", nargs='+')

    args = parser.parse_args()

    if args.gen_anchors:
        path_to_annot_file = args.gen_anchors[0]
        num_anchors = args.gen_anchors[1]
        print('Generating {} anchors from {}'.format(num_anchors, path_to_annot_file))
        knn = KnnAnchors(path_to_annot_file, num_anchors)
        anchors = knn.kmeans_clustering()
        anchors = sorted(anchors, key=lambda x: max(x[0],x[1]))
        print(list(anchors))

    else if args.gen_annotations:
        # This is specific to FDDB face dataset
        dataset = FDDBDataset()
        dataset.readEllipsefiles()

    else if args.convert:
        path_to_weights = args.convert[1]
        path_to_cfg = args.convert[0]
        cv = ConvertWeights(path_to_weights, path_to_cfg)
        cv.save_weights()

    else if:
        print('No argument found :(')


if __name__ == '__main__':
    main()
