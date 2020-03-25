"""
Utility functions for handling hyperparams/logging/storing model

Also handling data
"""

import numpy as np
import os
from sklearn.cluster import KMeans

class FDDBDataset():
    """
    Convert Annotations of FDDB face dataset to be compatible with YOLO
    ====> Convert elliptical bounding boxes to rectangular
    ====> Generate single annotation file
    """

    def __init__(self):
        self.annot_dir = './data/FDDB-folds'

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

# Uncomment for generating YOLO annotation file (run once)
# dataset = FDDBDataset()
# dataset.readEllipsefiles()

# Uncomment for running KMeans Clustering
# path_to_annot_file = './data/annotations.txt'
# num_anchors = 9 # for yolo v3
# knn = KnnAnchors(path_to_annot_file, num_anchors)
# anchors = knn.kmeans_clustering()
# anchors = sorted(anchors, key=lambda x:max(x[0],x[1]))
# print(list(anchors))
