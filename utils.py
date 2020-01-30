"""
Utility functions for handling hyperparams/logging/storing model

Also handling data
"""

import numpy as np
import os


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



dataset = FDDBDataset()
dataset.readEllipsefiles()
