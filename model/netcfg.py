import torch
import torch.nn as nn

from experiments.helpers import trackOutput, Concatenate

torch.set_printoptions(precision=10)
class modelCfg(nn.Module): 

    def __init__(self, modules_list, yolo_output_indices):
        super(modelCfg, self).__init__()
        self.modules_list = modules_list
        self.yolo_outputs = yolo_output_indices
        self.module_outputs = []

    def forward(self, x):
        for i, l in enumerate(self.modules_list):
            if isinstance(l, nn.Sequential):
                x = l(x)
                self.module_outputs.append(x)
            elif isinstance(l, trackOutput):
                indices = l()
                x = sum([self.module_outputs[i] for i in indices])
                self.module_outputs.append(x)
            elif isinstance(l, Concatenate):
                i1, i2 = l()
                x = torch.cat((self.module_outputs[i1], self.module_outputs[i2]), dim=1)
                self.module_outputs.append(x)
            elif isinstance(l, nn.Upsample):
                x = l(x)
                self.module_outputs.append(x)
            elif l==None:
                self.module_outputs.append(x)
        
        yolo_out = [self.module_outputs[o].permute([0,2,3,1]) for o in self.yolo_outputs]
        self.module_outputs = []
        return yolo_out