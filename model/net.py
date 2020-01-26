import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import torch

# Use the simple rule:
# --- For operations that do not involve trainable parameters
# --- use torch.nn.functional module, as for rest use nn

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
        Add comments
        """
        super(_residualBlocks_, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.Lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.Lrelu_x=nn.LeakyReLU(inplace=True)        # Not sure if a new definition here is required

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
        Add comments later
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
        Add comments later
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
        Add comments
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
    """
    def __init__(self):
        super(YOLOv3Net, self).__init__()
        self.features = Darknet53FeatureExtrator(_residualBlocks_, (416,416), 3, 32)
        self.route1, self.mapping1 = self.lastBlocks(384, 128)
        self.route2, self.mapping2 = self.lastBlocks(768, 256)
        self.route3, self.mapping3 = self.lastBlocks(1024, 512)
        self.conv_match1 = nn.Conv2d(256,128, kernel_size=1, stride=1, padding=0)
        self.conv_match2 = nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0)
        self.upscale = nn.Upsample(scale_factor=2)


    def lastBlocks(self, in_channels, out_channels):
        """
        """
        conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
        bn1 = nn.BatchNorm2d(out_channels)
        Lrelu = nn.LeakyReLU(inplace=True)
        conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1)
        bn2 = nn.BatchNorm2d(out_channels*2)
        # self.Lrelu = nn.LeakyReLU(inplace=True)  Defined for completeness
        convL = nn.Conv2d(out_channels*2, 255, kernel_size=1, stride=1, padding=0)

        lastBlock = [
                conv_in, bn1, Lrelu,
                conv2, bn2, Lrelu,
                conv1, bn1, Lrelu,
                conv2, bn2, Lrelu,
                conv1, bn1, Lrelu,
                ]
        lastLayer = [
                conv2, bn2, Lrelu,
                convL
                ]

        return nn.Sequential(*lastBlock), nn.Sequential(*lastLayer)

    def forward(self, x):
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
        return out1, out2, out3

# Check for errors
model = YOLOv3Net()
input = torch.rand(1,3,416,416)
y1,y2,y3 = model(input)
print(y1.shape,y2.shape,y3.shape)
