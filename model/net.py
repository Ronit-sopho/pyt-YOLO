import torch.nn as nn
import torch.nn.functional as F

# Use the simple rule:
# --- For operations that do not involve trainable parameters
# --- use torch.nn.functional module, as for rest use nn

class _residualBlocks_(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
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
        super(_residualBlocks_, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.Lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.Lrelu_x=nn.LeakyReLU(inplace=True)        # Not sure if a new definition here is required

    def forward(self, x):
        """
        comments
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
    def __init__(self, resBlock, netsize):
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

        downsample = nn.Sequential(
                                    conv_down,
                                    bn_down,
                                    Lrelu_down)
        layers = [downsample]
        in_channels = out_channels
        for i in range(numBlocks):
            layers.append(resBlock(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Lrelu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        
        return out
