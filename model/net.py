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

        /* Using from : https://github.com/pjreddie/darknet/issues/950  */
                        padding = (kernel_size-1)//2


        """
        super(_residualBlocks_, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.Lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu=nn.LeakyReLU(inplace=True)        # Not sure if a new definition here is required

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
    def __init__(self, netsize):
        """
        Add comments later
        """
        super(Darknet53FeatureExtrator, self).__init__()
        self.conv1 = nn.Conv2d()

    def _netBlock_(numBlocks):
        """
        Add comments later
        """
        self.
