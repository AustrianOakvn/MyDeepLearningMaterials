import torch
import torch.nn as nn


# For max pooling kernel size is 2 and stride is 2

vgg_16_architecture_config = [
        (3, 64, 1, 1),
        "M",
        (3, 128, 1, 1),
        "M",
        (3, 256, 1, 1),
        (3, 256, 1, 1),
        "M",
        (3, 512, 1, 1),
        (3, 512, 1, 1),
        "M",
        (3, 512, 1, 1),
        (3, 512, 1, 1),
        "M"
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class VGG(nn.Module):
    def __init__(self, architecture_config, in_channels=3, out_features=10, **kwargs):
        super(VGG, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.out_features = out_features
        self.conv_net = self._create_conv_net(self.architecture)
        self.fcs = self._create_fcs(**kwargs)



    def forward(self, x):
        x = self.conv_net(x)
        return self.fcs(torch.flatten(x, start_dim=1))


    def _create_conv_net(self, architecture):
        layers = []
        in_channels = self.in_channels

        for l in architecture:
            if type(l) == tuple:
                layers += [CNNBlock(in_channels, l[1], kernel_size=l[0], stride=l[2], padding=l[3],)]
                in_channels = l[1]

            elif type(l) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

                

    def _create_fcs(self):
        return nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, 4096),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, self.out_features))



def test():
    vgg_16 = VGG(vgg_16_architecture_config)
    x = torch.rand((2, 3, 224, 224))
    print(vgg_16(x).shape)

test()

