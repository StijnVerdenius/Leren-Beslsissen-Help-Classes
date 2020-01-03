import torch
import torch.nn as nn


class ResNet(nn.Module):

    def __init__(self, device="cpu", n_classes=2, input_dim=(1, 1, 1)):
        super(ResNet, self).__init__()

        channels, _, _ = input_dim

        self.conv1 = nn.Conv2d(channels,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1).to(device)

        self.layer1 = nn.Sequential(
            BasicBlock(input_dim=64, output_dim=64),
            BasicBlock(input_dim=64, output_dim=64)
        ).to(device)

        self.layer2 = nn.Sequential(
            BasicBlock(input_dim=64, output_dim=128, downsample=True),
            BasicBlock(input_dim=128, output_dim=128)
        ).to(device)

        self.layer3 = nn.Sequential(
            BasicBlock(input_dim=128, output_dim=256, downsample=True),
            BasicBlock(input_dim=256, output_dim=256)
        ).to(device)

        self.layer4 = nn.Sequential(
            BasicBlock(input_dim=256, output_dim=512, downsample=True),
            BasicBlock(input_dim=512, output_dim=512)
        ).to(device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        ).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 downsample=False,
                 groups=1,
                 base_width=64,
                 padding=1,
                 norm_layer=nn.BatchNorm2d,
                 conv_layer=nn.Conv2d):
        super(BasicBlock, self).__init__()

        self._check_input(base_width, groups, padding)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_layer(input_dim, output_dim,
                                kernel_size=3,
                                stride=2 if downsample else 1,
                                padding=padding,
                                groups=groups,
                                bias=False,
                                dilation=padding)
        self.bn1 = norm_layer(output_dim,
                              eps=1e-05,
                              momentum=0.1,
                              affine=True,
                              track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(output_dim, output_dim,
                                kernel_size=3,
                                stride=1,
                                padding=padding,
                                groups=groups,
                                bias=False,
                                dilation=padding)

        self.bn2 = norm_layer(output_dim,
                              eps=1e-05,
                              momentum=0.1,
                              affine=True,
                              track_running_stats=True)
        if downsample:
            downsample = nn.Sequential(
                conv_layer(input_dim, output_dim,
                           kernel_size=1,
                           stride=2,
                           bias=False),
                norm_layer(output_dim,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True)
            )
        self.downsample = downsample
        self.stride = 2 if downsample else 1

    def _check_input(self, base_width, groups, padding):
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if padding > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not False:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


if __name__ == '__main__':
    """ run to test, output should be shaped [5 x 10] = (batch x classes) """
    batch_size = 5
    channels = 3
    spatial_dim = 244
    n_classes = 10
    example_batch_rgb_picture = torch.randn((batch_size, channels, spatial_dim, spatial_dim))
    model = ResNet(n_classes=n_classes, input_dim=(channels, spatial_dim, spatial_dim))
    output = model.forward(example_batch_rgb_picture)
    print(output.shape)
