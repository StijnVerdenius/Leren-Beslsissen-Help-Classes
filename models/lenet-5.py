import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """ lenet5-CNN (pictures) network implementation """

    def __init__(self, device="cpu", n_classes=2, input_dim=(1, 1, 1)):
        super(LeNet5, self).__init__()

        # convention with pictures is: [batchsize x channels x spatial-dim1 x spatial-dim2]
        channels, dim1, dim2 = input_dim

        post_conv_dim1 = (dim1 // 4) // 4
        post_conv_dim2 = (dim2 // 4) // 4

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU(),
        ).to(device)

        self.fc = nn.Sequential(
            nn.Linear(120 * post_conv_dim1 * post_conv_dim2, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        ).to(device)

    def forward(self, x: torch.Tensor):
        """ input shape = [batch x channel x dim1 x dim2] """

        # a convnet can process images (batch-dim + 3d data)
        x = self.conv.forward(x)

        # then switch back to fully connected, we need it to be (batch-dim + 1d)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    """ run to test, output should be shaped [5 x 10] = (batch x classes) """
    batch_size = 5
    channels = 3
    spatial_dim = 28
    n_classes = 10
    example_batch_rgb_picture = torch.randn((batch_size, channels, spatial_dim, spatial_dim))
    model = LeNet5(n_classes=n_classes, input_dim=(channels, spatial_dim, spatial_dim))
    output = model.forward(example_batch_rgb_picture)
    print(output.shape)
    assert output.shape == torch.randn((5, 10)).shape, "final shape is wrong"

