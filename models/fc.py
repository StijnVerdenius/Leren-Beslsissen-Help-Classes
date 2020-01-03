import torch
import torch.nn as nn


class FC(nn.Module):

    """ fully connected (FC) network implementation """

    def __init__(self, device="cpu", hidden_dim=2, n_classes=2, in_features=2):
        super(FC, self).__init__()

        # define network
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim, bias=True),  # weight matrix
            nn.LeakyReLU(0.05),  # nonlinearity
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
            nn.LeakyReLU(0.05),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
            nn.LeakyReLU(0.05),
            nn.Linear(in_features=hidden_dim, out_features=n_classes, bias=True)
        ).to(device)  # put network parameters to gpu or cpu

    def forward(self, x: torch.Tensor):
        # flattens any tensor into 2d: [batch-size x other-dims]
        x = x.view(x.shape[0], -1)
        return self.layers.forward(x)


if __name__ == '__main__':

    """ run to test, output should be shaped [5 x 10] = (batch x classes) """
    batch_size = 5
    features = 3000
    n_classes = 10
    example_batch_flat = torch.randn((batch_size, features))
    model = FC(n_classes=n_classes, in_features=features)
    output = model.forward(example_batch_flat)
    print(output.shape)