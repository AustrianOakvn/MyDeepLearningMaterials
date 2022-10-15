from lib import *


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        # scale multiply with weight
        self.scale = scale
        self.reset_parameters()

        self.eps = 1e-10

    def reset_parameters(self):
        # all weight would equal to scale
        nn.init.constant(self.weight, self.scale)

    def forward(self, x):
        # L_2 norm
        # x_size (batch, channels, height, width)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.divide(x, norm)
        # weight size (512) -> x_size
        # unsqueeze would screate a new dimension
        weights = self.weight.unsqueeze[0].unsqueeze[2].unsqueeze[3].expand_as(x) # (1, 512, 1, 1) -> x_size

        return weights*x