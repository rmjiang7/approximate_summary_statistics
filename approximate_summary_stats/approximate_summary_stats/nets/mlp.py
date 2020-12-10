import torch

class MLP(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dims, activation = torch.nn.ReLU):

        super().__init__()

        layers = []
        in_shape = in_dim
        for i in range(len(hidden_dims)):
            layers += [
                torch.nn.Linear(in_shape, hidden_dims[i]),
                activation()
            ]
            in_shape = hidden_dims[i]
        layers += [torch.nn.Linear(in_shape, out_dim)]

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
