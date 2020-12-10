import torch

class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, activation = torch.nn.ReLU):
        super().__init__()

        blocks = [
            torch.nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding),
            activation(),
            torch.nn.Conv1d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding),
            activation()
        ]
        self.block = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class CNN(torch.nn.Module):

    def __init__(self, input_shape, output_dim, con_len = 3, con_layers = [25, 50], pooling_len = 3,
                       last_pooling=torch.nn.AvgPool1d, dense_layers = [100, 100], activation = torch.nn.ReLU):
        self.input_shape = input_shape

        super().__init__()

        # Add convolutional layers
        layers = [ConvBlock(in_channels = input_shape[0], out_channels = con_layers[0], kernel_size = con_len, activation = activation)]
        layers += [torch.nn.MaxPool1d(pooling_len)]

        in_shape = con_layers[0]
        for i in range(1,len(con_layers)):
                layers += [ConvBlock(in_channels = in_shape, out_channels = con_layers[i], kernel_size = con_len)]
                in_shape = con_layers[i]

        layers += [
            last_pooling(input_shape[1] // pooling_len),
            torch.nn.Flatten()
        ]

        # Add dense layers
        for i in range(len(dense_layers)):
            layers += [
                torch.nn.Linear(in_shape, dense_layers[i]),
                torch.nn.BatchNorm1d(dense_layers[i], momentum = 0.99),
                activation()
            ]
            in_shape = dense_layers[i]

        layers += [torch.nn.Linear(in_shape, output_dim)]

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
