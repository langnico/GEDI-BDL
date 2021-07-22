import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self,
                 in_features=1,
                 out_features=(16, 32, 64, 128, 128, 128, 128, 128),
                 num_outputs=1,
                 global_pool=nn.AdaptiveAvgPool1d,
                 max_pool=True,
                 dilation_rate=1):
        """
        The constructor.
        :param in_features: input channel dimension (we say waveforms have channel dimension 1).
        :param out_features: list of channel feature dimensions.
        :param num_outputs: number of the output dimension .
        :param global_pool: the global pooling to use before the fully connected layer.
        :param max_pool: whether to use max pooling or not.
        :param dilation_rate: must be >= 1. Defaults to 1 (no dilation).
        """
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        layers = list()
        for i in range(len(out_features)):
            in_channels = in_features if i == 0 else out_features[i-1]
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_features[i], kernel_size=3,
                                    padding=1, dilation=dilation_rate))
            layers.append(nn.BatchNorm1d(num_features=out_features[i]))
            layers.append(self.relu)
            if max_pool:
                layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = global_pool(output_size=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=out_features[-1], out_features=num_outputs)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = SimpleCNN()
    net.to(device)
    summary(net, input_size=(1, 1420))
