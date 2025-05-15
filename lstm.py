import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation_name):
    """Returns a callable activation function given its name."""
    name = activation_name.lower()
    if name == "relu":
        return F.relu
    elif name == "tanh":
        return torch.tanh
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "linear":
        return lambda x: x
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")



class NewLstmModule(nn.Module):

    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.activation = get_activation("linear")
        self.fc = nn.Linear(in_features=hidden_size,out_features=1) #Dense

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        x = self.fc(output[-1, :])
        x = self.activation(x)
        return x


class ManyLstmModule(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=False)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.activation = get_activation("linear")

    def forward(self, X):
        """
        Expected X shape if batch_size=1:
            (seq_len, 1, n_features)

        Output shape:
            (seq_len, 1)
        """
        output, (hn, cn) = self.lstm(X)  # output: (seq_len, batch_size=1, hidden_size)
        x = self.fc(output)  # x: (seq_len, 1, 1)
        x = self.activation(x)

        # Remove the batch dimension (batch_size=1)
        x = x.squeeze(dim=1)  # x: (seq_len, 1)
        return x
