import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, 
                 frame_encoder: nn.Module,
                 latent_frame_size: int,
                 encoding_size: int,
                 hidden_state_size=32,
                 num_lstm_layers=1):
        super(VideoEncoder, self).__init__()

        self.hidden_state_size = hidden_state_size

        self.frame_encoder = frame_encoder

        self.linear_in = nn.Linear(latent_frame_size, hidden_state_size)
        self.lstm = nn.LSTM(hidden_state_size, hidden_state_size, num_layers=num_lstm_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_state_size, encoding_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        batch_size, seq_length, height, width = x.shape

        with torch.no_grad():
            x = self.frame_encoder(x.view(batch_size * seq_length, height, width)).view(batch_size, seq_length, -1)

        x = self.linear_in(x)
        x = self.activation(x)

        x, _ = self.lstm(x)

        x = self.activation(x)
        x = self.linear_out(x)
        x = self.activation(x)

        return x

class ConvVideoEncoder(nn.Module):
    def __init__(self, 
                 encoding_size: int,
                 hidden_state_size=32,  # Further reduced hidden state size
                 num_lstm_layers=1):
        super(ConvVideoEncoder, self).__init__()

        self.hidden_state_size = hidden_state_size

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # Reduced number of filters
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear_in = nn.Linear(16 * 16 * 16, hidden_state_size)  # Adjusted to match output size
        self.lstm = nn.LSTM(hidden_state_size, hidden_state_size, num_layers=num_lstm_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_state_size, encoding_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        if x.shape[2] != 3:
            x = torch.permute(x, (0, 1, 4, 3, 2))
        batch_size, seq_length, channels, height, width = x.shape

        x = x.view(batch_size * seq_length, channels, height, width)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)  # Apply pooling

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(batch_size, seq_length, -1)

        x = self.linear_in(x)
        x = self.activation(x)

        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)

        x = self.activation(x)
        x = self.linear_out(x)
        x = self.activation(x)

        return x
