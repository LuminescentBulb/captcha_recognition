import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class CaptchaModel(nn.Module):
    def __init__(self):
        """
        Initializes the CNN + RNN model layers
        """
        super(CaptchaModel, self).__init__()

        # --- CNN Feature Extractor ---
        # Input: (B, 1, 90, 280)

        # Layer 1
        self.conv1 = nn.Conv2d(config.IMAGE_CHANNELS, 32, kernel_size=5, padding='same')
        # Output: (B, 32, 90, 280) -RELU and pooling-> (B, 32, 45, 140)

        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        # Output: (B, 64, 45, 140) -RELU and pooling-> (B, 64, 22, 70)


        # Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        # Output: (B, 128, 22, 70) -RELU and pooling-> (B, 128, 11, 35)

        # Layer 4 (more complexity if needed)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        # Output: (B, 256, 5, 17)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # CNN Output Shape for RNN
        # After Layer 3: Channels=128, Height=11, Width=35
        self.cnn_output_channels = 128
        self.cnn_output_height = 11
        self.cnn_output_width = 35 # sequence length for the RNN
        rnn_input_size = self.cnn_output_channels * self.cnn_output_height # 128 * 11 = 1408

        # --- RNN Sequence Processor ---
        self.rnn_hidden_size = 256
        self.rnn_num_layers = 2

        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=True,
            batch_first=True    # input shape (batch_size, seq_len, features)
        )

        # --- Output Classifier ---
        rnn_output_size = self.rnn_hidden_size * 2

        self.output_layer = nn.Linear(
            self.cnn_output_width * rnn_output_size, # 35 * 512 = 17920
            config.CAPTCHA_LENGTH * config.NUM_CHARACTERS # 6 positions * 62 characters = 372
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model, explicitly calling each CNN layer.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, CAPTCHA_LENGTH, NUM_CHARACTERS)
            representing logits for each character position.
        """
        batch_size = x.size(0)

        # --- CNN ---
        # Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Shape: (B, 32, 45, 140)

        # Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Shape: (B, 64, 22, 70)

        # Layer 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        # Shape: (B, 128, 11, 35)

        # Layer 4
        # x = self.conv4(x)
        # x = self.relu(x)
        # x = self.pool(x)

        # --- RNN ---
        x = x.permute(0, 3, 1, 2) # -> (B, 35, 128, 11)
        x = x.view(batch_size, self.cnn_output_width, -1)
        # Shape: (B, 35, 1408)

        # Pass through RNN
        rnn_output, _ = self.lstm(x)
        # Shape: (B, 35, 512)

        # Prepare for Output Layer
        rnn_output_flat = rnn_output.contiguous().view(batch_size, -1)
        # Shape: (B, 35 * 512) -> (B, 17920)

        # Pass through the final classification layer
        output = self.output_layer(rnn_output_flat)
        # Shape: (B, 372)

        # Reshape the output
        output = output.view(batch_size, config.CAPTCHA_LENGTH, config.NUM_CHARACTERS)
        # Shape: (B, 6, 62)

        return output