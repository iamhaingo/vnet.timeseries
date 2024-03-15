import torch.nn as nn


class Encode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encode, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class Decode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decode, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.up = nn.Upsample(scale_factor=2, mode="linear")

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        x = self.up(x)
        return x


class VnetTorch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VnetTorch, self).__init__()
        self.enc1 = Encode(in_channels, 16)
        self.enc2 = Encode(16, 32)
        self.enc3 = Encode(32, 64)
        self.enc4 = Encode(64, 128)
        self.enc5 = Encode(128, 256)
        self.dec1 = Decode(256, 128)
        self.dec2 = Decode(128, 64)
        self.dec3 = Decode(64, 32)
        self.dec4 = Decode(32, 16)
        self.dec5 = Decode(16, num_classes)
        self.skip = nn.Identity()
        self.upsam = nn.Upsample(scale_factor=2, mode="linear")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.enc1(x)
        skip = self.skip(x)
        x = self.enc2(x)
        skip2 = self.skip(x)
        x = self.enc3(x)
        skip3 = self.skip(x)
        x = self.enc4(x)
        skip4 = self.skip(x)
        x = self.enc5(x)
        x = self.dec1(x)
        x = x + self.upsam(skip4)
        x = self.dec2(x)
        x = x + self.upsam(self.upsam(skip3))
        x = self.dec3(x)
        x = x + self.upsam(self.upsam(self.upsam(skip2)))
        x = self.dec4(x)
        x = x + self.upsam(self.upsam(self.upsam(self.upsam(skip))))
        x = self.dec5(x)
        x = nn.functional.softmax(x, dim=1)
        return x.mean(dim=2)
