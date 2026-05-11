from .unet_parts import *

class Unet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear = False):
        super(Unet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.input = (DoubleConv(n_channels, 64))
        self.down1 = (DownSampling(64, 128))
        self.down2 = (DownSampling(128, 256))
        self.down3 = (DownSampling(256, 512))

        factor = 2 if bilinear else 1

        self.down4 = (DownSampling(512, 1024 // factor))

        self.up1 = (UpSampling(1024, 512 // factor, bilinear))
        self.up2 = (UpSampling(512, 256 // factor, bilinear))
        self.up3 = (UpSampling(256, 128 // factor, bilinear))
        self.up4 = (UpSampling(128, 64, bilinear))

        self.ouput = (OutConv(64, n_classes))
    
    def forward(self, x):

        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.ouput(x)
        return logits

    def use_checkpointing(self):

        self.input = torch.utils.checkpoint(self.input)

        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)

        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)

        self.output = torch.utils.checkpoint(self.output)

