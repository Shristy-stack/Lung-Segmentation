#unet_model
import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, encoder_name='resnet50', pretrained=True, pretrained_path=None):
        super(UNet, self).__init__()
        
        # Initialize encoder
        if encoder_name == 'resnet50':
            self.encoder = models.resnet50(pretrained=False)
            if pretrained and pretrained_path:
                state_dict = torch.load(pretrained_path)
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
                self.encoder.load_state_dict(state_dict, strict=False)
        
        # Encoder layers
        self.encoder_layers = list(self.encoder.children())
        self.encoder1 = nn.Sequential(*self.encoder_layers[:4])
        self.encoder2 = self.encoder_layers[4]
        self.encoder3 = self.encoder_layers[5]
        self.encoder4 = self.encoder_layers[6]
        self.encoder5 = self.encoder_layers[7]
        
        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder
        d4 = self.upconv4(e5)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenate before applying decoder4
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Concatenate before applying decoder3
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Concatenate before applying decoder2
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        
        # Resize e1 to match d1's dimensions before concatenation
        d1 = nn.functional.interpolate(d1, size=(e1.size(2), e1.size(3)), mode='bilinear', align_corners=True)
        
        d1 = torch.cat([d1, e1], dim=1)  # Concatenate before applying decoder1
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        
        # Resize output to match input size
        out = nn.functional.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        
        return out
