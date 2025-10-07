import torch
import torch.nn as nn
from torch.nn.functional import relu

class UNet(nn.Module):
    def __init__(self, num_of_classes, num_of_out_channels, image_size):
        super().__init__()

        # kernel_size
        self.kernel_size_1 = 2
        if image_size != None and image_size[0] % 112 != 0:    # checking if the image size if a multiple of 112
           self.kernel_size_1 = 3
     
        # Encoder
        self.en11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.en12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.5)

        self.en21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.en22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.5)

        self.en31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.en32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.5)

        self.en41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.en42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.5)

        self.en51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(1024)
        self.en52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(1024)
        self.dropout5 = nn.Dropout(0.5)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=self.kernel_size_1, stride=2)
        self.dec11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn_dec11 = nn.BatchNorm2d(512)
        self.dec12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_dec12 = nn.BatchNorm2d(512)
        self.dropout_dec1 = nn.Dropout(0.5)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=self.kernel_size_1, stride=2)
        self.dec21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn_dec21 = nn.BatchNorm2d(256)
        self.dec22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_dec22 = nn.BatchNorm2d(256)
        self.dropout_dec2 = nn.Dropout(0.5)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_dec31 = nn.BatchNorm2d(128)
        self.dec32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_dec32 = nn.BatchNorm2d(128)
        self.dropout_dec3 = nn.Dropout(0.5)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_dec41 = nn.BatchNorm2d(64)
        self.dec42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_dec42 = nn.BatchNorm2d(64)
        self.dropout_dec4 = nn.Dropout(0.5)

        self.outconv = nn.Conv2d(64, num_of_out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        xen11 = relu(self.bn11(self.en11(x)))
        xen12 = relu(self.bn12(self.en12(xen11)))
        xp1 = self.pool1(xen12)
        xp1 = self.dropout1(xp1)

        xen21 = relu(self.bn21(self.en21(xp1)))
        xen22 = relu(self.bn22(self.en22(xen21)))
        xp2 = self.pool2(xen22)
        xp2 = self.dropout2(xp2)

        xen31 = relu(self.bn31(self.en31(xp2)))
        xen32 = relu(self.bn32(self.en32(xen31)))
        xp3 = self.pool3(xen32)
        xp3 = self.dropout3(xp3)

        xen41 = relu(self.bn41(self.en41(xp3)))
        xen42 = relu(self.bn42(self.en42(xen41)))
        xp4 = self.pool4(xen42)
        xp4 = self.dropout4(xp4)

        xen51 = relu(self.bn51(self.en51(xp4)))
        xen52 = relu(self.bn52(self.en52(xen51)))
        xen52 = self.dropout5(xen52)

        # Decoder
        xu1 = self.upconv1(xen52)
        xu11 = torch.cat([xu1, xen42], dim=1)
        xdec11 = relu(self.bn_dec11(self.dec11(xu11)))
        xdec12 = relu(self.bn_dec12(self.dec12(xdec11)))
        xdec12 = self.dropout_dec1(xdec12)

        xu2 = self.upconv2(xdec12)
        xu22 = torch.cat([xu2, xen32], dim=1)
        xdec21 = relu(self.bn_dec21(self.dec21(xu22)))
        xdec22 = relu(self.bn_dec22(self.dec22(xdec21)))
        xdec22 = self.dropout_dec2(xdec22)

        xu3 = self.upconv3(xdec22)
        xu33 = torch.cat([xu3, xen22], dim=1)
        xdec31 = relu(self.bn_dec31(self.dec31(xu33)))
        xdec32 = relu(self.bn_dec32(self.dec32(xdec31)))
        xdec32 = self.dropout_dec3(xdec32)

        xu4 = self.upconv4(xdec32)
        xu44 = torch.cat([xu4, xen12], dim=1)
        xdec41 = relu(self.bn_dec41(self.dec41(xu44)))
        xdec42 = relu(self.bn_dec42(self.dec42(xdec41)))
        xdec42 = self.dropout_dec4(xdec42)

        out = self.outconv(xdec42)

        return out

import engine
engine.run(UNet)