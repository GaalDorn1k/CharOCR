import torch
from torch import nn
from torch.nn.functional import softmax, relu


class CharOCRBase(nn.Module):
    def __init__(self, alphabet_len: int, fields_num: int, base_channels: int,
                 dropout_rate: float, mode = 'eval') -> None:
        super(CharOCRBase, self).__init__()
        self.mode = mode

        '''
        Encoder
        '''
        self.dropout_rate = dropout_rate
        self.enc_conv0 = nn.Conv2d(3, base_channels // 2, kernel_size=3, padding=1)
        self.enc_batchnorm0 = nn.BatchNorm2d(base_channels // 2)
        self.enc_convStride0 = nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=2)
        self.enc_batchnorm1 = nn.BatchNorm2d(base_channels)
        self.enc_conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.enc_batchnorm2 = nn.BatchNorm2d(base_channels)
        self.enc_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.enc_batchnorm3 = nn.BatchNorm2d(base_channels)
        self.enc_drop1 = nn.Dropout(p=dropout_rate)
        self.enc_convStride1 = nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, stride=2)
        self.enc_batchnorm4 = nn.BatchNorm2d(2 * base_channels)
        self.enc_conv3 = nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1)
        self.enc_batchnorm5 = nn.BatchNorm2d(2 * base_channels)
        self.enc_conv4 = nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1)
        self.enc_batchnorm6 = nn.BatchNorm2d(2 * base_channels)
        self.enc_drop2 = nn.Dropout(p=dropout_rate)
        self.enc_convStride2 = nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=3, stride=2)
        self.enc_batchnorm7 = nn.BatchNorm2d(4 * base_channels)
        self.enc_conv5 = nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=3, dilation=2, padding=2)
        self.enc_batchnorm8 = nn.BatchNorm2d(4 * base_channels)
        self.enc_conv6 = nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=3, dilation=2, padding=2)
        self.enc_batchnorm9 = nn.BatchNorm2d(4 * base_channels)
        self.enc_drop3 = nn.Dropout(p=dropout_rate)
        self.enc_conv7 = nn.Conv2d(4 * base_channels, 8 * base_channels, kernel_size=3, dilation=4, padding=4)
        self.enc_batchnorm10 = nn.BatchNorm2d(8 * base_channels)
        self.enc_conv8 = nn.Conv2d(8 * base_channels, 8 * base_channels, kernel_size=3, dilation=4, padding=4)
        self.enc_batchnorm11 = nn.BatchNorm2d(8 * base_channels)
        self.enc_conv9 = nn.Conv2d(8 * base_channels, 8 * base_channels, kernel_size=3, dilation=4, padding=4)
        self.enc_batchnorm12 = nn.BatchNorm2d(8 * base_channels)
        self.enc_conv10 = nn.Conv2d(8 * base_channels, 8 * base_channels, kernel_size=3, dilation=8, padding=8)
        self.enc_batchnorm13 = nn.BatchNorm2d(8 * base_channels)
        self.enc_conv11 = nn.Conv2d(8 * base_channels, 8 * base_channels, kernel_size=3, dilation=8, padding=8)
        self.enc_batchnorm14 = nn.BatchNorm2d(8 * base_channels)
        self.enc_conv12 = nn.Conv2d(8 * base_channels, 8 * base_channels, kernel_size=3, dilation=8, padding=8)
        self.enc_batchnorm15 = nn.BatchNorm2d(8 * base_channels)

        '''
        Mask Decoder
        '''
        self.dec0_convTransp0 = nn.ConvTranspose2d(12 * base_channels, 4 * base_channels, kernel_size=3, stride=2,
                                                   output_padding=(0, 0))
        self.dec0_batchnorm0 = nn.BatchNorm2d(4 * base_channels)
        self.dec0_conv0 = nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=3, padding=1)
        self.dec0_batchnorm1 = nn.BatchNorm2d(4 * base_channels)
        self.dec0_conv1 = nn.Conv2d(4 * base_channels, 2 * base_channels, kernel_size=3, padding=1)
        self.dec0_batchnorm2 = nn.BatchNorm2d(2 * base_channels)
        self.dec0_drop1 = nn.Dropout(p=dropout_rate)
        self.dec0_convTransp1 = nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=3, stride=2,
                                                   output_padding=(1, 0))
        self.dec0_batchnorm3 = nn.BatchNorm2d(2 * base_channels)
        self.dec0_conv2 = nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1)
        self.dec0_batchnorm4 = nn.BatchNorm2d(2 * base_channels)
        self.dec0_conv3 = nn.Conv2d(2 * base_channels, base_channels, kernel_size=3, padding=1)
        self.dec0_batchnorm5 = nn.BatchNorm2d(base_channels)
        self.dec0_drop2 = nn.Dropout(p=dropout_rate)
        self.dec0_convTransp2 = nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=3, stride=2,
                                                   output_padding=(1, 1))
        self.dec0_batchnorm6 = nn.BatchNorm2d(base_channels)
        self.dec0_drop3 = nn.Dropout(p=dropout_rate)
        self.dec0_conv4 = nn.Conv2d(base_channels + 3, base_channels, kernel_size=3, padding=1)
        self.dec0_batchnorm7 = nn.BatchNorm2d(base_channels)
        self.dec0_conv5 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.dec0_batchnorm8 = nn.BatchNorm2d(base_channels)
        self.out_a = nn.Conv2d(base_channels, alphabet_len + 1, kernel_size=3, padding=1)

        '''
        Field Decoder
        '''
        self.dec1_convTransp0 = nn.ConvTranspose2d(12 * base_channels, 4 * base_channels, kernel_size=3, stride=2,
                                                   output_padding=(0, 0))
        self.dec1_batchnorm0 = nn.BatchNorm2d(4 * base_channels)
        self.dec1_conv0 = nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=3, padding=1)
        self.dec1_batchnorm1 = nn.BatchNorm2d(4 * base_channels)
        self.dec1_conv1 = nn.Conv2d(4 * base_channels, 2 * base_channels, kernel_size=3, padding=1)
        self.dec1_batchnorm2 = nn.BatchNorm2d(2 * base_channels)
        self.dec1_drop1 = nn.Dropout(p=dropout_rate)
        self.dec1_convTransp1 = nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=3, stride=2,
                                                   output_padding=(1, 0))
        self.dec1_batchnorm3 = nn.BatchNorm2d(2 * base_channels)
        self.dec1_conv2 = nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1)
        self.dec1_batchnorm4 = nn.BatchNorm2d(2 * base_channels)
        self.dec1_conv3 = nn.Conv2d(2 * base_channels, base_channels, kernel_size=3, padding=1)
        self.dec1_batchnorm5 = nn.BatchNorm2d(base_channels)
        self.dec1_drop2 = nn.Dropout(p=dropout_rate)
        self.dec1_convTransp2 = nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=3, stride=2,
                                                   output_padding=(1, 1))
        self.dec1_batchnorm6 = nn.BatchNorm2d(base_channels)
        self.dec1_drop3 = nn.Dropout(p=dropout_rate)
        self.dec1_conv4 = nn.Conv2d(base_channels + 3, base_channels, kernel_size=3, padding=1)
        self.dec1_batchnorm7 = nn.BatchNorm2d(base_channels)
        self.dec1_conv5 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.dec1_batchnorm8 = nn.BatchNorm2d(base_channels)
        self.out_b = nn.Conv2d(base_channels, fields_num + 1, kernel_size=3, padding=1)

    def forward(self, x):
        input_layer = x
        x = self.enc_batchnorm0(relu(self.enc_conv0(input_layer)))
        x = self.enc_batchnorm1(relu(self.enc_convStride0(x)))
        x = self.enc_batchnorm2(relu(self.enc_conv1(x)))
        x = self.enc_batchnorm3(relu(self.enc_conv2(x)))
        x_enc_conv2 = self.enc_drop1(x)
        x = self.enc_batchnorm4(relu(self.enc_convStride1(x_enc_conv2)))
        x = self.enc_batchnorm5(relu(self.enc_conv3(x)))
        x = self.enc_batchnorm6(relu(self.enc_conv4(x)))
        x_enc_conv4 = self.enc_drop2(x)
        x = self.enc_batchnorm7(relu(self.enc_convStride2(x_enc_conv4)))
        x = self.enc_batchnorm8(relu(self.enc_conv5(x)))
        x = self.enc_batchnorm9(relu(self.enc_conv6(x)))
        x_enc_conv6 = self.enc_drop3(x)
        x = self.enc_batchnorm10(relu(self.enc_conv7(x_enc_conv6)))
        x = self.enc_batchnorm11(relu(self.enc_conv8(x)))
        x = self.enc_batchnorm12(relu(self.enc_conv9(x)))
        x = self.enc_batchnorm13(relu(self.enc_conv10(x)))
        x = self.enc_batchnorm14(relu(self.enc_conv11(x)))
        x_enc_conv12 = self.enc_batchnorm15(relu(self.enc_conv12(x)))
        x_t = torch.cat((x_enc_conv12, x_enc_conv6), 1)

        x = self.dec0_batchnorm0(relu(self.dec0_convTransp0(x_t)))
        x = self.dec0_batchnorm1(relu(self.dec0_conv0(x)))
        x = self.dec0_batchnorm2(relu(self.dec0_conv1(x)))
        x_dec0_conv1 = self.dec0_drop1(x)
        x = torch.cat((x_dec0_conv1, x_enc_conv4), 1)
        x = self.dec0_batchnorm3(relu(self.dec0_convTransp1(x)))
        x = self.dec0_batchnorm4(relu(self.dec0_conv2(x)))
        x = self.dec0_batchnorm5(relu(self.dec0_conv3(x)))
        x_dec0_conv3 = self.dec0_drop2(x)
        x = torch.cat((x_dec0_conv3, x_enc_conv2), 1)
        x = self.dec0_batchnorm6(relu(self.dec0_convTransp2(x)))
        x_dec0_convt2 = self.dec0_drop3(x)
        x = torch.cat((x_dec0_convt2, input_layer), 1)
        x = self.dec0_batchnorm7(relu(self.dec0_conv4(x)))
        x = self.dec0_batchnorm8(relu(self.dec0_conv5(x)))
        x1 = self.out_a(x) if self.mode == 'train' else softmax(self.out_a(x), dim=1)

        x = self.dec1_batchnorm0(relu(self.dec1_convTransp0(x_t)))
        x = self.dec1_batchnorm1(relu(self.dec1_conv0(x)))
        x = self.dec1_batchnorm2(relu(self.dec1_conv1(x)))
        x_dec1_conv1 = self.dec1_drop1(x)
        x = torch.cat((x_dec1_conv1, x_enc_conv4), 1)
        x = self.dec1_batchnorm3(relu(self.dec1_convTransp1(x)))
        x = self.dec1_batchnorm4(relu(self.dec1_conv2(x)))
        x = self.dec1_batchnorm5(relu(self.dec1_conv3(x)))
        x_dec1_conv3 = self.dec1_drop2(x)
        x = torch.cat((x_dec1_conv3, x_enc_conv2), 1)
        x = self.dec1_batchnorm6(relu(self.dec1_convTransp2(x)))
        x_dec1_convt2 = self.dec1_drop3(x)
        x = torch.cat((x_dec1_convt2, input_layer), 1)
        x = self.dec1_batchnorm7(relu(self.dec1_conv4(x)))
        x = self.dec1_batchnorm8(relu(self.dec1_conv5(x)))
        x2 = self.out_b(x) if self.mode == 'train' else softmax(self.out_b(x), dim=1)

        return x1, x2
