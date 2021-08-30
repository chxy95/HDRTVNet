import torch
import torch.nn as nn
  
### initalize the module
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=1, norm=True, act=True):
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding)
    n = nn.BatchNorm2d(out_nc, affine=True) if norm==True else None
    a = nn.ReLU(inplace=True) if act==True else None
    return nn.Sequential(c, n, a)

def up_block(in_nc, out_nc, sf=2, kernel_size=3, stride=1):
    c = nn.Conv2d(in_nc, out_nc*sf**2, kernel_size=kernel_size, stride=stride, padding=1)
    s = nn.PixelShuffle(sf)
    a = nn.ReLU(inplace=True)
    return nn.Sequential(c, s, a)

class Hallucination_Generator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, sf=2):
        super(Hallucination_Generator, self).__init__()

        self.maxpool = nn.MaxPool2d(sf)

        self.conv1 = conv_block(in_nc, nf)
        self.conv2 = conv_block(nf, 2*nf)

        self.conv3_1 = conv_block(2*nf, 4*nf)
        self.conv3_2 = conv_block(4*nf, 4*nf)

        self.conv4_1 = conv_block(4*nf, 8*nf)
        self.conv4_2 = conv_block(8*nf, 8*nf)

        self.conv5_1 = conv_block(8*nf, 8*nf)
        self.conv5_2 = conv_block(8*nf, 8*nf)

        self.conv_code1 = conv_block(8*nf, 8*nf)
        self.conv_code2 = conv_block(8*nf, 8*nf)

        self.Up_conv1 = up_block(8*nf, 8*nf, sf=2)
        self.conv6 = nn.Conv2d(16*nf, 8*nf, 1, 1)

        self.Up_conv2 = up_block(8*nf, 8*nf, sf=2)
        self.conv7 = nn.Conv2d(16*nf, 4*nf, 1, 1)

        self.Up_conv3 = up_block(4*nf, 4*nf, sf=2)
        self.conv8 = nn.Conv2d(8*nf, 2*nf, 1, 1)

        self.Up_conv4 = up_block(2*nf, 2*nf, sf=2)
        self.conv9 = nn.Conv2d(4*nf, nf, 1, 1)

        self.Up_conv5 = up_block(nf, nf, sf=2)
        self.conv10 = nn.Conv2d(2*nf, out_nc, 1, 1)

        self.conv_last = nn.Conv2d(2*out_nc, out_nc, 1, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        # img.shape: 3, 160, 160
        img = x[0]
        mask = x[1]

        conv1_out = self.conv1(img) # 64, 160, 160

        conv2_out = self.maxpool(conv1_out) # 64, 80, 80
        conv2_out = self.conv2(conv2_out) # 128, 80, 80

        conv3_out = self.maxpool(self.conv3_1(conv2_out)) # 256, 40, 40
        conv3_out = self.conv3_2(conv3_out) # 256, 40, 40

        conv4_out = self.maxpool(self.conv4_1(conv3_out)) # 512, 20, 20
        conv4_out = self.conv4_2(conv4_out) # 512, 20, 20

        conv5_out = self.maxpool(self.conv5_1(conv4_out)) # 512, 10, 10
        conv5_out = self.conv5_2(conv5_out) # 512, 10, 10

        conv_code = self.maxpool(self.conv_code1(conv5_out)) # 512, 5, 5
        conv_code = self.conv_code2(conv_code) # 512, 5, 5

        # upsampling
        conv6_out = torch.cat((self.Up_conv1(conv_code), conv5_out), 1) # 1024, 10, 10
        conv6_out = self.conv6(conv6_out) # 512, 10, 10

        conv7_out = torch.cat((self.Up_conv2(conv6_out), conv4_out), 1) # 1024, 20, 20
        conv7_out = self.conv7(conv7_out) # 256, 20, 20

        conv8_out = torch.cat((self.Up_conv3(conv7_out), conv3_out), 1) # 512, 40, 40
        conv8_out = self.conv8(conv8_out) # 128, 40, 40

        conv9_out = torch.cat((self.Up_conv4(conv8_out), conv2_out), 1) # 256, 80, 80
        conv9_out = self.conv9(conv9_out) # 64, 80, 80

        conv10_out = torch.cat((self.Up_conv5(conv9_out), conv1_out), 1) # 128, 160, 160
        conv10_out = self.conv10(conv10_out) # 3, 160, 160

        out = torch.cat((conv10_out, img), 1) # 6, 160, 160
        out = self.conv_last(out)
        
        out = mask * out + img
        return out


