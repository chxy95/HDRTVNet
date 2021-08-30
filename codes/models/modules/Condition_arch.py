import torch.nn as nn


def color_block(in_filters, out_filters, normalization=False):
    conv = nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
    pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
    act = nn.LeakyReLU(0.2)
    layers = [conv, pooling, act]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class Color_Condition(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class ConditionNet(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_Condition(out_c=cond_c)

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        condition = x[1]
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out