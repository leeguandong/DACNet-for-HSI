from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None

        # TODO 初始化
        # nn.init.kaiming_uniform_(self.weight, )

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups, self.kernel_size,
                                                                    self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=False, groups=groups))


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck, gate_factor, squeeze_rate, group_3x3, heads):
        super(_DenseLayer, self).__init__()
        # 1x1 conv: i --> bottleneck * k
        # self.conv_1 = DynamicMultiHeadConv(in_channels, bottleneck * growth_rate, kernel_size=1, heads=heads,
        #                                    squeeze_rate=squeeze_rate, gate_factor=gate_factor)
        self.conv_1 = Dynamic_conv3d( in_channels, bottleneck * growth_rate, kernel_size=1, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34)

        # 3x3 conv: bottleneck * k --> k
        self.conv_2 = Conv(bottleneck * growth_rate, growth_rate, kernel_size=3, padding=1, groups=group_3x3)
        # self.conv_2 = Dynamic_conv3d(bottleneck * growth_rate, growth_rate, kernel_size=3, padding=1, groups=group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x_)
        x = self.conv_2(x)
        x = torch.cat([x_, x], 1)
        return x


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck, gate_factor, squeeze_rate, group_3x3, heads):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck, gate_factor, squeeze_rate,
                                group_3x3, heads)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class DydenseNet(nn.Module):
    def __init__(self, band, num_classes):
        super(DydenseNet, self).__init__()
        self.name = 'dydensenet'
        # self.stages = [14, 14, 14]
        # self.growth = [8, 16, 32]
        # self.stages = [14, 14, 14]
        # self.growth = [8, 16, 32]
        self.stages = [4, 6, 8]
        self.growth = [8, 16, 32]
        # self.stages = [10, 10, 10]
        # self.growth = [8, 16, 32]
        self.progress = 0.0
        self.init_stride = 2
        self.pool_size = 7
        self.bottleneck = 4
        self.gate_factor = 0.25
        self.squeeze_rate = 16
        self.group_3x3 = 4
        self.heads = 4

        self.features = nn.Sequential()
        # Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        # Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv3d(1, self.num_features, kernel_size=3, stride=self.init_stride,
                                                        padding=1, bias=False))
        for i in range(len(self.stages)):
            # Dense-block i
            self.add_block(i)

        # Linear layer
        self.bn_last = nn.BatchNorm3d(self.num_features)
        self.relu_last = nn.ReLU(inplace=True)
        # self.pool_last = nn.AvgPool3d(self.pool_size)
        self.pool_last = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(self.num_features, num_classes)
        self.classifier.bias.data.zero_()

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return

    def add_block(self, i):
        # Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            bottleneck=self.bottleneck,
            gate_factor=self.gate_factor,
            squeeze_rate=self.squeeze_rate,
            group_3x3=self.group_3x3,
            heads=self.heads
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features)
            self.features.add_module('transition_%d' % (i + 1), trans)

    def forward(self, x, progress=None, threshold=None):
        # if progress:
        #     DynamicMultiHeadConv.global_progress = progress
        features = self.features(x)
        features = self.bn_last(features)
        features = self.relu_last(features)
        features = self.pool_last(features)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    net = DydenseNet(200, 12)

    from torchsummary import summary

    summary(net, input_size=[(1, 200, 7, 7)], batch_size=1)

    from thop import profile

    input = torch.randn(1, 1, 200, 7, 7)
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))
