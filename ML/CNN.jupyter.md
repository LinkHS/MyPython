# CNN

```python
import torch
import torch.nn.functional as F

from torch import nn
```

## Basic

```python
# target output size of 5x7
m = nn.AdaptiveAvgPool2d((5,7))
input = torch.randn(1, 64, 8, 9)
output = m(input)
output.shape
```

```python
# target output size of 7x7 (square)
m = nn.AdaptiveAvgPool2d(7)
input = torch.randn(1, 64, 10, 9)
output = m(input)
output.shape
```

```python
# target output size of 10x7
m = nn.AdaptiveMaxPool2d((None, 7))
input = torch.randn(1, 64, 10, 9)
output = m(input)
```

## Paper

<!-- #region -->
### SENet
- 参考：[[论文笔记]-SENet和SKNet(附代码)](https://zhuanlan.zhihu.com/p/76033612)
- Pytorch代码：https://github.com/moskomule/senet.pytorch


![](http://static.zybuluo.com/AustinMxnet/6sfp5yrczet76xl5qitd9cv4/image.png)

一共有三步，分别是Squeeze，Excitation和Fscale。代码中的`r`是一个缩放参数，默认16，文中说引入这个参数是为了减少`channel`个数从而降低计算量。
<!-- #endregion -->

```python
class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y
```

```python
x = torch.rand(1, 128, 28, 28)
out = SEBlock(128)(x)

out.shape
```

可以看到`SEBlock`并没有改变`x.shape`，只是给每个通道根据计算的权重重新赋值。

`SEBlock`很容易集成到现有的模块中，例如对ResNet来说只需要对`Residual`加一步`SEBlock`即可：

![](http://static.zybuluo.com/AustinMxnet/jk0x9zla6pe93few53nz7vow/image.png)

集成的SE-ResNet可以参考[github](https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py#L11)，部分代码如下所示。注意这个仓库中命名的是`SELayer`而不是`SEBlock`。

```script magic_args="true"
from torchvision.models import ResNet


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model
```

### SKNet
- 参考：[[论文笔记]-SENet和SKNet(附代码)](https://zhuanlan.zhihu.com/p/76033612)

SKNet的核心就是Selective Kernel Convolution，如下图所示：

![](http://static.zybuluo.com/AustinMxnet/j8m6bvndtwx4zgq6m9sy73tm/image.png)

Selective Kernel Convolution主要有三步：
1. **Split**：用了两组不同大小的Kernel对$X$分别做卷积运算，得到两个相同shape的输出$\widetilde{\mathbf{U}}$和$\widehat{\mathbf{U}}$。

2. **Fuse**：将$\widetilde{\mathbf{U}}$和$\widehat{\mathbf{U}}$相加得到$\mathbf{U}$，然后类似SENet对$\mathbf{U}$计算通道之间的权重$a,b$。但是不同于SENet计算一组通道之间的权重，即一次softmax运算；而SKNet计算每个通道在两个分支上的权重，共channel次softmax运算，也就是$a,b$每个相同位置上的值加起来为1。

3. **Select**：根据计算$a,b$对$\widetilde{\mathbf{U}}$和$\widehat{\mathbf{U}}$做加权求和，得到$\mathbf{V}$。


下面的代码实现了Selective Kernel Convolution。注意几点：
1. `M`对应分支数
2. `reduction`对应SENet中的r，是一个缩放参数，目的减少channel个数从而降低计算量
3. 论文中说可以用dilated的`conv3x3`代替`conv5x5`，对应代码`dilation=1+i`
4. `forward`中的`feats`对应$\mathbf{U}$，shape和`x`相同

```python
class SKConv(nn.Module):
    def __init__(self, channels, stride=1, M=2, reduction=4):
        super().__init__()
        self.conv1 = nn.ModuleList([])
        for i in range(M):
            self.conv1.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride,
                          padding=1+i, dilation=1+i, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ))
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels * M, 1)
        )

    def forward(self, x):
        splited = [conv(x) for conv in self.conv1]
        feats = sum(splited)  
        att = self.att(feats) # shape = (batch num, (channels*M), 1, 1)
        # shape = (batch num, M, channels)
        att = att.view(x.size(0), len(self.conv1), x.size(1)) 
        att = F.softmax(att, dim=1)
        att = att.view(x.size(0), -1, 1, 1)
        att = torch.split(att, x.size(1), dim=1)

        return sum([a * s for a, s in zip(att, splited)])
```

测试一下$14\times 14$块（见下面第二个图）中的`SKConv`。注意如果不是块中第一次卷积运算（即输入不是$28\times 28$的输出），是不需要改变feature maps的大小，使用默认`stride=1`。

```python
x = torch.rand(2, 1024, 14, 14)
skconv = SKConv(1024)
out = skconv(x)
print('out shape : {}'.format(out.shape))

# 测试backward()和loss
criterion = nn.L1Loss()
loss = criterion(out, x)
loss.backward()
print('loss value : {}'.format(loss))
```

有了`SKConv`，我们就可以构建基于SKNet的ResNet了，例如SKNet-50，只需要替换ResNet模块中的$3\times 3$卷积。ResNet模块如下图所示，左边是普通的ResNet模块，右边是bottlenecck结构的ResNet模块：

![](http://static.zybuluo.com/AustinMxnet/geba8sxfed73kwumnyrdwiy7/image.png)

完整的SKNet结构如下图所示。在max pool之后，每个block会重复一定的次数（3，4，6，3），这些block第一次时候都需要将feature maps减半，此时输入的通道数是输出的一半。例如$56\times 56$中最后一次输出的通道数为256，即$28\times 28$的输入，而$28\times 28$的输出为512。

![](http://static.zybuluo.com/AustinMxnet/f7cgwxi5o8xmvyry1oysr56h/image.png)

代码中使用了`in_channels == out_channels`来判断是否需要对feature maps的大小减半：

```python
from torch.nn.quantized import FloatFunctional


class SKUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels == out_channels:
            mid_channels = in_channels // 2
            stride = 1
            self.shortcut = nn.Sequential()
        else:
            mid_channels = in_channels
            stride = 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            SKConv(mid_channels, stride),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = FloatFunctional()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        x = self.shortcut(x)
        return self.relu.add_relu(x, out)
```

测试不需要减半的`SKUnit`：

```python
x = torch.rand(8, 64, 32, 32)
out = SKUnit(64, 64)(x)

out.shape
```

测试需要减半的SKUnit：

```python
out = SKUnit(64, 128)(x)

out.shape
```
