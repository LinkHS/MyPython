# CNN

```python
import torch
import torch.nn.functional as F

from torch import nn
from torchsummary import summary
```

## Basic


## BatchNorm
BatchNorm的介绍具体参考[动手深度学习](https://zh.d2l.ai/chapter_convolutional-neural-networks/batch-norm.html)。对于FC层输出在每个通道上进行BS(batch size)级别的归一化；对于Conv层输出在每个通道上HxWxBS级别的归一化。

BatchNorm虽然好用，但是也有一些问题（详见[Devils in BatchNorm](https://www.techbeat.net/talks/MTU5NzEyNzg2MjU2MC0yOTktNzUzMjI=)），例如不一致性（inconsistency）问题：
1. 使用了[指数移动平均](https://zhuanlan.zhihu.com/p/68748778)会让学习的参数更加适应最新训练批次的样本
2. 训练集得到的batchnorm参数不一定适合测试集

第1个不一致性问题可以使用Precise BatchNorm，例如暂停更新网络参数，只更新BatchNorm层的参数。

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

### ResNeXt
- [ResNeXt的分类效果为什么比Resnet好?](https://www.zhihu.com/question/323424817)
- [薰风读论文：ResNeXt 深入解读与模型实现](https://zhuanlan.zhihu.com/p/78019001)

神经网络有两个重要的参数，深度和宽度（这里指的是通道数：the number of channels in a layer），经过ResNet等文章改进后，这两个参数对目前的网络的提升效果不是很明显了，大家开始对各种超参下手，这样很容易导致某一数据集碰巧适合一个“乱调”的超参，使网络丧失了泛化性。本文提出了一个新的参数cardinality，如下图右边网络中的“total 32 paths”，本质上就是对图中左边的3x3 conv做分组卷积：

![](http://static.zybuluo.com/AustinMxnet/4i1l4zpoi5h4f197on71hcs2/image.png)

![](http://static.zybuluo.com/AustinMxnet/g0ttoq6klzia3adzxtz81prs/image.png)


作者这么做的原因是受到Inception结构和AlexNet分组卷积启发，认为**split-transform-merge结构能达到大型密集网络的表达能力**，而计算量却要小很多。

> [ResNeXt的分类效果为什么比Resnet好?](https://www.zhihu.com/question/323424817/answer/1078704765) 一个答案认为多个cardinality和NLP中的multi-head attention是一个思路。每组是不同的subspace，就能学到更diverse的表示。

![](http://static.zybuluo.com/AustinMxnet/b5cahsk10t89licwxt5s8ek1/image.png)

接着为了简化计算，作者证明了上图中3个block是等价的，于是**输入和输出就简化成了一次1x1的卷积**，而不是原来cardinality（上图中为32）次。对比原来的ResNet结构（第一张图左），ResNeXt中的通道总数反而增多了（从64变成了128），这样其实也是增加了模型的能力，**但是重点是几乎没有增加任何的计算量和参数量！！！，原理类似Depthwise Conv**，计算量和参数量参见下图最后一行。

代码很简单，只需要对ResNet的代码微调：一是输入的通道数；二是将中间的conv3x3变成分组卷积，只要传入`groups=cardinality`参数即可：

```python
class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_channels, gw, cardinality=32, stride=1):
        """
        @gw, group width
        """
        super().__init__()
        
        out_channels = gw * self.expansion
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, gw, kernel_size=1, bias=False),
            nn.BatchNorm2d(gw),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(gw, gw, kernel_size=3, stride=stride,
                      padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(gw),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(gw, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

测试下图中conv4第一次之后的输入：

```python
x = torch.rand(2, 1024, 14, 14)
out = Block(1024, 512)(x)

print('out shape : {}'.format(out.shape))
```

测试下图中conv4第一次输入，即conv3的输出：

```python
x = torch.rand(2, 512, 28, 28)
out = Block(512, 512, stride=2)(x)

print('out shape : {}'.format(out.shape))
```

![](http://static.zybuluo.com/AustinMxnet/a8sd9f5m8g0vz762s07iqxco/image.png)

有了基础的`Block`就可以构建完整的`ResNeXt`了，例如上图对比了`ResNet-50`和`ResNeXt-50`。代码类似[ResNet](https://github.com/pytorch/vision/blob/3942b192e33dd79b6d9770149371bd58a483d47b/torchvision/models/resnet.py#L101)，提换为上面的`Block`并新增`cardinality`参数：

```python
class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0],
                                       in_channels=64)
        self.layer2 = self._make_layer(block, 256,  layers[1])
        self.layer3 = self._make_layer(block, 512,  layers[2])
        self.layer4 = self._make_layer(block, 1024, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

    def _make_layer(self, block, group_width, num_blocks, in_channels=None):
        layers = []

        if in_channels == None:
            in_channels = group_width
        layers.append(block(in_channels, group_width, self.cardinality, 2))

        inchannels = group_width * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(inchannels, group_width, self.cardinality, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def ResNeXt50_32x4d():
    return ResNeXt(Block, layers=[3, 4, 6, 3], cardinality=32)
```

```python
x = torch.rand(2, 3, 224, 224)
out = ResNeXt50_32x4d()(x)

print('out shape : {}'.format(out.shape))
#summary(ResNeXt50_32x4d(), (3, 224, 224))
```

### ResNeSt
- 作者视频讲解：[张航-ResNeSt：拆分注意力网络](https://www.bilibili.com/video/BV1PV411k7ch)

虽然论文中给的图比较了SENet和SKNet，但是ResNeSt主要结合了SKNet的分支间通道attention，和ResNeXt多分支的特点。在ResNeSt提出cardinality的基础上，在每个cardinality维度中又新增了radix参数，也就是分支中的分支：

![](http://static.zybuluo.com/AustinMxnet/c4tn4s3wfk9m1f70o871zero/image.png)

首先看下单独cardinality模块的处理，先经过1x1卷积缩小通道，然后经过3x3卷积提取特征，这和标准的ResNet没区别（除了是radix个分支）。下面就是ResNeSt中重点**Split Attention**：


![](http://static.zybuluo.com/AustinMxnet/z6bbl89fjw9k7wpgm068z6nz/image.png)

图中$r$个$h \times w \times c'$的输入经过Global Average Pooling和2个FC层后，得到$r$（radix）个`Dense c`，然后在$c$（channel）维度上做softmax，得到$r \times c$的权重图，权重图的第$i$列对应第$i$个channel的$r$个权重分布，下面的代码省略了GAP和FC：

```python
r = 2
c = 3
x = torch.rand(r, c)
xs = F.softmax(x, dim=0)
xs
```

结果权重`xs`中每一列有$r=2$个权重（每一列和为1）。

**这样虽然能求得cardinality个大分支的输出，但是要计算cardinality次**。在附录中，作者将$\text{radix} \times \text{cardinality}$等价变换为$\text{cardinality} \times \text{radix}$，这样只需计算一次就可以得到$\text{radix} \times \text{cardinality} \times c$的softmax权重图：

```python
r = 2
cardinality = 4
c = 3
x_gap = torch.rand(cardinality, r, c) # after global average pooling
xs = x_gap.transpose(0, 1)
xs = F.softmax(xs, dim=0)
xs
```

注意虽然输入的shape为$\text{cardinality} \times r \times c$，但是经过`transpose(0, 1)`后就对调了$\text{cardinality}$和$r$。如下图所示（k=cardinality），若将$(h,w,c)$分为$(k, r, h, w, c')$，并按照相同$r$的$k \times (h,w,c')$放在一起，只需要用一个group conv生成：`nn.Conv2d(c, c'*radix, groups=cardinality*radix)`。

> 图中一共有$\text{cardinality}=k$组，每组有$\text{radix}=r$个分支，每个分支通道数为$c'/k$。所以当`Conv2d`的参数`out_channel=c*radix`而`groups=k*radix`时，`Conv2d`每一`group`输出的通道数就是等于$c'/k$！！！

有了权重`xs`后，只需要将`xs`乘上相同shape的`x`再加上`x`就得到了Split-Attention的输出（注意这里省略了1x1缩小和放大通道的步骤）。这里有证明两者等价证明和论文作者测试代码（载入等价的网络权重，提供相同的输入，通过测试输出是否相同来验证模型是否等价），详见[ResNeSt 实现有误？](https://zhuanlan.zhihu.com/p/135220104)。

![](http://static.zybuluo.com/AustinMxnet/8hjis3n8sybi782qv7g2fks1/image.png)


注意图中使用的是**r-Softmax**，当`radix=1`时用`sigmoid`，公式如下：

$$a_{i}^{k}(c)=\left\{\begin{array}{ll}
\frac{\exp \left(\mathcal{G}_{i}^{c}\left(s^{k}\right)\right)}{\sum_{j=0}^{R} \exp \left(\mathcal{G}_{j}^{c}\left(s^{k}\right)\right)} & \text { if } R>1 \\
\frac{1}{1+\exp \left(-\mathcal{G}_{i}^{c}\left(s^{k}\right)\right)} & \text { if } R=1
\end{array}\right.$$

具体实现如下，注意`x.transpose`操作调换了radix和cardinality维度：

```python
class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1)
            x = x.transpose(1, 2)  # batch, radix, cardinality, -1
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
```

```python
radix = 2
cardinality = 4
c = 16
x = torch.rand(1, radix*cardinality, c)
out = rSoftMax(radix, cardinality)(x)

print('out shape : {}'.format(out.shape))
```

增加了Split-Attention的ResNet模块代码，注意代码中**用1x1的Conv代替了cardinality\*radix个并行FC的预算**：

```python
class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False,
                 radix=2, reduction_factor=4, **kwargs):
        super(SplAtConv2d, self).__init__()
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        self.conv = nn.Conv2d(in_channels, channels*radix, kernel_size, stride,
                              padding, dilation, groups=groups*radix, bias=bias, **kwargs)
        self.bn0 = nn.BatchNorm2d(channels*radix)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels *
                             radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()
```

因为代码中对FC输出用了BatchNorm，所以测试时候batch size > 1：

```python
x = torch.rand(2, 64, 56, 56)
out = SplAtConv2d(64, 32)(x)
print('out shape : {}'.format(out.shape))
```

#### Tricks
本篇论文训练时用到了很多tricks。首先ResNet-D：

![](http://static.zybuluo.com/AustinMxnet/scdvz53jz5elyfjsgdwugxco/image.png)
1. ResNet-B将`s=2`下采样（downsampling）从第一个（最下面）1x1移到了3x3卷积中，避免信息丢失（因为`s=2`的1x1会直接跳过像素）。
2. ResNet-C中将ResNet第一层的7x7卷积用3个3x3卷积代替。
3. ResNet-D中解决了ResNet-B中旁路上1x1在`s=2`时信息丢失的问题，先用`AvgPool`进行下采样。

其他还有Label Smoothing，Mixup Training，Auto Augment等。作者实现的[代码](https://github.com/zhanghang1989/ResNeSt)提供了MXNet和PyTorch版本。


## CV Attention

注意力机制可以分为：
- 通道注意力机制：对通道生成掩码mask，进行打分，代表是SENet, Channel Attention Module
- 空间注意力机制：对空间进行掩码的生成，进行打分，代表是Spatial Attention Module
- 混合域注意力机制：同时对通道注意力和空间注意力进行评价打分，代表的有BAM, CBAM

文章：
- 专栏：[机器视觉Attention机制的研究](https://zhuanlan.zhihu.com/cvattention)
  - [Attention算法调研——视觉应用概况](https://zhuanlan.zhihu.com/p/52925608)
  - [Attention算法调研(一)——机器翻译中的Attention](https://zhuanlan.zhihu.com/p/52786464)
  - [Attention算法调研(二)——机器翻译中的Self Attention](https://zhuanlan.zhihu.com/p/52861193)
  - [Attention算法调研(三)——视觉应用中的Hard Attention](https://zhuanlan.zhihu.com/p/52958865)
  - [Attention算法调研(四)——视觉应用中的Soft Attention](https://zhuanlan.zhihu.com/p/53026371)
  - [Attention算法调研(五)——视觉应用中的Self Attention](https://zhuanlan.zhihu.com/p/53155423)


### CBAM
为了强调空间和通道这两个维度上的有意义特征，作者依次应用通道和空间注意力模块，分别在通道和空间维度上学习关注什么、在哪里关注。此外，通过了解要强调或抑制的信息也有助于网络内的信息流动。

![](http://static.zybuluo.com/AustinMxnet/z8o0mdlygap9p5mkkgpbkcae/image.png)

主要网络架构也很简单，上图展示了和ResBlock的结合，对Feature Maps依次通过Channel attention和Spatial attention两个module。原文：   Given an intermediate feature map $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$ as input, CBAM sequentially infers a 1D channel attention map $\mathbf{M}_{\mathbf{c}} \in \mathbb{R}^{C \times 1 \times 1}$ and a 2D spatial attention map $\mathbf{M}_{\mathbf{s}} \in \mathbb{R}^{1 \times H \times W}$:

$$\begin{aligned}
\mathbf{F}^{\prime} &=\mathbf{M}_{\mathbf{c}}(\mathbf{F}) \otimes \mathbf{F} \\
\mathbf{F}^{\prime \prime} &=\mathbf{M}_{\mathbf{s}}\left(\mathbf{F}^{\prime}\right) \otimes \mathbf{F}^{\prime}
\end{aligned}$$

其中$\otimes$表示element-wise multiplication。两个模块详细的结构如下图所示：

![](http://static.zybuluo.com/AustinMxnet/gkkkb0jf3q352fr8mdwgjgda/image.png)

至于为什么Channel在前，Spatial在后，是因为实验结果更好。下面分别看下两个模块。


**Channel attention module**的公式如下：

$$\begin{aligned}
\mathbf{M}_{\mathbf{c}}(\mathbf{F}) &=\sigma(\textit{MLP}(\textit{AvgPool}(\mathbf{F}))+\textit{MLP}(\textit{MaxPool}(\mathbf{F}))) \\
&=\sigma\left(\mathbf{W}_{1}\left(\mathbf{W}_{\mathbf{0}}\left(\mathbf{F}_{\text{avg}}^{\mathbf{c}}\right)\right)+\mathbf{W}_{\mathbf{1}}\left(\mathbf{W}_{\mathbf{0}}\left(\mathbf{F}_{\text{max}}^{\mathbf{c}}\right)\right)\right)
\end{aligned}$$

其中：
- $\mathbf{F}_{\text{avg}}^{\mathbf{c}}$和$\mathbf{F}_{\text{max}}^{\mathbf{c}}$分别表示在空间$HW$维度上average-pooled和max-pooled features，大小为通道数$c$。作者认为结合max-pooling和average-pooling能提供更多的信息。

- $\mathbf{W}_{\mathbf{0}}, \mathbf{W}_{\mathbf{1}} \in \mathbb{R}^{C / r \times C}$，$r$是reduction ratio，减少计算量的。**注意这两个weight参数是被$\textit{MLP}$共享的**，所以下面的代码使用了一个`sharedMLP`。

- $\sigma$表示sigmod函数。

- $M_c\in \mathbb{R}^{C\times 1 \times 1}$就是channel attention map。

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
```

$\mathbf{F}^{\prime} =\mathbf{M}_{\mathbf{c}}(\mathbf{F}) \otimes \mathbf{F}$

```python
x = torch.rand(1, 128, 14, 14)
# channel attention map
Mc = ChannelAttention(128)(x)

print('Mc shape : {}'.format(Mc.shape))

# features with channel attention
xc = x * Mc 
print('xc shape : {}'.format(xc.shape))
```

**Spatial attention module**的公式如下：

$$\begin{aligned}
\mathbf{M}_{\mathbf{s}}(\mathbf{F}) &=\sigma\left(f^{7 \times 7}([\textit{AvgPoll}(\mathbf{F}) ; \textit{MaxPool}(\mathbf{F})])\right) \\
&=\sigma\left(f^{7 \times 7}\left(\left[\mathbf{F}_{\text{avg}}^{\mathbf{s}} ; \mathbf{F}_{\text{max}}^{\mathbf{s}}\right]\right)\right)
\end{aligned}$$

其中：
- $\mathbf{F}_{\text{avg}}^{\mathbf{s}}, \mathbf{F}_{\text{max}}^{\mathbf{s}} \in \mathbb{R}^{1 \times H \times W}$，分别表示在通道$c$维度上average-pooled和max-pooled features，然后把这两个2D features被concatenate在一起。

- $f^{7 \times 7}$表示filter size为$7\times7$的卷积。

- $\sigma$表示sigmod函数。

- $M_s\in \mathbb{R}^{H\times W}$就是spatial attention map。

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
```

$\mathbf{F}^{\prime \prime} =\mathbf{M}_{\mathbf{s}}\left(\mathbf{F}^{\prime}\right) \otimes \mathbf{F}^{\prime}$

```python
x = torch.rand(1, 128, 14, 14)
# channel attention map
Ms = SpatialAttention()(x)

print('Ms shape : {}'.format(Ms.shape))

# features with channel attention
xs = x * Ms 
print('xc shape : {}'.format(xs.shape))
```

把两者结合起来就得到了`CBAM`模块：

```python
class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
```

```python
x = torch.rand(1, 128, 14, 14)

out = CBAM(128)(x)
print('out shape : {}'.format(out.shape))
```

原文中把`CBAM`和`ResNet`集成时是这么说的：“We apply CBAM on the convolution outputs in each block”，可能是加在每个`ResBlock`输出上（未验证）。在这篇[文章](https://zhuanlan.zhihu.com/p/99261200)中，**作者为了能够用预训练的参数**，把`CBAM`加在`ResBlock`之前和之后，见`ca/sa`和`ca1/sa1`：

```script magic_args="true"
class ResNet(nn.Module):
    x = self.conv1(x)

    x = self.ca(x) * x
    x = self.sa(x) * x

    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.ca1(x) * x
    x = self.sa1(x) * x

    x = self.avgpool(x)
    x = x.reshape(x.size(0), -1)
    x = self.fc(x)
```
