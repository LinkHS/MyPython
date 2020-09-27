```python
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd import Variable
```

# Transformer

![](http://static.zybuluo.com/AustinMxnet/rvmk73cqyxew8yqlvqxvz2pt/image.png)

Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成。更准确地讲，Transformer由且仅由self-Attenion和Feed Forward Neural Network组成。一个基于Transformer的可训练的神经网络可以通过堆叠Transformer的形式进行搭建，作者的实验是通过搭建编码器和解码器各6层，总共12层的Encoder-Decoder，并在机器翻译中取得了BLEU值得新高。

作者采用Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是顺序的，也就是说RNN相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：
1. 时间片$t$的计算依赖$t-1$时刻的计算结果，这样限制了模型的并行能力；
2. 顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象,LSTM依旧无能为力。

Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离是缩小为一个常量；其次它不是类似RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。

参考：
- [详解Transformer](https://zhuanlan.zhihu.com/p/48508221)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)


## Word Embeddeing
Word embedding的权重矩阵通常有两种选择：
1. 使用Pre-trained的Embeddings并固化，这种情况下实际就是一个 Lookup Table。
2. 对其进行随机初始化(当然也可以选择Pre-trained 的结果)，但设为Trainable。这样在training过程中不断地对Embeddings进行改进。

注意结果乘以`np.math.sqrt(d)`，在原文中是这样解释的："In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation."

```python
class Embedder(nn.Module):
    def __init__(self, vocab, d_model):
        """
        @vocab, 词汇表的数量
        @d_model, embedding的维度
        """
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
 
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

下面构造一个维度为1000的，词汇表数量为512的`embeddings`：

```python
# batch_size=2
x = torch.LongTensor([[1, 2, 4, 5],
                      [4, 3, 2, 9]])

out = Embedder(512, 1000)(x)

out.shape
```

## Positional Encoding
和RNN、LSTM按顺序输入不同，Transformer需要给输入加上位置信息，可以参考下面几篇文章：
- [如何理解Transformer论文中的positional encoding，和三角函数有什么关系？](https://www.zhihu.com/question/347678607/answer/864217252)

- [如何优雅地编码文本中的位置信息？三种positional encoding方法简述](https://zhuanlan.zhihu.com/p/121126531)

总结一下：
1. 位置编码需要有值域范围，避免后面的位置编码非常大
2. 位置编码步长要一致，避免在不同长度的文本中不一致
3. 不同维度上应该用不同的函数操纵位置编码（例如上例的embeddings的维度为1000）

Transformer中选择了正弦和余弦函数：

$$\begin{array}{l}
P E_{(p o s, 2 i)}=\sin \left(p o s / 10000^{2 i / d_{\text {model}}}\right) \\
P E_{(p o s, 2 i+1)}=\cos \left(p o s / 10000^{2 i / d_{\text {model}}}\right)
\end{array}$$

这样设计的好处是位置$pos+k$的positional encoding可以被位置$pos$**线性表示**，反应其相对位置关系，因为：

$$\begin{array}{l}
\sin (\alpha+\beta)=\sin \alpha \cdot \cos \beta+\cos \alpha \cdot \sin \beta \\
\cos (\alpha+\beta)=\cos \alpha \cdot \cos \beta-\sin \alpha \cdot \sin \beta
\end{array}$$

可以得到，其中系数$sin(w_ik)$和$cos(w_ik)为常数$：

$$\begin{aligned}
P E_{(p o s+k, 2 i)} &=\cos \left(w_{i} k\right) P E_{(p o s, 2 i)}+\sin \left(w_{i} k\right) P E_{(p o s, 2 i+1)} \\
P E_{(p o s+k, 2 i+1)} &=\cos \left(w_{i} k\right) P E_{(p o s, 2 i+1)}-\sin \left(w_{i} k\right) P E_{(p o s, 2 i)}
\end{aligned}$$

这种方式也有缺点，例如虽然能够反映相对位置的距离关系，但是无法区分方向：

$$P E_{p o s+k} P E_{p o s}=P E_{p o s-k} P E_{p o s}$$

后来也有其他方式，例如Bert中好像直接用网络去学习positional embedding。

```python
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)    # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)    # 奇数列
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
```

下面我们看下一个20维数据长度为100的`PositionalEncoding`情况，为了显示效果只展示4, 5, 6, 7四维：

```python
pe = PositionalEncoder(20, 0)
y = pe.forward(torch.zeros(1, 100, 20))

plt.figure(figsize=(15, 5))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
plt.show()
```

## Encoder
![](http://static.zybuluo.com/AustinMxnet/lc026w45sbgbb35jrxarl791/image.png)

### Self-Attention
Self-Attention的核心内容是为输入向量的每个单词学习一个权重，例如我们需要判断`The animal didn't cross the street because it was too tired`这句话中`it`代指的内容，可以让网络学习`it`和每个单词之间的相关性：

![](http://static.zybuluo.com/AustinMxnet/fb3qndx43fh7iy7wthczw2gg/image.png)

Self-attention公式如下：
$$\text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$$

实际运算时将$x_1, x_2$叠加起来组成矩阵形式（这里第一行是$x_1$，第二行是$x_2$）：
![](http://static.zybuluo.com/AustinMxnet/d73kejzqzg1x8zo1sy6twyel/image.png)

为了梯度的稳定，Transformer使用了score归一化，即除以$\sqrt{d_k}$，其中$Q、K、V$分别是Query、Key、Value，由$X$分别进行三次矩阵乘法得到的向量：
![](http://static.zybuluo.com/AustinMxnet/pjs6n09wzp4054dykqqcvdz9/image.png)

有了$Q, K, V$后的Softmax计算分解图，这里假设$q, k, v$的长度为64：
![](http://static.zybuluo.com/AustinMxnet/wiz4y8234dcit0u00zhjvi6a/image.png)

下面是Attention的代码，并且将Softmax的结果`score`打印出来供分析：

```python
def attention(q, k, v, d_k):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)
    print('scores:', scores, sep='\n')
    return output
```

```python
x = torch.randn((2, 4))

q = nn.Linear(4, 3)(x)
k = nn.Linear(4, 3)(x)
v = nn.Linear(4, 3)(x)

attention(q, k, v, 3)
```

从`scores`的结果可以看到，每一行的sum都为1，即上图中的$z_1, z_2$都是由$v_1, v_2$加权求和得到。

### Multi-Head Attention
Multi-head Attention可以增加模型关注不同位置的能力，例如"it"，网络不仅要知道它是指代"the animal"，也要知道它的属性"tired"：
![](http://static.zybuluo.com/AustinMxnet/8hp2oqamcje0ajtt3hpfpx7l/image.png)

实现起来很简单，可以让$X$多经过几个Self-Attention即可：
![](http://static.zybuluo.com/AustinMxnet/y88wq32mcqt9jxwsjev6i4vs/image.png)

代码的实现和图中稍微有点区别，将$X$切分为$n_{heads}$份（$n_{heads}$就是attention的次数），假设emmbedding后的$X$维度为$d_{model}=512$，那么每个attention的输入数据的维度就是$512/n_{heads}$，整个运算过程都是矩阵形式，而非分别运行$n_{heads}$次。代码中通过通过`view()`函数将`k, q, v`从$bs\times n_{words} \times d_{model}$划分成$bs\times n_{words} \times n_{heads} \times \frac{d_{model}}{n_{heads}}$，然后通过`transpose()`转为$bs\times n_{heads} \times n_{words}  \times \frac{d_{model}}{n_{heads}}$，再送入`attention()`同时做了$n_{heads}$次attention。

```python
def attention(q, k, v, d_k, mask=None, dropout=None, verbose=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    
    if verbose:
        print('scores:', scores, sep='\n')
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, verbose=False):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout, verbose=verbose)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output
```

可以看到`attention()`的结果为$bs\times n_{heads} \times n_{words}  \times \frac{d_{model}}{n_{heads}}$，通过`transpose()`和`view()`合并回了$bs\times n_{words}  \times d_{model}$，这里$d_{model}$是为了配合最后一步`nn.Linear()`（`out()`）的输入要求，也可以如下图所示在最后一步`nn.Linear()`完成数据维度转换，图中最后输出的$d_{model}=4$。

![](http://static.zybuluo.com/AustinMxnet/bnzhgvy62x2v2ldtdigldp42/image.png)

测试时需要将`dropout=0`，否则每行softmax的概率和不为1。

```python
x = torch.randn((1, 2, 512))
MultiHeadAttention(4, 512, dropout=0.0)(x, x, x, verbose=True).size()
```

### Normalization
经过Self-Attention后，先通过shutcut又称residual求和，然后用LayerNorm进行归一化：

![](http://static.zybuluo.com/AustinMxnet/y02t43jybl2tia980lkfht10/image.png)

LayerNorm是在一个样本上进行归一化，一个word embedding的维度为$1\times d_{model}$，一张图片则为$1\times H \times W \times C$：

![](http://static.zybuluo.com/AustinMxnet/ocffurtndo7kbkqdgbmkcj08/image.png)

```python
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / \
            (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
```

由于单个word维度为$1\times d_{model}$，只需要对最后一位$d_{model}$归一化即可：

```python
x = torch.ones((2, 3, 10))
x.mean(dim=-1, keepdim=True)
```

### Feed Forward
Feed Forward有两个线性层组成，很简单：

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
```

### Encoder Layer
上面实现了一个Encoder Layer的各个部分，只要组合起来即可。这里提一下`mask`，因为每个句子长短不一，所以通过padding将句子补齐至长度一致，这样`mask`让padding部分不参与attention操作。

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
```

### Encoder

```python
import copy

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        _enc = EncoderLayer(d_model, heads, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(_enc) for i in range(N)])
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
```

```python
x = torch.ones((2, 3), dtype=torch.int64)
mask = torch.ones((2, 3))

Encoder(vocab_size=1000, d_model=512, N=6, heads=8, dropout=0)(x, None).size()
```

## Decoder
### Target Mask
因为decoding是一个顺序操作的过程，也就是解码第$k$个特征向量时，此时输入网络的只有$k$之前的decoding结果。以翻译结果"I am a student"为例，在**预测**时，encoder每一步的输入为：
  1. encoder的输出和开始符号`</s>`，decoder正确情况下输出"I"
  2. encoder的输出和"</s>I"，decoder正确情况下输出"am"
  3. encoder的输出和"</s>I am"，decoder正确情况下输出"a"
  4. encoder的输出和"</s>I am a"，decoder正确情况下输出"student"
  5. encoder的输出和"</s>I am a student"，decoder正确情况下输出结束符号"</eos>"

但是在**训练时，由于我们已经知道了正确的结果，上面的5步是可以并行进行的**，例如第二步无需理会第一步的输出是什么，直接讲正确结果"I am a student"的"</s>I"（使用mask实现）作为输入，这样也可以**避免第一步如果输出错误信息，导致后面的结果都是错误的**。关于mask的分析可以看[Transformer 源码中 Mask 机制的实现](https://www.cnblogs.com/wevolf/p/12484972.html)。

如果包含开始符号`</s>`，那么"I am a student"的`target mask`应该如下：

```python
def subsequentmask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return subsequent_mask == 0

print(subsequentmask(5))
```

### Decoder Layer
![](http://static.zybuluo.com/AustinMxnet/klwfy4fo45lsow2ilyffsu2o/image.png)


Decoder Layer和Encoder Layer基本一致，但是有几点区别：
1. Decoder有两个MultiHead Attention，第一个Self-Attention的输入是上一个Encoder Layer的输出（DECODER#1除外，见下文），但是第二个称为Encoder-Decoder Attention，因为$K, V$来自最后一个Encoder Layer的输出。
2. 第一个MultiHead Attention的`target_mask`用于忽略还未预测的数据。

```python
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
                                           src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
```

### Decoder
![](http://static.zybuluo.com/AustinMxnet/jlkqskkw0hieniw5i8zqawrn/image.png)

第一个Decoder的输入是已经预测的结果（最后一层Softmax的输出），并且也要经过Embedding和Position Encoding，例如图中第一次预测的结果是"I"，作为第二次预测的输入。

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        _dec = DecoderLayer(d_model, heads, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(_dec) for i in range(N)])
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
```

## Transformer
![](http://static.zybuluo.com/AustinMxnet/eqefi7nvowb7li20hrlm1eca/image.png)

有了Encoder和Decoder，最后再加上一层`Linear`和`Softmax`就是完整的Transformer网络。但是**为了计算梯度方便，将`Softmax`和`Cross-entropy`合并了**，所以代码中没有Softmax：

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
```

## Beam search
之前说了预测是一个顺序过程，我们可以每次都选概率最高的word，这种方式叫作greedy decoding。另外也可以每次保留概率最高的2个words，这种方式称为beam search，实现方式可以参考[github](https://github.com/SamLynnEvans/Transformer/blob/master/Beam.py)。
