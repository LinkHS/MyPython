# Tensor 基本操作

References:
- [torch.Tensor](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch - Basic operations](https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/)

```python
import torch

torch.manual_seed(0)
```

```python
print(torch.rand(2))
```

## Create a Tensor

```python
torch.rand(2, 3) # Initialize with random number (uniform distribution)
```
```python
# With normal distribution (SD=1, mean=0)
torch.randn(2, 3) 
```
```python
# 可以用括号指定维度
torch.rand((1, 3, 2, 4))
```
### `tensor.dtype`

[dtype](https://pytorch.org/docs/stable/tensors.html):
- torch.double
- torch.long

```python
torch.zeros((2, 1, 3, 4), dtype=torch.int)
```
<!-- #region -->
### 其他初始化方法
```python
torch.ones(2, 3, dtype=torch.long) 
```
<!-- #endregion -->

```python
# Size 4. Random permutation of integers from 0 to 9
torch.randperm(10) 
```
## Tensor 属性

```python
tensorA = torch.randn(2,3,4,5)
```

```python
tensorA.shape, tensorA.size() # 形状
```
```python
tensorA.nelement() # 元素个数, total_train += mask.nelement()
```
## Operation

### `mean` 求均值

```python
"""mean(求均值)
"""
A = torch.arange(24, dtype=torch.float).reshape((2,3,4))
A.mean((0)).shape                # torch.Size([3, 4])
A.mean((0), keepdim=True).shape  # torch.Size([1, 3, 4])
# 先在维度1上求均值，再在维度2上求均值
A.mean((1,2), True).shape        # torch.Size([2, 1, 1])
A.mean((1,2), True).shape        # torch.Size([2])
```

### - `clamp` 数值截断

```python
# 在分割或者去模糊等任务评测时，需要注意将神经网路的输出截止到float型`[0.0, 1.0]`或者int型`[0, 255]`，因为最终要保存为图片看效果！
Tensor.clamp(0., 1.)
```
---

# 训练

## 获取每个epoch的batch_size

更新时候需要

```python
for image, target in metric_logger.log_every(data_loader, print_freq, header):
	batch_size = blur_imgs[0].shape[0]
```
---

# 模型

## 查看模型参数

### - `torchsummary`

```python
from torchsummary import summary

net = DeblurNet().cuda()
summary(net, (3, 256, 256)) # 单输入
summary(net, [(3, 256, 256), (1, 256, 256)]) # 多输入
```

### - Legacy

以下方式对模型有要求：每一层输入必须是单个Tensor；每一层顺序连接

```python
import torch
X = torch.rand((1, 3, 256, 256))
net = DeblurNet()

for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)
```

- https://discuss.pytorch.org/t/different-between-permute-transpose-view-which-should-i-use/32916)



## Save and Load

```python
def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

save_on_master({
                  'model': model_without_ddp.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch,
                  'args': args
               }, os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
```

### - Load

```python
model = my_model()
if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
```



## `ToTensor` normalize to `[0, 1]`

`Transforms.ToTensor`



## - Tensor在batch维度上求均值

```python
x = torch.ones(2, 3, 4, dtype=torch.double)

# -1表示自动计算这一维度的大小，1表示在第二维度上计算，注意均值结果的维度变成了1x2
x.view(2,-1).mean(axis=1)
>>>
tensor([1., 1.], dtype=torch.float64)

# 保持求均值之前的维度2x1
x.view(2,-1).mean(axis=1, keepdim=True)
>>>
tensor([[1.],
        [1.]], dtype=torch.float64)
```

## Save and Load

### Save
```python
def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

save_on_master({
                  'model': model_without_ddp.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch,
                  'args': args
               }, os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
```

### Load

```python
model = my_model()
if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
```
