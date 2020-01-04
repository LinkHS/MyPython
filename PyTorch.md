# 基本操作

## 数据

```python
X = torch.rand((1, 3, 256, 256))
```

## 查看模型参数

### `torchsummary`

```python
from torchsummary import summary

net = DeblurNet().cuda()
summary(net, (3, 256, 256)) # 单输入
summary(net, [(3, 256, 256), (1, 256, 256)]) # 多输入
```

### Legacy

以下方式对模型有要求：每一层输入必须是单个Tensor；每一层顺序连接

```python
import torch
X = torch.rand((1, 3, 256, 256))
net = DeblurNet()

for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)
```

---
##
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