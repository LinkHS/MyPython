## 画图

```python
import matplotlib.pyplot as plt

plt.plot(x, y, 'bo')  # plot x and y using blue circle markers
plt.plot(y)           # plot y using x as index array 0..N-1
plt.plot(y, 'r+')     # ditto, but with red plusses

# 给数据打上标签
plt.plot(x, 'r--', label="inv1") # 
plt.legend(loc="best",fontsize=15)
```

其他格式参考 [matplotlib.pyplot.plot](https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.htm)



## 调整图像大小

```python
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 9]
```





# Examples

## 画图比较
```python
def plot_comparison(ori_img, res_img, ori_tit='Original', res_tit='Result', fontsize=30, 
                    ori_cmap='gray', res_cmap='gray'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
    f.tight_layout()

    ax1.imshow(ori_img, cmap=ori_cmap)
    ax2.imshow(res_img, cmap=res_cmap)

    ax1.set_title(ori_tit, fontsize=fontsize)
    ax2.set_title(res_tit, fontsize=fontsize)


def plot_stack(imgs, titles=None, fontsizes=None, cmaps=None, hv='h', figsize=(15, 9)):
    """
    @hv, plot in a horizontal column or a vertical row 
    """
    if not isinstance(imgs, list): # incase `imgs` in a single image not a list
        imgs = [imgs]
    
    n = len(imgs)
    
    _titles, _fontsizes, _cmaps = ['']*n, [50/n]*n, ['gray']*n
    if titles != None:
        _titles[:len(titles)] = titles
    if fontsizes != None:
        _fontsizes[:len(fontsizes)] = fontsizes 
    if cmaps != None:
        _cmaps[:len(cmaps)] = cmaps 
    
    #print(_titles, _fontsize, _cmaps)
    
    f, axes = plt.subplots(1, n, figsize=figsize)
    f.tight_layout()
    
    # in case `imgs` in a single image not a list
    axes = [axes] if len(imgs) == 1 else axes.tolist()

    for ax, img, cmap, fs, tit in zip(axes, imgs, _cmaps, _fontsizes, _titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(tit, fontsize=fs)

    return f, axes
```
