```
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
