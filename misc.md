左移或右移一维 list/np.ndarray 中所有元素（循环）

> [English Writing] We can see that we correctly shifted all the values one position to the right,
wrapping up from the end of the array back to the begining.

```
def shift_1d(src, move):
    """ move the position by `move` spaces, where positive is 
    to the right, and negative is to the left
    """
    n = len(src)
    if isinstance(src, list):
        dst = [0] * n
    elif isinstance(src, np.ndarray):
        dst = np.zeros(n)
    else:
        raise Exception

    for i in range(n):
      dst[i] = src[(i-move) % n]
    
    return dst
```

---
