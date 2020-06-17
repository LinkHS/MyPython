# jupyter-book
## Build
Config jupytext to pair `.jupyter.md` and `.jupyter/.jupyter.ipynb` in `~/.config/jupytext`

```
default_jupytext_formats = ".jupyter.md,.jupyter//.jupyter.ipynb"
```

Run below command in the top level folder of this repo
```
sh _jbook/build.sh
```
The script will look into `_toc.yml` and copy needed files to `_build` folder: `*.jupyter.md` will be copied from the corresponding `*.jupyter/.jupyter.ipynb`, and the headings in `*.ipynb` will numbered (starts from level-2 - `##`); `*.md` files are directly copied.

## Known Issues
1. Be careful about the images linked in files are auto copied by jupyter-book, if not, there will be output warnings.

