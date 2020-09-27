set -e

rm -rf _build && mkdir _build

python _jbook/copy.py

rsync -a --exclude='intro.md' _jbook/ _build
cp -r _files _build/

cd _build
sed -i 's/.jupyter.md/.jupyter.ipynb/g' _toc.yml

jb build  ./
