# -*- coding: utf-8 -*-
import numpy as np
import os
import json
import pathlib
import pprint


def GetSrcFiles(build_dir, fn="_jbook/_toc.yml"):
    """1. 从"_toc.yml"配置文件读取目标文件
       2. 将后缀'.jupyter.md'的地址改成'.jupyter/*.jupyter.ipynb'
    """
    with open(fn, 'r') as f:
        lines = [line.rstrip().lstrip() for line in f]

    op = os.path
    src_files = []
    for line in lines:
        if not line.startswith("- file: "):
            continue
        
        fn = line[8:]
        if fn.endswith('.jupyter.md'):
            fn_dir = op.dirname(fn)
            bn = op.splitext(op.basename(fn))[0] # bn = *.jupyter
            name = ''.join([bn, '.ipynb']) # name = *.jupyter.ipynb
            fn = op.join(fn_dir, '.jupyter', name)
        src_files.append(fn)

    return src_files


# +
def TEST_GetSrcFiles():
    print('TEST_GetSrcFiles ...')
    build_dir = '_build'
    fn="../_jbook/_toc.yml"

    src_files = GetSrcFiles(build_dir, fn)
    pprint.pprint(src_files)

#TEST_GetSrcFiles()


# -

def NumberHeading(ipynb_dict):
    h1, h2, h3 = 0, 0, 0
    inside_code = False
    for cell in ipynb_dict['cells']:
        if cell['cell_type'] == 'markdown':
            source = cell['source']
            for i, msg in enumerate(source):
                if msg[0:3] == '```':
                    inside_code = ~inside_code
                if inside_code or (msg[0] != '#'):
                    continue

                if msg[0:5] == '#### ':
                    h3 += 1
                    source[i] = ''.join([f'#### {h1}.{h2}.{h3}', msg[4:]])
                elif msg[0:4] == '### ':
                    h2, h3 = h2+1, 0
                    source[i] = ''.join([f'### {h1}.{h2}', msg[3:]])
                    #print(msg)
                elif msg[0:3] == '## ':
                    h1, h2, h3 = h1+1, 0, 0
                    source[i] = ''.join([f'## {h1}', msg[2:]])
                    #print(msg)
                elif msg[0:2] == '# ':
                    h1, h2, h3 = 0, 0, 0
                    #print(msg)


# +
def TEST_NumberHeading():
    fn = '../Untitled.ipynb'
    with open(fn, 'r') as load_f:
        load_dict = json.load(load_f)

    # 对headings进行numbering，如"### 1.1.1 title"
    NumberHeading(load_dict)

#TEST_NumberHeading()


# -

def Copy(src_files, dst_dir, number_heading=True):
    dir_cache = ['']
    op = os.path

    for fn in src_files:
        if fn.endswith('.md'):
            cmd = f'cp --parents {fn} {dst_dir}'
            print(cmd, '......', os.system(cmd) == 0)
            continue
        else:
            fn_dir = os.path.dirname(fn)[:-len('/.jupyter')]
            if fn_dir not in dir_cache:
                _fn_dir = op.join(dst_dir, fn_dir)
                print('mkdir:', _fn_dir)
                pathlib.Path(_fn_dir).mkdir(parents=True, exist_ok=True)
                dir_cache.append(fn_dir)

            with open(fn, 'r') as load_f:
                load_dict = json.load(load_f)

            # 对headings进行numbering，如"### 1.1.1 title"
            NumberHeading(load_dict)

            fn_dst = op.join(dst_dir, fn_dir, op.basename(fn))
            with open(fn_dst, 'w') as dst_file:
                json.dump(load_dict, dst_file)

            print(f'cp {fn} {fn_dst}')


def TEST_Copy():
    print('TEST_Copy ...')
    build_dir = '_build'
    fn="_jbook/_toc.yml"

    src_files = GetSrcFiles(build_dir, fn)
    Copy(src_files, build_dir)


if __name__ == "__main__":
    build_dir = '_build'

    sf = GetSrcFiles(build_dir=build_dir)
    Copy(sf, build_dir)
