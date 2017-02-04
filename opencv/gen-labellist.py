from glob import glob
import os

## ./images/14229.bmp||./label/14229.png 0

image_path = 'images/'
label_path = 'label/'
img_names = glob(image_path+'*')
for fn in img_names:
    #print('processing %s... ' % fn)
    path, filename = os.path.split(fn)
    label_fn = filename.split('.')[0]+'.png'
    print './images/'+filename+'||'+'./label/'+label_fn+' 0'
    if not os.path.isfile(label_path+label_fn):
        print 'err '+label_fn
        exit()
