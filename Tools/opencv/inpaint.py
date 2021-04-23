# python inpaint.py --img="test.jpg"

import numpy as np
import cv2
import argparse
import threading

mutex = threading.Lock()
gx = 0
gy = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='Image input file.', default='meixi.png')
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--scope', type=int, default=3)
    parser.add_argument('--save', type=str, default="recovered.png")
    args = parser.parse_args()
    return args


def get_point(event, x, y, flags, param):
    global gx, gy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print "click at (%d, %d)" % (y, x)
        mutex.acquire()
        gx = x
        gy = y
        mutex.release()

if __name__ == '__main__':
    ## 1. init variables
    args = parse_args()
    key = 0
    x = gx
    y = gy
    scope = args.scope

    ## 2. read image
    img = cv2.imread(args.img)
    if img is None:
        print "[Error:] open %s failed, please check your image", args.img
        exit()

    wh = img.shape
    ## (wh[0], wh[1], 1) is: height, width, numchannels
    mask = np.zeros((wh[0], wh[1], 1), np.uint8)
    dst = img[:]
    print img.shape, mask.shape, dst.shape

    ## 3. Create a window to show image
    cv2.namedWindow('recovered')
    cv2.imshow('recovered', dst)
    cv2.waitKey(1)
    cv2.setMouseCallback('recovered', get_point)

    ## press space key to exit
    while key != 32:  # space key
        if x != gx or y != gy:
            ## Get the position which need to be repaired
            mutex.acquire()
            x = gx
            y = gy
            mutex.release()

            ## Do Recover !!!
            mask[y-scope:y+scope, x-scope:x+scope] = 1
            dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
            img = dst[:]
            mask[y-scope:y+scope, x-scope:x+scope] = 0

        cv2.imshow('recovered', dst)
        key = cv2.waitKey(100)
        if key != -1:
            print "press space to exit"

    ## 4. Save the recovered image
    cv2.imwrite(args.save, dst)