import cv2 
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='')
    parser.add_argument('ldmk_list', type=str, help='')
    args = parser.parse_args()

    if args.image_dir[-1] != os.sep:
        args.image_dir += os.sep
    return args

def skip_comments_and_blank(file, cm='#'):
    lines = list()
    for line in file:
        if not line.strip().startswith(cm) and not line.isspace():
            lines.append(line)
    return lines

if __name__ == '__main__':
    args = parse_args()

    lines = skip_comments_and_blank(open(args.ldmk_list))
    lines = map(lambda x: x.strip(), lines)

    idx = 0
    while True:
        print lines[idx]
        line = lines[idx].split()
        if len(line) > 1: 
            print "open image:", line[0]
            img = cv2.imread(args.image_dir+line[0])
            for i in range(1, len(line)-1, 2):
                x = int(float(line[i]))
                y = int(float(line[i+1]))
                cv2.circle(img, (x,y), 1, (255,0,0))
            cv2.imshow("img", img)
            key = cv2.waitKey(0)
            if key in [ord('q'), ord('Q')]:
                exit()
        else:
            idx += 1
            continue

        if key in [ord('n')]:
            idx += 10
        elif key in [ord('p')]:
            idx -= 10
        else:
            idx += 1
        
        if key > len(lines) or key < 0:
            key = 0

