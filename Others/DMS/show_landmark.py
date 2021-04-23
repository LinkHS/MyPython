import cv2 
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('image_dir', type=str, help='')
    parser.add_argument('source', type=str, help='')
    parser.add_argument('ldmk_list', type=str, help='')
    args = parser.parse_args()

    #if args.image_dir[-1] != os.sep:
    #    args.image_dir += os.sep
    return args

def skip_comments_and_blank(file, cm='#'):
    lines = list()
    for line in file:
        if not line.strip().startswith(cm) and not line.isspace():
            lines.append(line)
    return lines

class FrameReader:
    # source could be a name of a video or a folder containing pictures
    # such as "*.mp4", "dir"
    def __init__(self, source):
        if ".mp4" in source:
            # Loading mp4 file.
            mp4Cap = cv2.VideoCapture(source)
            if not mp4Cap.isOpened():
                print "ERROR : the mp4 file open failed:", source
                return None

            self.mode = 0
            self.cap = mp4Cap
        elif os.path.isdir(source):
            self.mode = 1
        else:
            print "Error source", source
            return None

        self.source = source

    def read_frame(self, img_name):
        if self.mode == 0:
            #self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_id)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(img_name.split('.')[-2]))
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                return None
        elif self.mode == 1:
            _img_name = self.source + os.sep + img_name
            img = cv2.imread(_img_name)
            return img



if __name__ == '__main__':
    args = parse_args()

    print "the source is", args.source
    frameReader = FrameReader(args.source)
    
    lines = skip_comments_and_blank(open(args.ldmk_list))
    lines = map(lambda x: x.strip(), lines)

    idx = 0
    while True:
        print lines[idx]
        line = lines[idx].split()
        if len(line) > 1: 
            print "open image:", line[0]
            img = frameReader.read_frame(line[0])
            for i in range(1, len(line)-1, 2):
                x = int(float(line[i]))
                y = int(float(line[i+1]))
                cv2.circle(img, (x,y), 1, (255,0,0))
            cv2.imshow("img", img)
            key = cv2.waitKey(0)
            if key in [ord('q'), ord('Q')]:
                print key, ord('q'), ord('Q')
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

