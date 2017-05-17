# -*- coding:UTF-8 -*-

# python /media/psf/Home/work/MyPython/DMS/getFaceRectfromJson_new.py\
#        --source=/media/psf/TF128-OV/uhome/DMS_VideoDB/20170414_JOC/day.mp4\
#        --json=20170414_day.json

# python /media/psf/Home/work/MyPython/DMS/getFaceRectfromJson_new.py\
#        --source=../haoming/\
#        --json=20170414_day.json
#        --dst=true

import os
import cv2
import numpy as np
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help='json file')
    parser.add_argument('--source', type=str, help='the dir of pictures' +
                                                   'or the corresponding video of json file.')
    parser.add_argument('--debug_print', type=bool, default=False, help='for debug')
    parser.add_argument('--dst', type=str, default=None, help='the destination dir to store pictures, for debug')
    args = parser.parse_args()
    ## add os.sep if needed
    if args.dst and args.dst[-1] != os.sep:
        args.dst += os.sep
    return args


def SelectDataFromJsonObj(jsobj, index):
    try:
        l = jsobj[index]["face_keypoint_72"][0]["data"]
        b = []
        for i in range(0, l.__len__()):
            b.append(l[i][0])
            b.append(l[i][1])
        # ID = jsonobj[index]["image_key"].replace(".png", "").replace(".jpg", "").replace("res_", "")
        frame_id = int(jsonobj[index]["image_key"].replace(".png", ""))  # 10303.png => 10303
        attrs = jsonobj[index]["face_keypoint_72"][0]["point_attrs"]
        if b.__len__() == 144 and attrs.__len__() == 72:
            return frame_id, b, attrs
        else:
            print "Error, frame_id =", frame_id
            return None, None, None
    except:
        print "Error, index =", index
        return None, None, None


def getFaceRecfromLanmark(items, attrs):
    box = None
    new_center_ind = 57  # Index of center point.
    lmks = np.array([float(s) for s in items]).reshape((72, 2))  # To reshape 144 list to (72*2) array.
    for j in range(lmks.shape[0]):  # To walk 72 points to processing data.
        if lmks[j, 0] == -1 or lmks[j, 1] == -1 or attrs[j] != '':
            continue
        if not box:
            box = [lmks[j, 0], lmks[j, 1], lmks[j, 0], lmks[j, 1]]
        else:
            box[0] = min(box[0], lmks[j, 0])
            box[1] = min(box[1], lmks[j, 1])
            box[2] = max(box[2], lmks[j, 0])
            box[3] = max(box[3], lmks[j, 1])

    if new_center_ind is not None:
        cx = lmks[new_center_ind, 0]
        cy = lmks[new_center_ind, 1]
    else:
        cx = (box[0] + box[2]) / 2.
        cy = (box[1] + box[3]) / 2.

    half_len = (box[3] - box[1] + box[2] - box[0]) / 4.
    box[0] = int(cx - half_len)
    box[1] = int(cy - half_len)
    box[2] = int(cx + half_len)
    box[3] = int(cy + half_len)
    return box


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


    def read_frame(self, frame_id):
        if self.mode == 0:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                return None
        elif self.mode == 1:
            img_name = self.source + os.sep + str(frame_id) + '.png'
            img = cv2.imread(img_name)
            return img


if __name__ == '__main__':
    args = parse_args()

    # Loading Json file.
    jsonfilebuf = open(args.json, 'r')
    jsonlines = jsonfilebuf.readlines()
    jsonobj = map(lambda x: json.loads(x), jsonlines)

    print "the source is", args.source
    frameReader = FrameReader(args.source)

    ### Create dest folders.
    if args.dst:
        path_frame = args.dst + "rectface_fromLandmark"
        path_face = args.dst + "subface_fromLandmark"
        if not os.path.exists(path_frame):
            os.makedirs(path_frame)
        if not os.path.exists(path_face):
            os.makedirs(path_face)

    for i in range(0, jsonobj.__len__()):
        ID, landmarklist, attrs = SelectDataFromJsonObj(jsonobj, i)

        if landmarklist is not None:
            frame = frameReader.read_frame(ID)

            x_min, y_min, x_max, y_max = getFaceRecfromLanmark(landmarklist, attrs)
            if args.debug_print:
                print x_min, y_min, x_max, y_max

            subframe = frame[y_min:y_max, x_min:x_max]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0))

            if args.dst:
                if args.debug_print:
                    print path_frame + os.path.sep + str(ID) + ".png"
                    print path_face + os.path.sep + str(ID) + ".png"

                cv2.imwrite(path_frame + os.path.sep + str(ID) + ".png", frame)
                cv2.imwrite(path_face + os.path.sep + str(ID) + ".png", subframe)