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
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help='json file')
    parser.add_argument('--source', type=str, help='the dir of pictures' +
                                                   'or the corresponding video of json file.')
    parser.add_argument('--debug_print', type=bool, default=False, help='[option]for debug')
    parser.add_argument('--debug_imshow', type=bool, default=False, help='[option]for debug')
    parser.add_argument('--dst', type=str, default=None, help='[option]the destination dir to store pictures, for debug')
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


def SelectIgnoreHeadFromJsonObj(jsobj, index):
    try:
        heads = jsobj[index]["head"]
        ignored_heads = []
        for i in range(0, heads.__len__()):
            if i["attrs"]["ignore"] == "yes":
                ignored_heads.append(i["data"])

        frame_id = int(jsonobj[index]["image_key"].replace(".png", ""))  # 10303.png => 10303
        return frame_id, ignored_heads
    except:
        print "Error, index =", index
        return None, None


def getFaceRectfromLanmark(items, attrs):
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


class Ldmk72Parser:
    def __init__(self, jsonfile):
        # Loading Json file.
        jsonfilebuf = open(jsonfile, 'r')
        jsonlines = jsonfilebuf.readlines()
        print "length of %s is %d" % (jsonfile, len(jsonlines))
        self.jsonobj = map(lambda x: json.loads(x), jsonlines)
        if jsonfilebuf is None:
            print "Ldmk72Parser.__init__ Error"


    def getAnnoRes(self, index):
        keywords = "face_keypoint_72"
        try:
            raw_data = self.jsonobj[index][keywords][0]["data"]

            frame_id = int(self.jsonobj[index]["image_key"].replace(".png", ""))  # 10303.png => 10303

            points_list = []
            for i in range(0, len(raw_data)):
                points_list.append((raw_data[i][0], raw_data[i][1]))

            attrs_list = self.jsonobj[index][keywords][0]["point_attrs"]

            if len(attrs_list) == 72 and len(attrs_list) == 72:
                return frame_id, points_list, attrs_list
            else:
                print "Ldmk72Parser.getAnnoRes Error1, line =", index+1
                raw_input("Press any key to continue")
                return None, None, None
        except:
            if 'raw_data' not in locals():
                err = "Missing " + keywords
            elif len(raw_data) == 4:
                err = "Mixed head and " + keywords
            else:
                err = ''
            print "Ldmk72Parser.getAnnoRes Error, %s, in line %d" % (err, index+1)
            # raw_input("Press any key to continue")
            return None, None, None


    def __len__(self):
        return self.jsonobj.__len__()


    def isFrontFace(self, points, attrs):
        for i in range(13, 21):
            if attrs[i] != '':
                return False
        for i in range(30, 38):
            if attrs[i] != '':
                return False

        max_l = points[13][0]
        max_r = points[34][0]
        max_c = (max_l + max_r) / 2
        p_nose = points[57][0]

        tilt_rate = (p_nose - max_c) / (max_r - max_l)
        # print "tilt_rate", tilt_rate
        if math.fabs(tilt_rate) < 0.2:
            return True
        else:
            return False


    def getEyeRect(self, points_list, attrs_list):
        attrs = attrs_list
        eye_rect = dict()
        ### right eye
        if attrs[13] != '' or attrs[17] != '' or attrs[15] != '' or attrs[19] != '':
            eye_rect['r'] = None
        else:
            rl = float(points_list[13][0])  # left border of right eye
            rr = float(points_list[17][0])  # right border of right eye
            rt = float(points_list[15][1])  # top border of right eye
            rb = float(points_list[19][1])  # bottom border of right eye
            eye_rect['r'] = (rl, rr, rt, rb)

        ### left eye
        if attrs[30] != '' or attrs[34] != '' or attrs[32] != '' or attrs[36] != '':
            eye_rect['l'] = None
        else:
            ll = float(points_list[30][0])  # left corner of left eye
            lr = float(points_list[34][0])  # left corner of left eye
            lt = float(points_list[32][1])  # up border of left eye
            lb = float(points_list[36][1])  # down border of left eye
            eye_rect['l'] = (ll, lr, lt, lb)

        return eye_rect


    def getEyeRectInFrontFace(self, points_list, attrs_list):
        attrs = attrs_list
        eye_rect = {'r': None, 'l':None}
        if not self.isFrontFace(points_list, attrs_list):
            return None

        ### right eye
        if attrs[13] != '' or attrs[17] != '' or attrs[15] != '' or attrs[19] != '':
            pass
        else:
            rl = float(points_list[13][0])  # left border of right eye
            rr = float(points_list[17][0])  # right border of right eye
            rt = float(points_list[15][1])  # top border of right eye
            rb = float(points_list[19][1])  # bottom border of right eye
            eye_rect['r'] = (rl, rr, rt, rb)

        ### left eye
        if attrs[30] != '' or attrs[34] != '' or attrs[32] != '' or attrs[36] != '':
            pass
        else:
            ll = float(points_list[30][0])  # left corner of left eye
            lr = float(points_list[34][0])  # left corner of left eye
            lt = float(points_list[32][1])  # up border of left eye
            lb = float(points_list[36][1])  # down border of left eye
            eye_rect['l'] = (ll, lr, lt, lb)

        return eye_rect




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
            if cv2.__version__[0] == '3':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            else:
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

    ldmk_parser = Ldmk72Parser(args.json)

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


    ### Read json line by line
    mode = -1
    for i in range(0, ldmk_parser.__len__()):
        if mode == -1:
            frame_id, landmarklist, point_attrs = ldmk_parser.getAnnoRes(i)
            if frame_id is None:
                continue

            frame = frameReader.read_frame(frame_id)
            eye_rect = ldmk_parser.getEyeRectInFrontFace(landmarklist, point_attrs)
            if eye_rect != None:
                l = map(lambda x: int(x), eye_rect['l'])
                r = map(lambda x: int(x), eye_rect['r'])

                # ml = frame[l[2]:l[3], l[0]:l[1]]
                # mr = frame[r[2]:r[3], r[0]:r[1]]
                # cv2.imwrite('left_'+str(frame_id)+'.png', ml)
                # cv2.imwrite('right_'+str(frame_id)+'.png', mr)

                cv2.rectangle(frame, (l[0], l[2]), (l[1], l[3]), (255, 0, 0))  # left eye
                cv2.rectangle(frame, (r[0], r[2]), (r[1], r[3]), (255, 0, 0))  # right eye
                cv2.imwrite('temp/'+str(frame_id) + '.png', frame)

            if args.debug_imshow:
                cv2.imshow('frame', frame)
                key = cv2.waitKey(0)
                if key == 27:
                    exit()

        elif mode == 0:
            ### Get ingore head
            frame_id, landmarklist, point_attrs = ldmk_parser.getAnnoRes(i)
            if frame_id is not None:
                frame = frameReader.read_frame(frame_id)
                eye_rect = ldmk_parser.getEyeRect(landmarklist, point_attrs)

                if eye_rect['l'] != None:
                    l = map(lambda x: int(x), eye_rect['l'])
                    cv2.rectangle(frame, (l[0], l[2]), (l[1], l[3]), (255, 0, 0))  # left eye
                if eye_rect['r'] != None:
                    r = map(lambda x: int(x), eye_rect['r'])
                    cv2.rectangle(frame, (r[0], r[2]), (r[1], r[3]), (255, 0, 0))  # right eye

                if args.debug_imshow:
                    cv2.imshow('frame', frame)
                    key = cv2.waitKey(0)
                    if key == 27:
                        exit()
        elif mode == 1:
            ### Get face result
            ID, landmarklist, attrs = SelectDataFromJsonObj(jsonobj, i)

            if landmarklist is not None:
                frame = frameReader.read_frame(ID)

                x_min, y_min, x_max, y_max = getFaceRectfromLanmark(landmarklist, attrs)
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
