#-*- coding: UTF-8 -*-

# Read pictures from 5 pre-categorized folders respecitively, save the image name and its eye status into
# a txt file. Below are 5 types of pre-categorized folders:
# 'open', 'closed', 'uncertain', 'occlusion', 'invisible'

# usage: python sumUp_eye.py

import os
import glob
import numpy as np
import cv2


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source_dir', type=str, default=source_dir)
#     parser.add_argument('--algofile_list', type=str, default=source_dir)
#     parser.add_argument('--save_file', type=str)
#     parser.add_argument('--pic_ext', type=str, default='.png')
#     args = parser.parse_args()
#     if args.source_dir[-1] != os.sep:
#         args.source_dir += os.sep
#     return args

def read_frame(cap, frameid):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameid);
    return cap.read()

class A:
    def __init__(self, file, debug=0):
        lines = open(file).readlines()
        self.algofile_list = map(lambda x: x.strip().split(), lines)

        if debug == 1:
            for dir, file in self.algofile_list:
                print dir, file

    def get_eye_rect(self, label, left_right):
        if left_right == 'right':
            left = float(label[13])
            right = float(label[17])
            pcy = float(label[16])  # point center y axis
            width = right - left  # 1.5w, 0.8w

            left = float(left - width * 0.3)
            right = float(right + width * 0.2)
            top = float(pcy - width * 0.5)
            bottom = float(pcy + width * 0.5)
        elif left_right == 'left':
            left = float(label[19])
            right = float(label[23])
            pcy = float(label[22])  # point center y axis
            width = right - left # 1.5w, 0.8w

            left = float(left - width * 0.3)
            right = float(right + width * 0.2)
            top = float(pcy - width * 0.5)
            bottom = float(pcy + width * 0.5)

        return left, top, right, bottom

    def generate_disturb_rect(self, rect, context, img_w, img_h):
        import random
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        x0 = rect[0] - random.random() * w * context
        x1 = rect[2] + random.random() * w * context
        x2 = rect[2] + random.random() * w * context
        x3 = rect[0] - random.random() * w * context
        y0 = rect[1] - random.random() * w * context
        y1 = rect[1] - random.random() * w * context
        y2 = rect[3] + random.random() * w * context
        y3 = rect[3] + random.random() * w * context
        x0 = 0 if x0 < 0 else x0
        x1 = img_w if x1 > img_w else x1
        x2 = 0 if x2 < 0 else x2
        x3 = img_w if x3 > img_w else x3
        y0 = 0 if y0 < 0 else y0
        y1 = 0 if y1 < 0 else y1
        y2 = img_h if y2 > img_h else y2
        y3 = img_h if y3 > img_h else y3
        res = np.zeros((4, 2), dtype="float32")
        res[:, 0] = [x0, x1, x2, x3]
        res[:, 1] = [y0, y1, y2, y3]
        return res

    def process(self):
        def skip_comments_and_blank(file, cm='#'):
            lines = list()
            for line in file:
                if not line.strip().startswith(cm) and not line.isspace():
                    lines.append(line)
            return lines

        dst_id = 0
        for dir, algo_res_file, video in self.algofile_list:
            print "[Processing]", dir, algo_res_file
            ### 1. Open algo res
            f = open(algo_res_file)
            lines = skip_comments_and_blank(f)
            lines = map(lambda x: x.strip().split(), lines)
            ## "0.png:PrintMX21Landmark" => 0
            frame_id_list = map(lambda x: int(x[0].split(':')[0].split('.')[0]), lines)
            frame_res_list = map(lambda x: x[1:], lines)
            print 'Total %d frames and %d res' % (len(frame_id_list), len(frame_res_list))

            ### 2. Open video
            cap = cv2.VideoCapture(video);
            if not cap.isOpened():
                print "ERROR : the mp4 file open failed."
                raw_input('press any key to continue')
            img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print "video res", img_w, img_h

            ###
            src_files = glob.glob(dir + os.path.sep + '*')
            for img_name in src_files:
                ## /Volumes/TF128-OV/uhome/DMS_Pic/20170418_JOC/12PM/1794856009/closed/left_10842.png => 10842
                frame_id = int(img_name.split(os.path.sep)[-1].split('.')[0].split('_')[-1])
                left_right = img_name.split(os.path.sep)[-1].split('.')[0].split('_')[0]
                idx = frame_id_list.index(frame_id)

                # ldmk_label = PrintCNNFace
                if frame_res_list[idx][0] != 'PrintCNNFace':  # the list may not have face result
                    left, top, right, bottom = self.get_eye_rect(frame_res_list[idx], left_right)
                    eye_rect = [left, top, right, bottom]
                    # left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                    # print 'rect', left, top, right, bottom
                    ret, img = read_frame(cap, frame_id)

                    dst_vertex = np.zeros((4, 2), dtype="float32")
                    dst_vertex[:, 0] = [left, right, right, left]  # [left, right, right, left]
                    dst_vertex[:, 1] = [top, top, bottom, bottom]  # [top, top, bottom, bottom]
                    for k in range(0, 10):
                        context = 0.1
                        smp_vertex = self.generate_disturb_rect(eye_rect, context, img_w, img_h)
                        M = cv2.getPerspectiveTransform(smp_vertex, dst_vertex)

                        # print 'smp_vertex', smp_vertex
                        # print 'dst_vertex', dst_vertex
                        # print 'rect', left, top, right, bottom
                        wrap_img = cv2.warpPerspective(img, M, (img_w, img_h))
                        left, top, right, bottom = int(left), int(top), int(right), int(bottom)


                        # cv2.imshow("src", img[top:bottom+1, left:right+1])
                        # cv2.imshow("wrap", wrap_img[top:bottom+1, left:right+1])
                        # key = cv2.waitKey(0)
                        # if key == 27:
                        #     exit()

                        # fn = os.path.join(outimg_dir, "%06d.png" % k)
                        fn = "/Volumes/TF128-OV/uhome/disturb_closed/" + str(dst_id) + '.png'
                        dst_id += 1
                        print fn
                        wrap_img_roi = wrap_img[top:bottom+1, left:right+1]
                        cv2.imwrite(fn, wrap_img_roi)

                        # out_fn = "%s %d\n" % (fn, ts_types[tstype])
                        # fp_smp_flist.write(out_fn)
                        # if smp_id % 100 == 0:
                        #     print 'generated samples:', smp_id
                else:
                    print "Error", frame_id, frame_res_list[idx]
                    raw_input('press any key to continue')
                # break


if __name__ == '__main__':
    mode = int(raw_input('choose a mode (0, 1):'))
    # scan in the path, find all sub-directory named "closed" or the one user given
    if mode == 0:
        source_dir = raw_input('type in the path, such as\n/Volumes/TF128-OV/uhome/DMS_Pic/\n')
        dirs = os.walk(source_dir)
        for dir in dirs:
            if os.path.sep + 'closed' in dir[0]:
                print dir[0]
    # read a file which includes a folder name list and the corresponding algo result file,
    # the file can be create by run mode 0
    # -------------------------- file example ----------------------------------
    # 20170418_JOC/12PM/1794856009/closed  20170418_JOC/12PM/1794856009_algo.txt  1794856009.mp4
    # 20170418_JOC/12PM/1794856010/closed  20170418_JOC/12PM/1794856010_algo.txt  1794856010.mp4
    elif mode == 1:
        dirs = A('test.txt')
        dirs.process()
    elif mode == 2:  # Geometric Image Transformations
        img = cv2.imread("/Users/austin/Pictures/yema.jpg")
        h, w, c = img.shape
        print h, w, c
        smp_vertex = np.zeros((4, 2), dtype="float32")
        smp_vertex[:, 0] = [-40, w, w+40, 0]
        smp_vertex[:, 1] = [-40, 0, h+40, h]
        dst_vertex = np.zeros((4, 2), dtype="float32")
        dst_vertex[:, 0] = [0, w, w, 0]
        dst_vertex[:, 1] = [0, 0, h, h]
        M = cv2.getPerspectiveTransform(smp_vertex, dst_vertex)
        wrap_img = cv2.warpPerspective(img, M, (w, h))
        cv2.imshow("src", img)
        cv2.imshow("wrap", wrap_img)
        cv2.waitKey(0)
    elif mode == 3:  # Geometric Image Transformations
        img = cv2.imread("/Users/austin/Pictures/yema.jpg")
        h, w, c = img.shape
        print h, w, c
        smp_vertex = np.zeros((4, 2), dtype="float32")
        smp_vertex[:, 0] = [85-40, 460, 460+40, 85]
        smp_vertex[:, 1] = [115-40, 115, 300+40, 300]
        dst_vertex = np.zeros((4, 2), dtype="float32")
        dst_vertex = np.zeros((4, 2), dtype="float32")
        dst_vertex[:, 0] = [85, 460, 460, 85]
        dst_vertex[:, 1] = [115, 115, 300, 300]
        M = cv2.getPerspectiveTransform(smp_vertex, dst_vertex)
        wrap_img = cv2.warpPerspective(img, M, (w, h))
        cv2.imshow("src", img)
        cv2.imshow("src_roi", img[115:301, 85:461])
        cv2.imshow("wrap_roi", wrap_img[115:301, 85:461])
        cv2.waitKey(0)
    else:
        print 'error input'