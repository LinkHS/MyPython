### python get_landmarks_from_anno.py anno.json out.txt
import json
import math
import argparse
import os


### minimum number of valid landmark numbers, that is invisible number should less than (total - min_lmk_num)
# min_lmk_num = 68

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fn_json', type=str, help='json file to read')
    parser.add_argument('fn_ldmk', type=str, help='landmark result to save')
    parser.add_argument('-img_path_prefix', type=str, default=None, help='[option]add path before image name in the fn_ldmk')
    parser.add_argument('-debug_print', type=bool, default=False, help='[option]for debug')
    parser.add_argument('-debug_imshow', type=bool, default=False, help='[option]for debug')
    args = parser.parse_args()

    if args.img_path_prefix != None and args.img_path_prefix[-1] != os.sep:
        args.img_path_prefix += os.sep
    return args


### points_72_list format: [(x1,y1), (x2,y2), ...]
### attrs_72_list format: ['', 'invisible', ...]
def convertLdmkFrom72To21(points_72_list, attrs_72_list):
    points_21_paired_72 = [0,  2,  3,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    points_72_paired_21 = [22, 26, 39, 43, 13, 21, 17, 30, 38, 34, 50, 57, 53, 58, 60, 67, 70, 64, 62]
    points_21_list = [(-1.0, -1.0) for n in range(21)]

    for i in range(len(points_72_paired_21)):
        idx_72 = points_72_paired_21[i]
        idx_21 = points_21_paired_72[i]
        if attrs_72_list[idx_72] != '':
            points_21_list[idx_21] = (-1.0, -1.0)
        else:
            points_21_list[idx_21] = points_72_list[idx_72]

    if attrs_72_list[24] != '' or attrs_72_list[28] != '':
        points_21_list[1] = (-1, -1)
    else:
        xsum = points_72_list[24][0] + points_72_list[28][0]
        ysum = points_72_list[24][1] + points_72_list[28][1]
        points_21_list[1] = (xsum/2, ysum/2)

    if attrs_72_list[41] != '' or attrs_72_list[46] != '':
        points_21_list[4] = (-1, -1)
    else:
        xsum = points_72_list[41][0] + points_72_list[45][0]
        ysum = points_72_list[41][1] + points_72_list[45][1]
        points_21_list[4] = (xsum/2, ysum/2)

    return points_21_list


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
                # return frame_id, points_list, attrs_list
                return self.jsonobj[index]["image_key"], points_list, attrs_list
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
     #
     # def getVaildLdmkNum(self, attrs):


if __name__ == '__main__':
    args = parse_args()

    print "open", args.fn_json
    ldmk_parser = Ldmk72Parser(args.fn_json)
    out_f = open(args.fn_ldmk, 'w')
    for i in range(0, ldmk_parser.__len__()):
        image_name, points_list, attrs_list = ldmk_parser.getAnnoRes(i)
        if image_name != None:
            out_line = ''
            ldmk21 = convertLdmkFrom72To21(points_list, attrs_list)

            for j in range(len(ldmk21)):
                out_line += ' '
                out_line += ' '.join(['%.2f' % b for b in ldmk21[j]])

            if args.img_path_prefix is not None:
                image_name = args.img_path_prefix + image_name 
            out_f.write(image_name + out_line + '\n')
    out_f.truncate(out_f.tell()-1)
    out_f.close()
