# encoding: UTF-8
import re
import matplotlib.pyplot as plt
import numpy as np

scale_wid_list = list()
scale_hei_list = list()
scale_dis_list = list()
frame_id_list = list()
ttc_list = list()
car_id_list = list()

pattern_sc = re.compile(r'scale_\w \d+\.\d+')  # scale change
pattern_fi = re.compile(r'frame \d+') # frame time
pattern_ttc = re.compile(r'ttc -?\d+\.\d+')  # frame time
pattern_car_id = re.compile(r'ttc vehID \d+')  # frame time


scaleFound = 0
ttcFound = 0
f = open('/Users/austin/work/AdasRule-PC/log.txt', 'r')

# findall() return a list
def analyse_state_machine(file_op):
    def checkCarInList(car_list, car_id):
        idx = None
        _idx = 0
        for _car_info in car_list:
            if car_id == _car_info[0]:
                idx = _idx
                break
            _idx = _idx + 1
        return idx

    ## init variables
    frame_id_list = list()
    car_info = list()  # [car_id, frame_id, [ttc,], [scale_wid,], [scale_hei,], [scale_dis,]]
    car_list = list()  # [car_info, car_info, ...]
    state = 0

    for line in file_op.readlines():
        frame_id = pattern_fi.findall(line)
        if state != 0 and frame_id:
            frame_id = frame_id[0].split()[1]
            # start again, and reset
            print 'reset'

        if state == 0:  # looking for frame id
            if frame_id:
                frame_id = frame_id[0].split()[1]
                frame_id_list.append(frame_id)
                state = 1
        elif state == 1:  # looking for ttc vehId, and ttc
            car_id = pattern_car_id.findall(line)
            if car_id:
                idx = checkCarInList(car_list, car_id)
                if idx:
                    car_list[idx][1] = frame_id
                else:
                    car_list.append([car_id, frame_id])
                print car_list[-1]
                state = 2
        elif state == 2:  # looking for scale change
            scale_changes = pattern_sc.findall(line)
            if scale_changes:
                scale_wid_list.append(scale_changes[0].split()[1])
                scale_hei_list.append(scale_changes[1].split()[1])
                scale_dis_list.append(scale_changes[2].split()[1])
                state = 0
        elif state == 3:  # looking for scale change or ttc
            state = 3


# for line in f.readlines():
#     # findall return a list
#     frame_id = pattern_fi.findall(line)
#     if frame_id:
#         scaleFound = 0
#         ttcFound = 0
#         frame_id_list.append(frame_id[0].split()[1])
#         if len(frame_id_list) > (len(scale_wid_list) + 1):
#             scale_wid_list.append(np.nan)
#             scale_hei_list.append(np.nan)
#             scale_dis_list.append(np.nan)
#             ttc_list.append(np.nan)
#         continue
#
#     scale_changes = pattern_sc.findall(line)
#     if (scaleFound == 0) and scale_changes:
#         scaleFound = 1
#         scale_wid_list.append(scale_changes[0].split()[1])
#         scale_hei_list.append(scale_changes[1].split()[1])
#         scale_dis_list.append(scale_changes[2].split()[1])
#         continue
#
#     _ttc = pattern_ttc.findall(line)
#     if (ttcFound == 0) and _ttc:
#         ttcFound = 1
#         ttc = float(_ttc[0].split()[1])
#         if ( abs(ttc) > 3 ) or ttc < 0:
#             ttc_list.append(np.nan)
#         else:
#             ttc_list.append(ttc)
#         continue

analyse_state_machine(f)

f.close()

exit()

if len(frame_id_list) > len(scale_wid_list):
    scale_wid_list.append(np.nan)
    scale_hei_list.append(np.nan)
    scale_dis_list.append(np.nan)
    ttc_list.append(np.nan)

print len(frame_id_list), len(scale_wid_list), len(scale_hei_list), len(scale_dis_list), len(ttc_list)

#fig = plt.figure(1)
(obj, [ax_ttc, ax_scale]) = plt.subplots(2, 1, True)
ax_ttc.plot(frame_id_list, ttc_list, 'b')
ax_ttc.grid()
ax_scale.plot(frame_id_list, scale_wid_list, 'r', frame_id_list, scale_hei_list, 'g', frame_id_list, scale_dis_list, 'b')
ax_scale.grid()
# plt.figure()
# plt.plot(frame_id_list, scale_wid_list, 'r', frame_id_list, scale_hei_list, 'g', frame_id_list, scale_dis_list, 'b')
# plt.legend(['width', 'height', 'dist'])
# plt.hold()
# plt.figure()
# plt.plot(frame_id_list, ttc_list, 'b')
# plt.grid()
plt.show()

