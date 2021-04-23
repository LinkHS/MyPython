import cv2
import numpy as np  # mouse callback function

events = [i for i in dir(cv2) if 'EVENT' in i]
print events

# Create a black image, a window and bind the function to window
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')


# choose demo
print 'press key to choose which demo you want to run'
print '1: demo1\n' \
      '2: demo2\n' \
      '3: demo3'
k = input("Input: ")

if k == 1:
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

    cv2.setMouseCallback('image', draw_circle)
    print 'double click left button on your mouse'

    while 1:
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

elif k == 2:
    drawing = False  # true if mouse is pressed
    mode = True # if True, draw rectangle. Press 'm' to toggle to curve
    ix, iy = -1, -1

    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
                else:
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    cv2.setMouseCallback('image', draw_circle)

    print "default draw rectangle. Press 'm' to toggle to curve"
    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break