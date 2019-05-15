#!/usr/bin/env python
import freenect
import cv2
import numpy as np
import frame_convert2

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')

cv2.namedWindow("tracking")

def get_depth():
    array = freenect.sync_get_depth()[0]
    # print(array[225][326])
    return frame_convert2.pretty_depth_cv(array)


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

image = get_video()
bbox = cv2.selectROI("tracking", image)
tracker = cv2.TrackerMIL_create()
init_once = False



while 1:
    cv2.imshow('Depth', get_depth())
    image = get_video()
    print(image.shape)
    cv2.imshow('Video', image)
    print(type(image))
    if not init_once:
        ok = tracker.init(image, bbox)
        init_once = True
    ok, newbox = tracker.update(image)
    print ok, newbox

    if ok:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        print(p1, p2)
        cv2.rectangle(image, p1, p2, (200,0,0))
    cv2.imwrite('final.jpg', image)
    assert False
    cv2.imshow("tracking", image)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break # esc pressed
    # if cv2.waitKey(10) == 27:
    #     break
