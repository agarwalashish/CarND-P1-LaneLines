import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

cap = cv2.VideoCapture("solidWhiteRight.mp4")
ret, frame = cap.read()

while(1):
    #decodes and returns the next frame
    ret, frame = cap.read()

    #Copy the frame
    frameCopy = np.copy(frame)

    red_threshold = 195
    green_threshold = 195
    blue_threshold = 195
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    thresholds = (frameCopy[:, :, 0] < rgb_threshold[0]) \
    |(frameCopy[:, :, 1] < rgb_threshold[1]) \
    |(frameCopy[:, :, 2] < rgb_threshold[2])

    frameCopy[thresholds] = [0, 0, 0]

    cv2.imshow('frame', frameCopy)

    if cv2.waitKey(25) & 0xFF == ord('q') or ret == False :
        cap.release()
        cv2.destroyAllWindows()
        break
    cv2.imshow('frame', frameCopy)

