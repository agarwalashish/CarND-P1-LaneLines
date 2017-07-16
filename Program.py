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
    line_image = np.copy(frame)

    #Get the size of the image
    ysize = frameCopy.shape[0]
    xsize = frameCopy.shape[1]

    #Defining the color selection criteria
    red_threshold = 195
    green_threshold = 195
    blue_threshold = 195
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    left_bottom = [100, 539]
    right_bottom = [900, 539]
    apex = [400, 0]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    #Mask the pixels below the threshold
    color_thresholds = (frameCopy[:, :, 0] < rgb_threshold[0]) \
                       | (frameCopy[:, :, 1] < rgb_threshold[1]) \
                       | (frameCopy[:, :, 2] < rgb_threshold[2])

    #Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                        (YY > (XX * fit_right[0] + fit_right[1])) & \
                        (YY < (XX * fit_bottom[0] + fit_bottom[1]))

    #Mask color selection
    frameCopy[color_thresholds] = [0, 0, 0]

    #Find where image is both colored right and in the region
    line_image[~color_thresholds & region_thresholds]= [0, 0, 255]

    #cv2.imshow('frame', frameCopy)
    cv2.imshow('frame', line_image)

    if cv2.waitKey(25) & 0xFF == ord('q') or ret == False :
        cap.release()
        cv2.destroyAllWindows()
        break
    #cv2.imshow('frame', frameCopy)
    cv2.imshow('frame', line_image)
