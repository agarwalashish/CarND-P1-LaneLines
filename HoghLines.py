import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

cap = cv2.VideoCapture("solidWhiteRight.mp4")
ret, frame = cap.read()

while(1):
    #decodes and returns the next frame
    ret, frame = cap.read()

    region_line = np.copy(frame)

    #Get the size of the image
    ysize = frame.shape[0]
    xsize = frame.shape[1]

    #Convert the frame to greyscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #Kernel size for gaussian blurring
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(grayFrame, (kernel_size, kernel_size), 0)

    #Defining the color selection criteria
    red_threshold = 195
    green_threshold = 195
    blue_threshold = 195
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    #Mask the pixels below the threshold
    color_thresholds = (frame[:, :, 0] < rgb_threshold[0]) \
                       | (frame[:, :, 1] < rgb_threshold[1]) \
                       | (frame[:, :, 2] < rgb_threshold[2])

    left_bottom = [100, 539]
    right_bottom = [900, 539]
    apex = [400, 0]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    #Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                        (YY > (XX * fit_right[0] + fit_right[1])) & \
                        (YY < (XX * fit_bottom[0] + fit_bottom[1]))

    # Find where image is both colored right and in the region
    #region_line[~color_thresholds & region_thresholds] = [0, 0, 255]

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = frame.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    print (vertices)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #Hough transform parameters
    rho = 5
    theta = np.pi/180
    threshold = 15
    min_line_length = 100
    max_line_gap = 2
    line_image = np.copy(frame) * 0 #create a blank to draw lines on

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image, (x1,y1),(x2,y2),(0,0,255), 10)

    color_edges = np.dstack((edges, edges, edges))

    # Find where image is both colored right and in the region
    region_line[~line_image & region_thresholds] = [0, 0, 255]


    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # cv2.imshow('frame', frameCopy)
    cv2.imshow('frame', combo)

    if cv2.waitKey(25) & 0xFF == ord('q') or ret == False:
        cap.release()
        cv2.destroyAllWindows()
        break
    # cv2.imshow('frame', frameCopy)
    cv2.imshow('frame', combo)