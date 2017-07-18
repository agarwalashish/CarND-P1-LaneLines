#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import pdb

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Applies Canny transform
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

#Applies a Gaussian Blur to the image with the noise filter
def gauusian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#Only keeps the region of interest defined by the polygon
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #Fill the pixels in the polygon with the specified color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #retrun the image where only the pixels are non-zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0, 0, 255], thickness =2):

    leftLinePointsX = []
    leftLinePointsY = []
    rightLinePointsX = []
    rightLinePointsY = []
    count = len(lines)
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope_m = (y1 - y2)/(x1 - x2)

            if slope_m < 0:
                leftLinePointsX.append(x1)
                leftLinePointsX.append(x2)
                leftLinePointsY.append(y1)
                leftLinePointsY.append(y2)
            else:
                rightLinePointsX.append(x1)
                rightLinePointsX.append(x2)
                rightLinePointsY.append(y1)
                rightLinePointsY.append(y2)

            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_image(img, initial_img, alpha = 2, beta = 1., gamma = 0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def slope_line(x1, x2, m, c):
    y1 = m*x1 + c
    y2 = m*x2 + c

def image_pipeline(image):
    grayscale_image = grayscale(image)

    gaussian_blurred_image = gauusian_blur(grayscale_image, 7)

    canny_image = canny(gaussian_blurred_image, 50, 150)

    region = [[(440, 300), (500, 300), (880, 540), (80, 540)]]
    region_of_interest_image = region_of_interest(canny_image, np.array(region, dtype=np.int32))

    theta = np.pi / 180
    hough_lines_image = hough_lines(region_of_interest_image, 2, theta, 3, 20, 5)

    weighted = weighted_image(image, hough_lines_image, alpha=2.0, beta=1.0, gamma=1.0)

    return weighted

def video_pipeline(file_name):
    capture = cv2.VideoCapture(file_name)
    ret, frame = capture.read()

    while(1):
        ret, frame = capture.read()

        weighted = image_pipeline(frame)

        cv2.imshow('frame', weighted)

        if cv2.waitKey(25) & 0xFF == ord('q') or ret == False:
            capture.release()
            cv2.destroyAllWindows()
            break

        cv2.imshow('frame', weighted)

def main():
    video = 'test_videos/solidWhiteRight.mp4'
    video_pipeline(video)

    # image = mpimg.imread('test_images/solidYellowLeft.jpg')
    #
    # weightedimg = image_pipeline(image)
    #
    # # printing out some stats and plotting
    # print('This image is:', type(weightedimg), 'with dimensions:', weightedimg.shape)
    # plt.imshow(weightedimg)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    # plt.show()

main()