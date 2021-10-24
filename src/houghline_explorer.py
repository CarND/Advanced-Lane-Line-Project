# code to test effect of hsv tweaks on image to determine min and max 
# values of hsv image for masking yellow/white lane lines
# from https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv#comment34389698_22588395

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pathlib import Path
root = Path(".")
p = Path(__file__).parents[1]
print(root)
print(p)
import sys
sys.path.append(p)

def draw_lines(img, houghLines, color=[0, 255, 0], thickness=2):
    for line in houghLines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
 
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)   
 
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def explore_houghline(image):
    def nothing(x):
        pass

    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('rho','image',1,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('theta','image',0,int(np.pi/180),nothing)
    cv2.createTrackbar('threshold','image',1,255,nothing)
    cv2.createTrackbar('minLineLength','image',1,179,nothing)
    cv2.createTrackbar('maxLineGap','image',1,255,nothing)
 

    # # Set default value for MAX HSV trackbars.
    # cv2.setTrackbarPos('rho', 'image', 1)
    # cv2.setTrackbarPos('theta', 'image', np.pi/2)
    # cv2.setTrackbarPos('threshold', 'image', 1)
    # cv2.setTrackbarPos('minLineLength', 'image', 1)
    # cv2.setTrackbarPos('maxLineGap', 'image', 1)

    # Initialize to check if HSV min/max value changes
    rho = theta = threshold = minLineLength = maxLineGap = 0
    prho = ptheta = pthreshold = pminLineLength = pmaxLineGap = 0

    output = image
    wait_time = 33

    while(1):

        # get current positions of all trackbars
        rho = cv2.getTrackbarPos('rho','image')
        theta = cv2.getTrackbarPos('theta','image')
        threshold = cv2.getTrackbarPos('threshold','image')
        minLineLength = cv2.getTrackbarPos('minLineLength','image')
        maxLineGap = cv2.getTrackbarPos('maxLineGap','image')

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. Blur image 
        blurred_img = cv2.medianBlur(gray_img, ksize=3)
        # 3. Do Canny transform to get edges
        lo_thresh, hi_thresh = (100, 250)
    #    lo_thresh, hi_thresh = (60, 250)    
        canny_edges_img = cv2.Canny(blurred_img, lo_thresh, hi_thresh)
        # 4. Use crude roi to isolate general region of interest
        # 5. Feed Canny output to Houghlines to find lines in region of interest 
        lines = cv2.HoughLinesP (
            canny_edges_img, 
            rho=rho, 
            theta=theta, 
            threshold=threshold, 
            minLineLength=minLineLength, 
            maxLineGap=maxLineGap ) 
        

        hough_lines_image = np.zeros_like(image)
        
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(hough_lines_image,(x1,y1),(x2,y2),(0,255,0),2)

 
        # draw_lines(hough_lines_image, hough_lines)
        original_image_with_hough_lines = weighted_img(hough_lines_image,image)

        # Print if there is a change in HSV value
        if( (prho != rho) | (ptheta != theta) | (pthreshold != threshold) | 
            (pminLineLength != minLineLength) | (pmaxLineGap != maxLineGap) ):
            print("(rho = %d , theta = %d, threshold = %d), (minLineLength = %d , maxLineGap = %d)" % (rho , theta , threshold, minLineLength, maxLineGap))
            prho = rho
            ptheta = theta
            pthreshold = threshold
            pminLineLength = minLineLength
            pmaxLineGap = maxLineGap


        # Display output image
        cv2.imshow('image',original_image_with_hough_lines)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# image = cv2.imread('test_images/straight_lines2.jpg')
# plt.imshow('original', image)
# explore_houghline(image)
    # good for 1st solidWhiteRight
    # (hMin = 0 , sMin = 0, vMin = 157), (hMax = 179 , sMax = 25, vMax = 255)

    # good for 1st solidYellowCurve
    # (hMin = 95 , sMin = 10, vMin = 129), (hMax = 179 , sMax = 155, vMax = 255)