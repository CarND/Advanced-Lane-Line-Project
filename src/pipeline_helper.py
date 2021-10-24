
from re import L
from matplotlib import image
import numpy as np 
import math
import cv2 
import matplotlib.pyplot as plt
from collections import defaultdict

from numpy.core.fromnumeric import mean
from lane_processor import LaneProcessor

from numpy.__config__ import show
from numpy.lib.polynomial import _raise_power
from image_historian import ImageHistorian
# from camera_calibrater import calibrate_camera

import sys 

# prints entire np array for testing
np.set_printoptions(threshold=sys.maxsize)

class PipelineHelper:

    # offset used to project SRC pts to DST pts where DST_OFFSET provides
    # a buffer from the left and right edge to prevent curvatures from
    # being cut off at edge 
    DST_OFFSET = 260
    # relative tolerance - for comparing if two values are close (0.01 = 1 percent)
    RTOL = 0.01

    # Number of frames where lanes not found before reseting cornerpts
    max_iter_lanes_not_found = 5
    
    # disortion coefficient needed to fix camera distortion
    dist_coef = {}
    # original image (or undistorted image if camera calibrated)
    img = None
    # binary of original (or undistorted image if camera calibrated)
    binary = None
    # lanes instance holds info for road lines(driver view) and lanes(bird view)
    lanes = None    
    # region of interest parameters used to narrow image area to identify lane
    roi_params = {}
    # source and destination points required to warp img to bird-eye view and back
    src = None 
    dst = None

    historian = None 

    left_lane_pixel_count = 0
    right_lane_pixel_count = 0

    # margin to left/right to look for lane pixels
    margin = 15 

    def __init__(self, mtx=None, dist=None) -> None:
        # initialize with distortion coefficients
        self.dist_coef["mtx"] = mtx 
        self.dist_coef["dist"] = dist

    def slope_intercept(self, line):
        # calculate y intercept (b) and returns slope, b  
        if len(line) == 5:
            m, x1, y1, _, _ = line    
            b = y1 - m * x1
        elif len(line) == 4:
            x1, y1, x2, y2 = line
            m = y2-y1 / x2-x1 
            b = y1 - m * x1
        return (m, b)        

    def get_y_hor(self):
        """
        Get x at y nearest horizon (convergence of lanes).

        Return
        ------
        (int) y near horizon for evaluating polyfit to find x
        """
        # y at which to calculate x for determining coord of lane ends
        try:
            y_horizon = self.roi_params["bottom"]
            return y_horizon
        except KeyError:
            print("please make sure to run pipeline.set_roi_params(img) to determine y eval")
            return (self.img.shape[0]//2) # half height of image 

    def get_y_car(self):
        """
        Get x at y nearest car (top of unside-down image).

        Return
        ------
        (int) y at car for evaluating polyfit to find x
        """
        # y at which to calculate x for determining coord of lane ends
        try:
            y_car = self.roi_params["top"]
            return y_car
        except KeyError:
            print("please make sure to run pipeline.set_roi_params(img) to determine y eval")
            return (self.img.shape[0]) # height of image

    def set_roi_params(self, img):
        self.img = img
        h, w = img.shape[:2]
        self.roi_params["img_height"] = h 
        self.roi_params["img_width"] = w
        # horizontal center, left-right, int value
        self.roi_params["hor_center"] = w // 2 
        # vertical center, up-down, int value 
        self.roi_params["ver_center"] = h // 2 # center up-down  
        # top and bottom relative to flipped image where image poits down
        # 0,0 is bottom while 0, h is top 
        self.roi_params["bottom"] = self.roi_params["ver_center"] + 93 
        self.roi_params["top"] = self.roi_params["img_height"] - 30
        self.roi_params["offset"] = int(w * .05) # 5% of width
        # offset left and right edge for birds eye transform
        self.roi_params["dst_offset"] = self.DST_OFFSET
        # print("roi_params: ", self.roi_params)
        # set image dimensions for lane processor instance
        self.lanes = LaneProcessor()
        self.lanes.set_img(img)
        self.lanes.set_img_dim(img)
        self.lanes.set_y_eval(self.get_y_hor(), self.get_y_car())
        self.dst = self.default_dst_pts()

    def get_birds_eye_lane_width(self):
        lt, rt, _, _ = self.default_dst_pts()
        x1, _ = lt
        x2, _ = rt 
        return x2-x1

    def set_binary_image(self, binary_img):
        self.binary = binary_img

    def set_src(self, corner_pts):
        self.src = corner_pts

    def add_label(self, img, label, loc=(50,50)):
        cv2.putText(img, label, loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1, color=[50,250,50], thickness=2)
        return img

    def calc_curvature(self, A, B, y):
        curvature = (1 + (2*A*y + B)**2)**(3/2) / abs(2*A)
        return curvature
    
    def plot_cornerpts(self, cornerpts):
        if cornerpts is None:
            return

        plt.imshow(self.img)
        for (x,y) in cornerpts:
            plt.scatter(x, y)
        plt.show()

    def draw_poly(self, img, poly, description='draw poly'):
        canvas = img 
        h, w = self.img.shape[:2]
        start = self.get_y_hor()
        end = self.get_y_car()
        y = np.linspace(start, end, end-start)
        # print("h, w, y ", h, w, max(y))
        if len(poly) == 2:
            # line x = my + b
            x = poly[0] * y + poly[1]             
        elif len(poly) == 3:
            # curve x = Ay2 + By + C
            x = poly[0] * y**2 + poly[1] * y + poly[2]
        else:
            x = [0 for _ in y]
        # print("canvas shape ", canvas.shape)
        y = np.array([min(max(i,0), h-1) for i in y], dtype=np.int32)
        x = np.array([min(max(i,0), w-1) for i in x], dtype=np.int32)
        # print("y min, max, x min, max ", min(y), max(y), min(x), max(x))
        canvas[y,x] = [0,255,0]
        self.show_image(description, canvas)

    def show_image(self, description, img):
        if img is not None:
            cv2.imshow(description, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_image(self, description, img):
        if img is not None:
            cv2.imshow(description, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def optimize_lane_lines(self, img):
        """
        Use threshold to optimize lane pixels in image    
        
        Return:
        Image (3 channels) 

        """
        image = np.copy(img)

        # R channel binary is good in finding colored lanes         
        R = image[:,:,0]
        thresh = (200, 255)
        r_binary = np.zeros_like(R)
        r_binary[(R > thresh[0]) & (R <= thresh[1])] = 1

        # S channel binary good for finding lanes in shadow conditions    
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        s_thresh = (170, 230)
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # Sobel x detects changes in x, good for detecting vertical oriented lines
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        thresh = (20, 70)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Stack all 3 binary channels and convert to color for easy id of 
        # contribution of binary channel to final image 
        color_binary = np.dstack((r_binary, sxbinary, s_binary)) * 255    
        
        return color_binary

    def correct_image_distortion(self, img, mtx, dist):

        if mtx is not None or dist is not None:
            undistorted_image = cv2.undistort(img, mtx, dist, None, mtx)
            return undistorted_image
        else:
            print("Run calibrate(chessboard_images_folder_path) first.")
            return img

    def undistort_image(self, img):
        # corrects image disortion if distortion coeffs available
        if not self.dist_coef:
            # threshold orig img if no calibration coef to correct distortion
            self.img = img
            return img 
        else:
            mtx = self.dist_coef["mtx"]
            dist = self.dist_coef["dist"]
            undistorted_img = self.correct_image_distortion(img, mtx, dist)
            self.img = undistorted_img
            return undistorted_img
            
    def visualize_houghlines(self, lines):
        canvas = np.zeros(self.img.shape)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(canvas, (x1,y1), (x2,y2), [255,0,0], 3)
        return canvas

    def default_roi_vertices(self):
        """
        given image and height and width of image, returns tuple of 4 point vertices
        of (x, y) points (top-left, top-right, bottom-right, bottom-left)
        """
        center = self.roi_params["hor_center"]
        offset = self.roi_params["offset"]
        w = self.roi_params["img_width"]
        y_car = self.roi_params["top"]
        y_horizon = self.roi_params["bottom"]
        x_left_horizon = center - offset * 2.3 #2.5 # 3 # 4 #2.5   
        x_right_horizon = center + offset * 2.3 #2.5 # 3 # 4 #2.5
        x_left_car = 0 + offset * 2.2 #2
        x_right_car = w - offset  * 2.2 #2
        # order of 4 corners: lt -> rt -> rb -> lb
        vertices = ((x_left_horizon, y_horizon), (x_right_horizon, y_horizon), 
                (x_right_car, y_car), (x_left_car, y_car))
        # array of int required by np.fillPoly
        return np.array([vertices], dtype=np.int32)

    def region_of_interest(self, img, vertices=None):
        """        
        given image and vertices, return image which only keeps region defined by 
        the polygon formed from `vertices` and rest of the image is set to black
        `vertices` should be a numpy array of integer points.
        """
        if vertices is None:
            vertices = self.default_roi_vertices() #self.roi_vertices(img)


        #defining a blank mask to start with
        mask = np.zeros_like(img)   

        #define a 3 or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        # self.show_image("region of interest", cv2.bitwise_and(img, mask))
        #return image with nonzero mask pixels 
        return cv2.bitwise_and(img, mask)

    def categorize_houghlines(self, lines): 
        """
        Categorizes lines as left or right lane lines by slope, position.

        Return
        ------
        left and right list of line tuple (slope, length, x1, y1, x2, y2) 

        """

        left_lane_candidates = [] 
        right_lane_candidates = []
        # print("lines count and coords: ", len(lines), lines)
        for line in lines:
            for x1,y1,x2,y2 in line:
                # skip if vertical line 
                if x2-x1 == 0:
                    continue

                slope = (y2-y1)/(x2-x1)
                if self.is_possible_left_lane(line):
                    left_lane_candidates.append((slope, x1, y1, x2, y2))

                if self.is_possible_right_lane(line):
                    right_lane_candidates.append((slope, x1, y1, x2, y2))
        # print("left and right houghline count ", len(left_lane_lines), len(right_lane_lines))
        return (left_lane_candidates, right_lane_candidates)

    def x_intercepts(self, line, y_tip, y_base):
        
        # print("x_interecept line ", line)
        if len(line) == 5:
            m, x1, y1, _, _ = line
            b = y1 - (m * x1)
        elif len(line) == 4:
            x1, y1, x2, y2 = line
            m = y2-y1 / x2-x1
            b = y1 - (m * x1)
        elif len(line) == 2:
            m, b = line 
        else:
            return (None, None)
        x_tip = (y_tip - b) // m
        x_base = (y_base - b) // m
        return (int(x_tip), int(x_base))

    def x_given_liney(self, line, y):
        m, x1, y1, _, _ = line    
        b = y1 - m * x1
        x = np.int((y - b) / m)
        return x

    def xfit(self, fit, y_eval):
        x = fit[0]*y_eval**2 + fit[1]*y_eval + fit[2]
        return int(x)

    def image_to_binary(self, img, thresh=1):
        """ 
        Converts any image to binary, with optional threshold for pixel.
        Accepts both grayscale or color.
        
        Return:
        -------
        Binary image in 2D.

        """
        # print("img len, img width", len(img), len(img[1]))
        img = np.asarray(img)
        binary = np.zeros(img.shape[:2])             

        if len(img.shape) == 2:
            binary[ img>thresh ] = 1            
        elif len(img.shape) >= 3:
            R = img[:,:,0]
            G = img[:,:,1]
            B = img[:,:,2]
            binary[ (R>thresh) | (G>thresh) | (B>thresh) ] = 1

        return binary

    def plot_pts(self, x, y, img):
        plt.imshow(img)
        plt.scatter(x, y)
        plt.show()
   

    def fitpoly(self, x, y, dim=2):
        # switch y, x since fitting to curve that opens to left and right
        if x is not None and y is not None and np.sum(x + y) > 0:
            fit = np.polyfit(y, x, dim)
            return fit 
        else:
            return None 

    def fit2inds(self, img, polyfit=None) -> tuple:
            """
            Use polyfit to find lane indices 

            Return
            ------
            tuple of list representing indices of x and y ([x], [y]) or
            (None, None) if no indices found  

            """
            if polyfit is None:
                return (None, None)

            # Set the width of the windows +/- margin
            margin = self.margin # 5
            
            # Identify the x and y positions of all nonzero pixels in the image
            nz = img.nonzero()
            nzy = np.array(nz[0])
            nzx = np.array(nz[1])
            
            x = y = None

            A = polyfit[0]
            B = polyfit[1]
            C = polyfit[2]

            # Create empty lists to receive left and right lane pixel indices
            lane_inds = ((nzx >= A * nzy**2 + B * nzy + C - margin) & 
                    (nzx <= A * nzy**2 + B * nzy + C + margin)).nonzero()[0]
            
            if lane_inds != []:
                # Extract line pixel positions   
                x = nzx[lane_inds]
                y = nzy[lane_inds]
            
            return (x, y)

    def fits2inds(self, img, lfit=None, rfit=None) -> tuple:
            """
            Use left and right polyfit to find lane indices 

            Return
            ------
            tuple of 4 list representing indices of x and y of left 
            and right lane ([lx], [ly], [rx], [ry]) 
            (None, None) if no indices found  

            """
            if lfit is None and rfit is None:
                return (None, None, None, None)

            # Set the width of the windows +/- margin
            margin = self.margin #5
            
            # Identify the x and y positions of all nonzero pixels in the image
            nz = img.nonzero()
            nzy = np.array(nz[0])
            nzx = np.array(nz[1])
            
            lx = ly = rx = ry = None

            A = lfit[0]
            B = lfit[1]
            C = lfit[2]

            # Create empty lists to receive left and right lane pixel indices
            l_inds = ((nzx >= A * nzy**2 + B * nzy + C - margin) & 
                    (nzx <= A * nzy**2 + B * nzy + C + margin)).nonzero()[0]
            
            if l_inds != []:
                # Extract line pixel positions   
                lx = nzx[l_inds]
                ly = nzy[l_inds]

            A = rfit[0]
            B = rfit[1]
            C = rfit[2]

            # Create empty lists to receive left and right lane pixel indices
            r_inds = ((nzx >= A * nzy**2 + B * nzy + C - margin) & 
                    (nzx <= A * nzy**2 + B * nzy + C + margin)).nonzero()[0]
            
            if r_inds != []:
                # Extract line pixel positions   
                rx = nzx[r_inds]
                ry = nzy[r_inds]

            return (lx, ly, rx, ry)

    def lane_inds_from_line(self, binary_img, line) -> tuple:
        """
        Used for driver view images, finds indices that overlap given line 
        +/- margin, which is needed to capture curvature of road in distance

        Note that range of y is within region of interest only

        Return
        ------
        tuple of indices (xs, ys)

        """
        if line is None:
            return (None, None)

        # Set the width of +/- margin, want to be small enough for precision
        # but large enough to capture small curve in distant 
        margin = self.margin #2 
        
        # Identify the x and y positions of all nonzero pixels in the image
        nz = binary_img.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])
        
        # x and y indices where line overlaps nonzero pixels
        xs = ys = []

        # calculate slope and y-intercept of line
        m, b = self.slope_intercept(line)

        # Create empty lists to receive left and right lane pixel indices
        inds = ((nzx >= (nzy - b)/m - margin) & 
                (nzx <= (nzy - b)/m + margin) & 
                (nzy >= self.roi_params["bottom"]) &
                (nzy <= self.roi_params["top"])
                ).nonzero()[0]
        
        if inds != []:
            # Extract line pixel positions   
            xs = nzx[inds]
            ys = nzy[inds]
        
        return (xs, ys)


    def lane_inds_from_polyfit(self, img, polyfit) -> tuple:
            """
            Use polyfit to find lane indices 

            Return
            ------
            indices tuple (x, y)

            """
            if polyfit is None:
                return (None, None)

            # Set the width of the windows +/- margin
            margin = self.margin # 5
            
            # Identify the x and y positions of all nonzero pixels in the image
            nz = img.nonzero()
            nzy = np.array(nz[0])
            nzx = np.array(nz[1])
            
            x = y = None

            A = polyfit[0]
            B = polyfit[1]
            C = polyfit[2]

            # Create empty lists to receive left and right lane pixel indices
            lane_inds = ((nzx >= A * nzy**2 + B * nzy + C - margin) & 
                    (nzx <= A * nzy**2 + B * nzy + C + margin)).nonzero()[0]
            
            if lane_inds != []:
                # Extract line pixel positions   
                x = nzx[lane_inds]
                y = nzy[lane_inds]
            
            return (x, y)

    def lane_inds_from_histogram(self, birds_eye_binary, minpix=50):
        """
        Finds left and right lane pixels using maximum pixel density 
        histogram. Only useful if lines relatively straight up and down 
        or starting x of lane equals max density x (curves but beginning and end are
        at same x so max density captures beginning of lane)

        Return
        ------
        tuple of list representing indices of leftx, lefty, rightx, 
        righty ([lx], [ly], [rx], [ry]) or
        (None, None, None, None) if no indices found  

        """

        # Set the width of the windows +/- margin
        margin = self.margin #10
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = birds_eye_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # print("lane_inds_from_histogram nz len", len(nonzero))

        img_height, _ = birds_eye_binary.shape[:2]

        # maximum pixel density histogram used to find lane line
        histogram = np.sum(birds_eye_binary[img_height//2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # print("leftx_base, midpoint, rightx_base: ", leftx_base, midpoint, rightx_base)
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = img_height // 5
            
        # Set minimum number of pixels found to recenter window
        # minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img_height//nwindows)
            
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
            
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img_height - (window+1)*window_height
            win_y_high = img_height - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
                                
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # print("left, right lane inds length ", len(left_lane_inds), len(right_lane_inds))

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                # print("left x adjusted, leftx_current ", leftx_current)
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                # print("right x adjusted, rightx_current ", rightx_current)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            # print("left_lane_inds length after concat ", len(left_lane_inds))
        except ValueError:
            # Avoids an error if the above is not implemented fully
            # print("error in concatenating left inds")
            pass

        try:
            right_lane_inds = np.concatenate(right_lane_inds)
            # print("right_lane_inds length after concat ", len(right_lane_inds))
        except ValueError:
            # Avoids an error if the above is not implemented fully
            # print("error in concatenating right inds")
            pass

        leftx = lefty = rightx= righty = None

        if len(left_lane_inds) > 0:
            # Extract line pixel positions   
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            # print("leftx, y length ", len(leftx), len(lefty))

        else:
            print("left_lane_inds len == 0")

        if len(right_lane_inds) > 0:
            # Extract line pixel positions   
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # print("rightx, y length ", len(rightx), len(righty))

        else:
            print("right_lane_inds len == 0")

        return (leftx, lefty, rightx, righty)


    def find_lane_inds_from_polyfit(self, birds_eye_binary, lp, rp) -> tuple:
        """
        Use left and right polyfit to find lane indices in binary image.

        Return
        ------
        indices tuple (leftx, lefty, rightx, righty)

        """

        margin = self.margin
        
        # Identify the x and y positions of all nonzero pixels in the image
        nz = birds_eye_binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])
        
        leftx  = lefty = rightx = righty = []
        
        h, w = birds_eye_binary.shape[:2]

        # Create empty lists to receive left and right lane pixel indices
        if lp is None:
            left_lane_inds = []
        else:
            left_lane_inds = ((nzx >= lp[0] * nzy**2 + lp[1] * nzy + lp[2] - margin) & 
                    (nzx <= lp[0] * nzy**2 + lp[1] * nzy + lp[2] + margin)).nonzero()[0]
        if rp is None:
            right_lane_inds = []    
        else:
            right_lane_inds = ((nzx >= rp[0] * nzy**2 + rp[1] * nzy + rp[2] - margin) & 
                    (nzx <= rp[0] * nzy**2 + rp[1] * nzy + rp[2] + margin)).nonzero()[0]

        # print("left lane inds sum, right lane inds sum, lp, rp", sum(left_lane_inds), sum(right_lane_inds), lp, rp)
        
        if left_lane_inds != []:
            # Extract line pixel positions   
            leftx = nzx[left_lane_inds]
            lefty = nzy[left_lane_inds]

        if right_lane_inds != []:
            # Extract line pixel positions   
            rightx = nzx[right_lane_inds]
            righty = nzy[right_lane_inds]
        # print("leftx, lefty, rightx, righty ", leftx, lefty, rightx, righty)

        # set to None to keep checks consistent 
        if len(leftx) == 0:
            leftx = None
        if len(lefty) == 0:
            lefty = None
        if len(rightx) == 0:
            rightx = None
        if len(righty) == 0:
            righty = None

        return (leftx, lefty, rightx, righty)

    def find_lane_inds_from_histogram(self, birds_eye_binary=None):
        """
        Finds left and right lane lines in a binary image by using 
        maximum pixel density histogram, which is then polyfit .

        Return
        ------
        tuple of lists of left and right nonzero pixel coords (lx, ly, rx, ry) or
        (None, None, None, None) if no nonzero pixels

        """

        # Set the width of the windows +/- margin
        margin = self.margin
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = birds_eye_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        img_height, img_width = birds_eye_binary.shape[:2]

        # maximum pixel density histogram used to find lane line
        histogram = np.sum(birds_eye_binary[img_height//2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # print("leftx_base, midpoint, rightx_base: ", leftx_base, midpoint, rightx_base)
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
            
        # Set minimum number of pixels found to recenter window
        minpix = 100

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img_height//nwindows)
            
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
            
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = birds_eye_binary.shape[0] - (window+1)*window_height
            win_y_high = birds_eye_binary.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
                                
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
                
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            print("error in concatenating left inds")
            pass

        try:
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            print("error in concatenating right inds")
            pass

        leftx = lefty = rightx= righty = []

        if left_lane_inds != []:
            # Extract line pixel positions   
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            # store pixel count of lane line for reference in adjust_lane_corner_pts
            self.left_lane_pixel_count = len(leftx)

        if right_lane_inds != []:
            # Extract line pixel positions   
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # store pixel count of lane line for reference in adjust_lane_corner_pts
            self.right_lane_pixel_count = len(rightx)
        
        # print("leftx, lefty, rightx, righty ", leftx, lefty, rightx, righty)
        h, w = birds_eye_binary.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        canvas[[lefty, leftx]] = np.array([0, 255, 0], dtype=np.float32)        
        canvas[[righty, rightx]] = np.array([0, 0, 255], dtype=np.float32)
        
        # set to None to keep checks consistent 
        if len(leftx) == 0:
            leftx = None
        if len(lefty) == 0:
            lefty = None
        if len(rightx) == 0:
            rightx = None
        if len(righty) == 0:
            righty = None

        return (leftx, lefty, rightx, righty)


    def lane_inds_from_unwarped_binary(self, unwarped_binary):
        """
        Finds left and right lane lines in a binary image by using 
        finding pixels on left and right half .

        Return
        ------
        tuple of lists of left and right nonzero pixel coords (lx, ly, rx, ry) or
        (None, None, None, None) if no nonzero pixels

        """

        # Set the width of the windows +/- margin
        margin = self.margin
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = unwarped_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        img_height, img_width = unwarped_binary.shape[:2]

        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(img_width//2)

        # Create empty lists to receive left and right lane pixel indices
        left_inds = (nonzerox < midpoint).nonzero()[0]
        right_inds = (nonzerox > midpoint).nonzero()[0]
            

        leftx = lefty = rightx= righty = []

        if left_inds != []:
            # Extract line pixel positions   
            leftx = nonzerox[left_inds]
            lefty = nonzeroy[left_inds]
            lfit = self.fitpoly(leftx, lefty)
            leftx, lefty = self.fit2inds(unwarped_binary, lfit)
            # store pixel count of lane line for reference in adjust_lane_corner_pts
            self.left_lane_pixel_count = len(leftx)

        if right_inds != []:
            # Extract line pixel positions   
            rightx = nonzerox[right_inds]
            righty = nonzeroy[right_inds]
            rfit = self.fitpoly(rightx, righty)
            rightx, righty = self.fit2inds(unwarped_binary, rfit)
            # store pixel count of lane line for reference in adjust_lane_corner_pts
            self.right_lane_pixel_count = len(rightx)
        
        # set to None to keep checks consistent 
        if len(leftx) == 0:
            leftx = None
        if len(lefty) == 0:
            lefty = None
        if len(rightx) == 0:
            rightx = None
        if len(righty) == 0:
            righty = None

        return (leftx, lefty, rightx, righty)

    def overlap_count(self, fit, nonzero_xys):
        """
        Calculates the number of nonzero pixels that overlap the polyfit line. 

        Returns
        -------
        An integer value that represents the number of nonzero pixels that overlaps 
        the fit line.  The more pixels that overlap the fit, the more likely the fit
        represents a lane line.
        """


        # y at horizon and car (upper and lower limit to detect lane inds)
        y_lo = self.roi_params["bottom"]
        y_hi = self.roi_params["top"]

        # generate y values where lane is expected to appear on line 
        ys = np.int_(np.linspace(y_lo, y_hi-1, y_hi - y_lo))
    
        if len(fit) == 2:
            xs = fit[0] * ys + fit[1]
        elif len(fit) == 3:
            xs = fit[0] * ys**2 + fit[1] * ys + fit[2]

        fit_xys = np.array(np.transpose(np.vstack(np.int32([xs, ys]))))
        
        xs = nonzero_xys[1]
        ys = nonzero_xys[0]
        nonzero_xys = np.array(np.transpose(np.vstack(np.int32([xs, ys]))))
        
        overlap_count = sum((nonzero_xys[:, None] == fit_xys).all(-1).any(-1))

        return overlap_count

    def lines_to_overlaps_and_lines(self, lane_optimized_img, left_lines, right_lines): 
        """ 
        Accepts left and right lines tuple (slope, x1, y1, x2 , y2) 
        and fits each line to generate 1d polyfit as well as calculating each 
        polyfit's overlap to the image's nonzero pixels.

        Return:
        -------
        Two list of tuples (for left and right). Each tuple is of form (int, List)
        representing (overlap_count, fit)   
        Returns (None, None) if both lines not found   
        """

        left_overlap_fits = []
        right_overlap_fits = []

        # convert image to binary  
        binary = self.image_to_binary(lane_optimized_img)
        # nonzero pixels
        nz = binary.nonzero()

        # find lane pixels by histogram method (unwarped driver's view)
        lx, ly, rx, ry = self.lane_inds_from_unwarped_binary(binary)

        dim = 1
        # fit poly and calculate overlap from nonzero pixels independent of houghline process
        # and add to list (yields more accurate lane lines at times) 
        if len(lx) > 0:
            l_fit = self.fitpoly(lx, ly, dim)
            left_overlap_count = self.overlap_count(l_fit, nz)
            left_overlap_fits.append((left_overlap_count, l_fit))

        if len(rx) > 0:
            r_fit = self.fitpoly(rx, ry, dim)
            right_overlap_count = self.overlap_count(r_fit, nz)
            right_overlap_fits.append((right_overlap_count, r_fit))
        

        if left_lines is not None:
            for line in left_lines:
                # fit from line, count overlap pixels, and add to list
                m, b = self.slope_intercept(line)
                fit = np.array([m, b])
                overlap_count = self.overlap_count(fit, nz)
                left_overlap_fits.append((overlap_count, fit))

        if right_lines is not None:
            for line in right_lines:
                # fit from line, count overlap pixels, and add to list
                m, b = self.slope_intercept(line)
                fit = np.array([m, b])
                overlap_count = self.overlap_count(fit, nz)
                right_overlap_fits.append((overlap_count, fit))

        # sort in place, by first element, from most overlap to least
        left_overlap_fits.sort(key= lambda tup: tup[0],  reverse=True)
        right_overlap_fits.sort(key= lambda tup: tup[0],  reverse=True)

        # convert empty list to None for consistency in checking for None
        if len(left_overlap_fits) == 0:
            left_overlap_fits = None

        if len(right_overlap_fits) == 0:
            right_overlap_fits = None

        return (left_overlap_fits, right_overlap_fits)

    def lines_to_overlaps_and_fits(self, lane_optimized_img, left_lines, right_lines): 
        """ 
        Accepts left and right lines tuple (slope, x1, y1, x2 , y2) 
        and fits each line to generate 2d polyfit as well as calculating each 
        polyfit's overlap to the image's nonzero pixels.

        Return:
        -------
        Two list of tuples (for left and right). Each tuple is of form (int, List)
        representing (overlap_count, fit)   
        Returns (None, None) if both lines not found   
        """

        left_overlap_fits = []
        right_overlap_fits = []

        # convert image to binary  
        binary = self.image_to_binary(lane_optimized_img)
        # nonzero pixels
        nz = binary.nonzero()

        # find lane pixels by histogram method (unwarped driver's view)
        lx, ly, rx, ry = self.lane_inds_from_unwarped_binary(binary)

        dim = 1
        # fit poly, calculate overlap, and add to list 
        if len(lx) > 0:
            l_fit = self.fitpoly(lx, ly, dim)
            left_overlap_count = self.overlap_count(l_fit, nz)
            left_overlap_fits.append((left_overlap_count, l_fit))

        if len(rx) > 0:
            r_fit = self.fitpoly(rx, ry, dim)
            right_overlap_count = self.overlap_count(r_fit, nz)
            right_overlap_fits.append((right_overlap_count, r_fit))
        

        if left_lines is not None:
            # if left lines found, find overlap in binary image using linear equation 
            # else look for lane line using histogram method         
            for line in left_lines:
                # get indices of pixels that are on line +/- margin
                xs, ys = self.lane_inds_from_line(binary, line)
                if len(xs) > 0:
                    # fit indices, count overlap pixels, and add to list
                    fit = self.fitpoly(xs, ys, dim)
                    overlap_count = self.overlap_count(fit, nz)
                    left_overlap_fits.append((overlap_count, fit))


        if right_lines is not None:
            for line in right_lines:
                # get indices of pixels that are on line +/- margin
                xs, ys = self.lane_inds_from_line(binary, line)
                if len(xs) > 0:
                    # fit indices, count overlap pixels, and add to list
                    fit = self.fitpoly(xs, ys, dim)
                    overlap_count = self.overlap_count(fit, nz)
                    right_overlap_fits.append((overlap_count, fit))

        # sort in place, by first element, from most overlap to least
        left_overlap_fits.sort(key= lambda tup: tup[0],  reverse=True)
        right_overlap_fits.sort(key= lambda tup: tup[0],  reverse=True)

        # convert empty list to None for consistency in checking for None
        if len(left_overlap_fits) == 0:
            left_overlap_fits = None

        if len(right_overlap_fits) == 0:
            right_overlap_fits = None

        return (left_overlap_fits, right_overlap_fits)

    def slope_from_pts(self, lt, lb):
        x1, y1 = lt
        x2, y2 = lb 
        return (y2 - y1) / (x2 - x1)

    def score_diff(self, val, expected_val):
        diff = val - expected_val 
        return abs(diff / mean([val, expected_val]))
    
    def overlap_score(self, left_overlap, left_overlap_hi, right_overlap, right_overlap_hi):    
        # add overlap score 
        score = 0
        score += abs((left_overlap_hi - left_overlap) / left_overlap_hi)
        score += abs((right_overlap_hi - right_overlap) + right_overlap_hi)
        return score

    def cornerpts_score(self, cornerpts):
        """
        compares passed in cornerpts to average cornerpts and returns a score
        between 0 and 1 with 1 being exact match and nears 0 the further away
        cornerpts are away from average
        """
             
        if self.lanes.best_cornerpts is None:
            return 0 

        score = 0
        lt, rt, rb, lb = cornerpts
        bestlt, bestrt, bestrb, bestlb = self.lanes.best_cornerpts
        # add score for lb and rb x values 
        score += self.score_diff(lb[0], bestlb[0])
        score += self.score_diff(rb[0], bestrb[0])

        # add score for lt and rt x values
        score += self.score_diff(lt[0], bestlt[0])
        score += self.score_diff(rt[0], bestrt[0])
        # add score for same side slope 
        l_slope = self.slope_from_pts(lt, lb)
        best_l_slope = self.slope_from_pts(bestlt, bestlb)
        score += self.score_diff(l_slope, best_l_slope)

        r_slope = self.slope_from_pts(rt, rb)
        best_r_slope = self.slope_from_pts(bestrt, bestrb)
        score += self.score_diff(r_slope, best_r_slope)
        
        # add score for opposite side slope comparison 
        score += self.score_diff(best_l_slope, best_r_slope)

        return score

    def lanefits_score(self, lfit, rfit):
        """
        compares passed in cornerpts to average cornerpts and returns a score
        between 0 and 1 with 1 being exact match and nears 0 the further away
        cornerpts are away from average
        """
        y_car = self.lanes.y_car
        score = 0

        # add score for same side fit first term (curve) 
        best_lfit = self.lanes.left_lane.best_fit
        best_rfit = self.lanes.right_lane.best_fit

        score += self.score_diff(lfit[0], best_lfit[0])
        score += self.score_diff(rfit[0], best_rfit[0])

        # add score of left and right x evaluated at y of car
        lx = self.xfit(lfit, y_car)
        best_lx = self.xfit(best_lfit, y_car)
        score += self.score_diff(lx, best_lx)

        rx = self.xfit(rfit, y_car)
        best_rx = self.xfit(best_rfit, y_car)
        score += self.score_diff(rx, best_rx)

        # add score for opposite side fit first term comparison 
        score += self.score_diff(lfit[0], rfit[0])

        return score

    def most_likely_laneline_pair(self, overlap_left_lines, overlap_right_lines):
        """
        Compares lines categorized as potential left and right lane lines 
        (list of tuples [(overlap, [fit]), ...])  

        Return
        ------
        tuple (left_line_fit, right_line_fit), where fit is result of cv2.polyfit()   
        returns (None, None) if no fits passed in
        """
        # compare left and right lines that select pair that most likely 
        # represents lane lines.
        likely_left_line = None 
        likely_right_line = None 

        if overlap_left_lines is None and overlap_right_lines is None:
            # left and right fits not found, returns (None, None)
            return  (likely_left_line, likely_right_line) 
        elif overlap_left_lines is None:
            # left fits not found, returns (None, likely_right_line)
            likely_right_line = max(overlap_right_lines)[1]
            return  (likely_left_line, likely_right_line)             
        elif overlap_right_lines is None:
            # right fits not found, returns (likely_left_fit, None)
            likely_left_line = max(overlap_left_lines)[1]
            return  (likely_left_line, likely_right_line)

        # calculate probabilities by adding a series of proportions 
        # that represent fractional differences from the optimum:
        # current line overlap / highest overlap (each side calc separately)
        # current line's absolute curvature / other line's abs curvature 
        # current line's absolute curvature / same side average curvature
        # current line's x / same side x average driver view
        

        # values used as divisor
        left_overlap_hi = max([overlap for overlap, _ in overlap_left_lines])

        right_overlap_hi = max([overlap for overlap, _ in overlap_right_lines])

        # dictionary of score values indexed by indices of left and right fits
        # separated by "-" lower score is better as lane lines should not deviate 
        # much from average and each other from frame to frame
        inds_score = defaultdict(lambda: None)
        # both left and right overlap fits not None, score all unique pairing
        for i, left in enumerate(overlap_left_lines):
            for j, right in enumerate(overlap_right_lines):               
                # each comparison adds at most 1 to score for perfect match
                # absolute value of scores taken in calculating differences 
                
                # create key (combines left and right indices with dash)
                inds = str(i) + "-" + str(j)

                # assign overlap, fit variables 
                left_overlap = left[0]
                left_line = left[1]

                right_overlap = right[0]
                right_line = right[1]

                # overlap score
                score = self.overlap_score(left_overlap, left_overlap_hi, 
                            right_overlap, right_overlap_hi)  # * 1.5 #  multiplier adds weight over cornerpts_score 

                # cornerpoint score
                cornerpts = self.cornerpoints_from_lines(left_line, right_line)
                score += self.cornerpts_score(cornerpts)

                # store score value in dictionary with left and right indices as key                
                inds_score[inds] = score  

        # select key with lowest score  
        most_likely_lane_pair = min(inds_score, key=inds_score.get)
        # split keys by dash
        lane_pair = most_likely_lane_pair.split('-')
        # index of left fit 
        left_lane = int(lane_pair[0])
        # index of right fit
        right_lane = int(lane_pair[1])
        # use index to get fit
        likely_left_line = overlap_left_lines[left_lane][1]
        likely_right_line = overlap_right_lines[right_lane][1]

        # print("likely left line", likely_left_line)
        # print("likely right line", likely_right_line)
        # self.draw_poly(self.img, likely_left_line, "likely left fit poly")        
        # self.draw_poly(self.img, likely_right_line, "likely right fit poly")        
        return (left_overlap_hi, right_overlap_hi, likely_left_line, likely_right_line)

    def most_likely_lanefit_pair(self, overlap_left_fits, overlap_right_fits):
        """
        Compares lines categorized as potential left and right lane lines 
        (list of tuples [(overlap, [fit]), ...])  

        Return
        ------
        tuple (left_line_fit, right_line_fit), where fit is result of cv2.polyfit()   
        returns (None, None) if no fits passed in
        """
        # compare left and right lines that select pair that most likely 
        # represents lane lines.
        likely_left_fit = None 
        likely_right_fit = None 

        if overlap_left_fits is None and overlap_right_fits is None:
            # left and right fits not found, returns (None, None)
            return  (likely_left_fit, likely_right_fit) 
        elif overlap_left_fits is None:
            # left fits not found, returns (None, likely_right_line)
            likely_right_fit = max(overlap_right_fits)[1]
            return  (likely_left_fit, likely_right_fit)             
        elif overlap_right_fits is None:
            # right fits not found, returns (likely_left_fit, None)
            likely_left_fit = max(overlap_left_fits)[1]
            return  (likely_left_fit, likely_right_fit)

        
        # values used as divisor
        left_overlap_hi = max([overlap for overlap, _ in overlap_left_fits])

        right_overlap_hi = max([overlap for overlap, _ in overlap_right_fits])

        # dictionary of score values indexed by indices of left and right fits
        # separated by "-" lower score is better as lane lines should not deviate 
        # much from average and each other from frame to frame
        inds_score = defaultdict(lambda: None)
        # both left and right overlap fits not None, score all unique pairing
        for i, left in enumerate(overlap_left_fits):
            for j, right in enumerate(overlap_right_fits):
                # square all calculations to make all differences positive                
                # each comparison adds at most 1 to score for perfect match
                
                # create key (combines left and right indices with dash)
                inds = str(i) + "-" + str(j)

                # assign overlap, fit variables 
                left_overlap = left[0] 
                left_fit = left[1]
                right_overlap = right[0]
                right_fit = right[1]

                # print("l fit, r fit ", left_fit, right_fit)

                # add overlap score 
                score = self.overlap_score(left_overlap, left_overlap_hi, 
                            right_overlap, right_overlap_hi)

                cornerpts = self.cornerpoints_from_fits(left_fit, right_fit)
                score += self.cornerpts_score(cornerpts)


                # store score value in dictionary with left and right indices as key                
                inds_score[inds] = score  

        # select key with lowest score  
        most_likely_lane_pair = min(inds_score, key=inds_score.get)
        # split keys by dash
        lane_pair = most_likely_lane_pair.split('-')
        # index of left fit 
        left_lane = int(lane_pair[0])
        # index of right fit
        right_lane = int(lane_pair[1])
        # use index to get fit
        likely_left_fit = overlap_left_fits[left_lane][1]
        likely_right_fit = overlap_right_fits[right_lane][1]
        # print("likely overlap left fit", overlap_left_fits[left_lane])
        # print("likely overlap right fit", overlap_right_fits[right_lane])
        # self.draw_poly(self.img, likely_left_fit, "likely left fit poly")        
        # self.draw_poly(self.img, likely_right_fit, "likely right fit poly")        
        return (left_overlap_hi, right_overlap_hi, likely_left_fit, likely_right_fit)

    def is_left_lane_candidate(self, line):

        if line is None:
            return False

        center = self.roi_params["hor_center"]
        h = self.roi_params["img_height"]
        m = y = x1 = y1 = x2 = 0

        if line is None:
            return False 
        elif len(line) == 5:
            m,x1,y1,x2,y2 = line
        elif len(line) == 4:
            x1,y1,x2,y2 = line
            if x2-x1 == 0:
                return False 
            else:
                m = (y2-y1) / (x2-x1) 
        else:
            return False 

        b = y1 - m*x1
        x = 0
        y = m * x + b

        # qualifies as left lane if has negative slope, is left of center, 
        # and visually crosses bottom of image at x=0         
        if  (m < 0) and (min(x1, x2) < center) and (y >= h):
            return True
        else:
            return False 

    def is_right_lane_candidate(self, line):

        if line is None:
            return False

        center = self.roi_params["hor_center"]
        h = self.roi_params["img_height"]
        m = y = x1 = y1 = x2 = 0

        if line is None:
            return False 
        elif len(line) == 5:
            m,x1,y1,x2,y2 = line
        elif len(line) == 4:
            x1,y1,x2,y2 = line
            if x2-x1 == 0:
                return False 
            else:
                m = (y2-y1) / (x2-x1) 
        else:
            return False 

        b = y1 - m*x1
        x = self.roi_params["img_width"]
        y = m * x + b

        # qualifies as right lane if has positive slope, is right of center, 
        # and visually crosses bottom of image at x=width_img-1 (zero index)         
        if  (m > 0) and (min(x1, x2) > center) and (y >= h):
            return True
        else:
            return False 

    def categorize_lines_left_right(self, lines): 
        """
        Accepts list of lines (each line is tuple (x1,y1,x2,y2)) separates them 
        into left and right lines using slope and position heuristic.

        Return
        ------
        List of left and right lines. Each line is a tuple (slope, x1, y1, x2, y2) 
        If lines is None, returns tuple (None, None)
        """
        if lines is None:
            return (None, None)
        # # Categorize lines(houghlines) as left or right lane    
        # h, w = self.img.shape[:2] 
        # center = w / 2 # horizontal center of image
        left_lane_lines = [] 
        right_lane_lines = []
        # print("lines count and coords: ", len(lines), lines)
        for line in lines:
            for x1,y1,x2,y2 in line:
                # skip if vertical line 
                if x2-x1 == 0:
                    continue

                slope = (y2-y1)/(x2-x1)
                l = (slope, x1, y1, x2, y2)

                if self.is_left_lane_candidate(l):
                    left_lane_lines.append(l)

                elif self.is_right_lane_candidate(l):
                    right_lane_lines.append(l)

        return (left_lane_lines, right_lane_lines)

    def img2gray(self, img):

        # uses cv2 to convert bgr color image to gray
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
              
    def find_lines(self, img=None):
        """ 
        Finds lines in image using cv2.houghlinesp()

        Return
        ------
        Results of cv2.HoughLinesP on binary in roi.  Each line has form (x1,y1,x2,y2).
        If no lines found, returns None.
        """
        img = self.img if img is None else img
        # 1) convert image to grayscale
        # self.show_image("findline img", img)
        gray_img = self.img2gray(img)
        self.save_image("grayscale", gray_img)        
        # self.historian.record(gray_img, "pipeline_grayscale.jpg")
        # self.show_image("findline gray", gray_img)

        # 2. Blur image to remove noise
        kernel_size = 3
        blurred_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
        self.save_image("blurred", blurred_img)        
        # self.historian.record(blurred_img, "pipeline_blurred.jpg")
        # self.show_image("findline blur", blurred_img)
        # 3. Do Canny transform to get edges

        lo_thresh, hi_thresh = (10, 255)
        canny_edges_img = cv2.Canny(blurred_img, lo_thresh, hi_thresh)
        self.save_image("canny", canny_edges_img)        
        # cv2.imwrite(os.path.join(save_directory, "pipeline_canny_edges.jpg"), canny_edges_img)
        # self.historian.record(canny_edges_img, "pipeline_canny_edges.jpg")
        # self.show_image("findline canny", canny_edges_img)
        # # 4. Crop image to region of interest

        roi_vertices = self.default_roi_vertices()
        roi_img = self.region_of_interest(canny_edges_img, 
                                    vertices=roi_vertices)
        self.save_image("roi", roi_img)        
        # self.historian.record(roi_img, "pipeline_roi.jpg")
        # self.show_image("findline roi", roi_img)

      
        # 5. look for lines in image 
        houghlines = cv2.HoughLinesP( roi_img, rho=1, theta=np.pi/180,
                            threshold=2, minLineLength=5, maxLineGap=10 )
        # self.historian.record(self.visualize_houghlines(houghlines), 
                                # "pipeline_houghlines.jpg")    
        houghlines_img = self.visualize_houghlines(houghlines)
        self.save_image("houghlines", houghlines_img)                
        return houghlines # returns None if no lines found
    
    def src_pts_to_lines(self, src):
        lt, rt, rb, lb = src

        x1, y1 = lt 
        x2, y2 = lb 
        m = y2-y1 / x2-x1 
        left_line = (m, x1, y1, x2, y2)

        x1, y1 = rt 
        x2, y2 = rb 
        m = y2-y1 / x2-x1 
        right_line = (m, x1, y1, x2, y2)

        return (left_line, right_line)

    def lane_center_offset(self, left_line, right_line):
        l_x = self.x_given_liney(left_line, self.roi_params["roi_bottom"])
        r_x = self.x_given_liney(right_line, self.roi_params["roi_bottom"])
        lane_center = l_x + (r_x - l_x)
        image_center = self.roi_params["img_width"] / 2
        return image_center - lane_center

    def paint_road(self, birds_eye_binary, l_poly, r_poly):
        """
        Paints binary road pixels demarcated by left and right polyfit lines

        Return
        ------
        Binary image with road painted.

        """
        
        # print("paint road l_poly, r_poly ", l_poly, r_poly)
        if l_poly is None or r_poly is None:
            return birds_eye_binary

        h, w = birds_eye_binary.shape[:2]
        # canvas = np.zeros([h,w], dtype=np.uint8)
        canvas = np.zeros([h,w,3], dtype=np.uint8)
        y = np.linspace(0, len(birds_eye_binary)-1, len(birds_eye_binary))
        leftx = l_poly[0] * y**2 + l_poly[1] * y + l_poly[2]
        rightx = r_poly[0] * y**2 + r_poly[1] * y + r_poly[2]

        left_pts = [[x, y] for x,y in zip(leftx,y)]
        right_pts = [[x, y] for x,y in zip(rightx,y)][::-1] # reverse

        pts = np.array(np.vstack((left_pts, right_pts)), dtype=np.int32) 

        painted_road = cv2.fillPoly(canvas, [pts], [0,255,0])
        
        return painted_road

    def reverse_birds_eye_transform(self, binary_img, orig_src, orig_dst):
        """
        Undo warp by calling original transform using original src and dst 
        points but swapping order  
        """
        return self.birds_eye_transform(binary_img, orig_dst, orig_src)
        
    def birds_eye_transform(self, binary, src, dst):

        if src is None or dst is None:
            print("Birds-eye transform failed, src and dst cannot be None.")
            return binary

        src = np.float32([pt for pt in src])
        dst = np.float32([pt for pt in dst])
        img_dim = binary.shape[:2][::-1] # reverse h,w -> w,h
        M = cv2.getPerspectiveTransform(src, dst)
        binary_warped = cv2.warpPerspective(binary, M, 
                        img_dim, flags=cv2.INTER_LINEAR)
        return binary_warped

    def fits_from_birds_eye(self, birds_eye_binary):
        """
        finds left and right fit fit from warped (bird's eye view) binary
        by looking for nonzero pixels using histogram method or, if 
        polyfit available, using polyfit.

        Return
        ------
        Tuple of fits (left_fit, right_fit) where fit is a 2d fit
        """
        left_fit = self.lanes.left_lane.best_fit
        right_fit = self.lanes.left_lane.best_fit
        if (left_fit is None or right_fit is None):
            lx, ly, rx, ry = self.lane_inds_from_histogram(birds_eye_binary)
            if lx is None or ly is None or rx is None or ry is None:
                print("None in lx, ly, rx, or ry histogram length ")

        else:
            lx, ly, rx, ry = self.find_lane_inds_from_polyfit(birds_eye_binary, 
                left_fit, right_fit)
            if lx is None or ly is None or rx is None or ry is None:
                print("None in lx, ly, rx, or ry polyfit length ")


        if lx is None:
            left_fit = None 
        else:
            left_fit = self.fitpoly(lx, ly)
        
        if rx is None:
            right_fit = None
        else:
            right_fit = self.fitpoly(rx, ry)

        return (left_fit, right_fit)

    def default_dst_pts(self):
        """
        Default destination coordinate points needed for birds eye transform.

        Return
        ------
        4 tuples of (x, y) coords (top-left, top-right, bottom-right, bottom-left)
        """

        dst_offset = self.roi_params["dst_offset"] 
        y_top = 0
        y_bottom = self.roi_params["img_height"] - 1
        x_left = 0 + dst_offset
        x_right = self.roi_params["img_width"] - dst_offset 
        
        lt = (x_left, y_top)
        rt = (x_right, y_top)
        rb = (x_right, y_bottom) 
        lb = (x_left, y_bottom) 
        # order of 4 corners: lt -> rt -> rb -> lb (keep consistent)        
        dst_pts = [lt, rt, rb, lb]
        
        return dst_pts

    def lane_confidence_score(self, lfit, rfit, lpx_ct, rpx_ct):
        # given left and right fits, left and right pixel count calculates 
        # a confidence score that fit represents left and right lane line

        # initialize score
        left_score = 0
        right_score = 0 
        
        # score for closeness to bestx
        bestx = self.lanes.left_lane.bestx
        if bestx is not None:
            x = self.xfit(lfit, self.get_y_car())
            left_score += self.score_diff(x, bestx)
        
        bestx = self.lanes.right_lane.bestx
        if bestx is not None:
            x = self.xfit(rfit, self.get_y_car())
            right_score += self.score_diff(x, bestx)
        
        # score for closeness to first term of best_fit
        best_fit = self.lanes.left_lane.best_fit
        if best_fit is not None:
            fit0 = lfit[0]
            left_score += self.score_diff(fit0, best_fit[0])
        
        best_fit = self.lanes.right_lane.best_fit
        if best_fit is not None:
            fit0 = rfit[0]
            right_score += self.score_diff(fit0, best_fit[0]) 

        # score for pixel count relative to total
        # not squared as is never negative and also to give more weight 
        best_px = (lpx_ct + rpx_ct) / 2
        left_score += self.score_diff(lpx_ct, best_px)
        right_score += self.score_diff(rpx_ct, best_px)
        # smaller score means less deviation from expected, thus 
        # higher confidence.  Easier to think of high score as 
        # higher confidence so take inverse.
        return (1/left_score, 1/right_score)

    def adjust_lane_corner_pts(self, cornerpts, binary_img):
        """
        uses pixels from the left and right lanes in birds eye view 
        to adjust cornerpoints in drivers view so that fits of left
        and right lanes are closer to being parallel
        """

        h, w = binary_img.shape[:2]

        # print("optimize cornerpts initial cornerpts: ", cornerpts)
        # if lanes not detected and any cornerpts not found, use image edge 
        # TODO: calc cornerpt if one visible lane curves to left or right 
        # side so that no-lane side visible road begins at visible lane
        # end at left or ride side 
        lt, rt, rb, lb = cornerpts
        if lt is None :
            lt = (0, 0)
        if lb is None:
            lb = (0, h)
        if rt is None:
            rt = (w, 0)
        if rb is None:
            rb = (w, h)

        # use cornerpts to warp image to birds eye view of road
        warped = self.birds_eye_transform(binary_img, cornerpts, self.default_dst_pts())

        # initialize left and right x,y lane pixels
        lx = ly = rx = ry = None

        # move top points down if it is above horizon
        horizon_pixel_count = np.sum(binary_img[lt[1]:lt[1]+1,:])
        # print("horizon pixel count, cornerpts ", horizon_pixel_count, cornerpts)
        if horizon_pixel_count > 50:

            x, y = lt 
            y += 1
            lt = (x,y)
            x, y = rt 
            y += 1
            rt = (x,y)
            adj_cornerpts = [lt, rt, rb, lb]
            stop_optimization = False
            # print("adjusted cornerpts", adj_cornerpts)
            return (stop_optimization, adj_cornerpts)


        if self.lanes.left_overlap_hi > self.lanes.right_overlap_hi:
            l_confidence = 1 
            r_confidence = 0
        else:
            l_confidence = 0
            r_confidence = 1

        # tolerance percentage (how close want comparisons to vary)
        tol = 0.05

        # assign passed in cornerpts for modification
        lt, rt, rb, lb = cornerpts        

        lane_width = self.get_birds_eye_lane_width()
        warped = self.birds_eye_transform(binary_img, cornerpts, self.default_dst_pts())

        # remove artifacts near top  by using buffer at 10% of height
        # buffer = int(h * .1)
        # warped = warped[buffer: h-buffer, :]
        # maximum pixel density histogram used to find lane line
        upper_histogram = np.sum(warped[:warped.shape[0]//2,:], axis=0)

        lower_histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(warped.shape[1]//2)

        upper_left_x = np.argmax(upper_histogram[:midpoint])
        upper_right_x = np.argmax(upper_histogram[midpoint:]) + midpoint

        lower_left_x = np.argmax(lower_histogram[:midpoint])
        lower_right_x = np.argmax(lower_histogram[midpoint:]) + midpoint



        if l_confidence > r_confidence:

            expected_right_x = lower_left_x + lane_width
            # print("change bottom right expected vs current: ", expected_right_x, upper_right_x)        
            if not np.isclose(expected_right_x, lower_right_x, tol):
                # using high confidence lane as guide, adjust bottom 
                # cornerpt until diff of left and right x is close to 
                # expected line width  
                x, y = rb
                if lower_right_x > expected_right_x:
                    x -= 1
                else:
                    x += 1
                rb = (x,y)
                cornerpts = [lt, rt, rb, lb]
                stop_optimization = False
                return (stop_optimization, cornerpts)
            

            expected_right_x = upper_left_x + lane_width
            # print("change top right expected vs current: ", expected_right_x, lower_right_x)        
            if not np.isclose(expected_right_x, upper_right_x, tol):
                # using high confidence lane as guide, adjust bottom 
                # cornerpt until diff of left and right x is close to 
                # expected line width  
                x, y = rt
                if upper_right_x > expected_right_x:
                    x += 1
                else:
                    x -= 1
                rt = (x,y)
                cornerpts = [lt, rt, rb, lb]
                stop_optimization = False
                return (stop_optimization, cornerpts)

        else:

            expected_x = lower_right_x - lane_width
            # print("change top left expected vs current: ", expected_x, lower_left_x)        
            if not np.isclose(expected_x, lower_left_x, tol):
                # using high confidence lane as guide, adjust bottom 
                # cornerpt until diff of left and right x is close to 
                # expected line width  
                x, y = lb
                if lower_left_x > expected_x:
                    x -= 1
                else:
                    x += 1
                lb = (x,y)
                cornerpts = [lt, rt, rb, lb]
                
                stop_optimization = False
                return (stop_optimization, cornerpts)


            expected_x = upper_right_x - lane_width
            # print("change bottom left expected vs current: ", expected_x, upper_left_x)        
            if not np.isclose(expected_x, upper_left_x, tol):
                # using high confidence lane as guide, adjust bottom 
                # cornerpt until diff of left and right x is close to 
                # expected line width  
                x, y = lt
                if upper_left_x > expected_x:
                    x -= 1
                else:
                    x += 1
                lt = (x,y)
                cornerpts = [lt, rt, rb, lb]
                stop_optimization = False
                return (stop_optimization, cornerpts)
            
        # both top and bottom pts close to expected position, return original cornerpts
        stop_optimization = True
        return (stop_optimization, cornerpts)


    def cornerpoints_from_lines(self, l_line, r_line):
        """
        given left and right line, returns tuple of 4 point vertices
        of (x, y) points (top-left, top-right, bottom-right, bottom-left)
        """

        y_car = self.roi_params["top"]
        y_horizon = self.roi_params["bottom"]
        
        # order of 4 corners: lt -> rt -> rb -> lb
        # calculate x-intercept of left line at y_tip and y_base 
        x_car, x_horizon = self.x_intercepts(l_line, y_car, y_horizon) 
        lt = (x_horizon, y_horizon)
        lb = (x_car, y_car) 

        # calculate x-intercept of right line at y_tip and y_base 
        x_car, x_horizon = self.x_intercepts(r_line, y_car, y_horizon) 
        rt = (x_horizon, y_horizon)
        rb = (x_car, y_car) 

        cornerpoints = [lt, rt, rb, lb]
        
        return cornerpoints

    def cornerpoints_from_fits(self, l_fit, r_fit):
        """
        given left and right polyfit, returns tuple of 4 point vertices
        in order of (top-left, top-right, bottom-right, bottom-left)
        """
        y_top = self.get_y_hor() 
        y_bottom = self.get_y_car()
        
        # order of 4 corners: lt -> rt -> rb -> lb

        # left lane pts 
        x_top = l_fit[0]*y_top**2 + l_fit[1]*y_top + l_fit[2] 
        x_bottom = l_fit[0]*y_bottom**2 + l_fit[1]*y_bottom + l_fit[2] 

        lt = (x_top, y_top)
        lb = (x_bottom, y_bottom) 

        # right lane pts 
        x_top = r_fit[0]*y_top**2 + r_fit[1]*y_top + r_fit[2] 
        x_bottom = r_fit[0]*y_bottom**2 + r_fit[1]*y_bottom + r_fit[2] 

        rt = (x_top, y_top)
        rb = (x_bottom, y_bottom) 
        
        return [lt, rt, rb, lb]

    def color_road(self, original_img, warped_img):
        """ 
        colors lane car's current lane    
        """
        # l_poly, r_poly = pipeline.set_polys_equal(l_poly, r_poly, default_width)
        painted_road = self.paint_road(warped_img, 
            self.lanes.left_lane.best_fit, self.lanes.right_lane.best_fit)           
        # pipeline.show_image("painted road", painted_road)
        # reverse birds eye transform 
        painted_road = self.reverse_birds_eye_transform(painted_road, self.src, self.dst)
        # self.show_image("painted road reverse birds eye", painted_road)
        
        final = cv2.addWeighted(original_img, 1, painted_road, 0.3, 0)

        return final

    def find_cornerpts(self, img):
        """
        Gets cornerpoints in unwarped (driver's view) image required for 
        perspective transofrm to bird's eye view.

        Return
        ------
        tuple of 4 (x,y) coordinates: (left-top, right-top, right-bottom, left-bottom)    
        """
        # l_x = l_y = r_x = r_y = None
        
        # use 2d fit to fine cornerpoints, 1d misaligns at tip of curved road

        dim = 1

        if self.lanes.current_cornerpts is None:
            # if no lines detected previously 

            # start by narrowing image to roi
            roi = self.region_of_interest(img)
            # self.show_image("roi", roi)

            # find all lines in roi (None return if no lines) 
            lines = self.find_lines(roi)
            # self.visualize_houghlines(lines)    

            # then categorize lines as left or right lane (None returned if no lines)
            left_lines, right_lines = self.categorize_lines_left_right(lines)

            if dim == 1:
                # then calculate overlap and 1d fit of each line (None returned if no lines)
                l_overlap_lines, r_overlap_lines = self.lines_to_overlaps_and_lines(img, 
                    left_lines, right_lines)
                l_overlap_hi, r_overlap_hi, l_line, r_line = self.most_likely_laneline_pair(l_overlap_lines, 
                    r_overlap_lines)
                self.lanes.set_overlap_hi(l_overlap_hi, r_overlap_hi)
                cornerpts = self.cornerpoints_from_lines(l_line, r_line)
                # self.lanes.update_cornerpts(cornerpts)
                return cornerpts            
            if dim ==2:    
                l_overlap_fits, r_overlap_fits = self.lines_to_overlaps_and_fits(img, 
                    left_lines, right_lines)
                # print("l and r overlap fits ", l_overlap_fits, r_overlap_fits)
                l_overlap_hi, r_overlap_hi, l_fit, r_fit = self.most_likely_lanefit_pair(l_overlap_fits, 
                    r_overlap_fits)
                # save overlap hi in lanes for cornerpts optimization
                self.lanes.set_overlap_hi(l_overlap_hi, r_overlap_hi)
                cornerpts = self.cornerpoints_from_fits(l_fit, r_fit)
                # self.lanes.update_cornerpts(cornerpts)
                return cornerpts

            # print("left, right overlap fits ", l_overlap_fits, r_overlap_fits)
            # returns (None, None) if no fit                      
        else:
            # use previous cornerpts    
            return self.lanes.best_cornerpts#current_cornerpts#best_cornerpts 

    def is_parallel(self, left_curve, right_curve):
        return np.isclose(left_curve, right_curve, 0.01)

    def get_fits(self, birds_eye_binary):
        l_fit = self.lanes.left_lane.best_fit
        r_fit = self.lanes.right_lane.best_fit
        if l_fit is None or r_fit is None:
            l_fit, r_fit = self.fits_from_birds_eye(birds_eye_binary)
        return (l_fit, r_fit)

    def adjust_corner_pts(self, cornerpts, lfit, rfit, cornerpts_adj_ct):
        """
        Adjusts cornerpts until x values evaluated at y=0 and y=img_height
        equal dst_offset and img_width-dst_offset. Since the 4 cornerpts 
        fall on the lane lines in driver's view, the warped transform 
        should translate those 4 cornerpts to the 4 default dst corner pts
        """
        # tolerance percentage (how close want comparisons to vary)
        tol = 0.02

        # assign passed in cornerpts for modification
        lt, rt, rb, lb = cornerpts        
        ltc, rtc, rbc, lbc = cornerpts_adj_ct

        expected_left_xfit = self.DST_OFFSET
        expected_right_xfit = self.roi_params["img_width"] - self.DST_OFFSET
        # print("expected l and r xfit: ", expected_left_xfit, expected_right_xfit)        

        y_top = 0
        y_bottom = self.roi_params["img_height"]
        ltop_xfit = self.xfit(lfit, y_top)
        lbottom_xfit = self.xfit(lfit, y_bottom)
        rtop_xfit = self.xfit(rfit, y_top)
        rbottom_xfit = self.xfit(rfit, y_bottom)
        # print("lt, lb, rt, rb xfit", ltop_xfit, lbottom_xfit, rtop_xfit, rbottom_xfit)

        if rbc > 0:
            if not np.isclose(expected_right_xfit, rbottom_xfit, tol):
                # adjust right-bottom pt    
                x, y = rb
                if rbottom_xfit > expected_right_xfit:
                    x -= 1
                else:
                    x += 1
                rb = (x,y)
                cornerpts = [lt, rt, rb, lb]
                stop_optimization = False
                rbc += 1 
                cornerpts_adj_ct = (ltc, rtc, rbc, lbc)
                return (stop_optimization, cornerpts, cornerpts_adj_ct)

        if rtc > 0:            
            if not np.isclose(expected_right_xfit, rtop_xfit, tol):
                # adjust right-top pt  
                x, y = rt
                if rtop_xfit > expected_right_xfit:
                    x += 1
                else:
                    x -= 1
                rt = (x,y)
                cornerpts = [lt, rt, rb, lb]
                stop_optimization = False
                rtc += 1 
                cornerpts_adj_ct = (ltc, rtc, rbc, lbc)
                return (stop_optimization, cornerpts, cornerpts_adj_ct)

        if lbc > 0:
            if not np.isclose(expected_left_xfit, lbottom_xfit, tol):
                # adjust left-bottom pt  
                x, y = lb
                if lbottom_xfit > expected_left_xfit:
                    x -= 1
                else:
                    x += 1
                lb = (x,y)
                cornerpts = [lt, rt, rb, lb]            
                stop_optimization = False
                lbc += 1 
                cornerpts_adj_ct = (ltc, rtc, rbc, lbc)
                return (stop_optimization, cornerpts, cornerpts_adj_ct)

        if ltc > 0:
            if not np.isclose(expected_left_xfit, ltop_xfit, tol):
                # adjust left-top pt   
                x, y = lt
                if ltop_xfit > expected_left_xfit:
                    x += 1
                else:
                    x -= 1
                lt = (x,y)
                cornerpts = [lt, rt, rb, lb]
                stop_optimization = False
                ltc += 1 
                cornerpts_adj_ct = (ltc, rtc, rbc, lbc)
                return (stop_optimization, cornerpts, cornerpts_adj_ct)


        # all pts close to expected position, return original cornerpts
        stop_optimization = True
        return (stop_optimization, cornerpts, cornerpts_adj_ct)

    def detected_lanes_reasonable(self, lfit, rfit):
        
        if self.lanes.left_lane.best_fit is None or self.lanes.right_lane.best_fit is None:
            # nothing to compare, assume lanes reasonable, perhaps check
            # if requires n fits before checking for reasonableness
            return True 

        # checks the difference between fit of current frame to best (average) fit
        left_diff = float(sum(lfit - self.lanes.left_lane.current_fit))
        right_diff = float(sum(rfit - self.lanes.right_lane.current_fit))

        if abs(left_diff) < 3 and abs(right_diff) < 3:
            return True
        else:
            return False

    def optimize_cornerpts(self, cornerpts, binary_img, cornerpts_adj_ct):
        """
        Accepts a tuple of cornerpts and optimizes it by changing it until 
        lines parallel , uses it to warp image, and checks if 
        """

        # optmizes corner points in driver view by changing corner points until
        # birds eye transform using corner pts yields relatively parallel lanes.  
        # NOTE: birds viewd params stored in lane obj  (drivers view in line obj)

        h = self.roi_params["img_height"]
        w = self.roi_params["img_width"]

        # print("optimize cornerpts initial cornerpts: ", cornerpts)
        # if lanes not detected and any cornerpts not found, use image edge 
        # TODO: calc cornerpt if one visible lane curves to left or right 
        # side so that no-lane side visible road begins at visible lane
        # end at left or ride side 
        lt, rt, rb, lb = cornerpts
        if lt is None :
            lt = (0, 0)
        if lb is None:
            lb = (0, h)
        if rt is None:
            rt = (w, 0)
        if rb is None:
            rb = (w, h)


        # use cornerpts to warp image to birds eye view of road
        warped = self.birds_eye_transform(binary_img, cornerpts, self.default_dst_pts())
        # self.show_image("optimizing cornerpts warped img", warped)
        # initialize left and right x,y lane pixels
        lx = ly = rx = ry = None

        if not self.lanes.left_lane.detected and not self.lanes.right_lane.detected:
            # no previous lanes detected: use histogram to find lane indices
            lx, ly, rx, ry = self.lane_inds_from_histogram(warped)           
            if lx is not None and rx is not None:
                l_fit = self.fitpoly(lx, ly)                
                r_fit = self.fitpoly(rx, ry)
                # check to stop optimization loop (first term equal ~ lanes parallel) 
                stop_optimization = self.is_parallel(l_fit[0], r_fit[0])
                

                if not stop_optimization :
                    # lanes not parallel yet, adjust cornerpoints 
                    stop_optimization, cornerpts, cornerpts_adj_ct = self.adjust_corner_pts(
                        cornerpts, l_fit, r_fit, cornerpts_adj_ct)

                return (stop_optimization, cornerpts, cornerpts_adj_ct, warped)

            # print("lane inds from histogram lx, ly, rx, ry: ", l_x, l_y, r_x, r_y)
            else:
                # 2 lanes not found,  stop optmization defaulted to true and 
                # set cornerpts to None to look for lane line pixels in next frame
                self.lanes.left_lane.detected = False
                self.lanes.right_lane.detected = False                
                self.lanes.current_cornerpts = None                
                stop_optimization = True
                # print("optimizing cornerpts no lanes detected initially or using histogram")
                return (stop_optimization, cornerpts, cornerpts_adj_ct, warped)
        elif self.lanes.left_lane.detected and self.lanes.right_lane.detected:
            # use previous fits to find current frame lane indices and calculate fit 
            # previous frame fit
            lfit, rfit = self.lanes.get_current_fits()
            # current frame lane indices
            lx, ly, rx, ry = self.fits2inds(warped, lfit, rfit)

            if lx is not None and rx is not None:
                # new fit for current frame 
                l_fit = self.fitpoly(lx, ly) 
                r_fit =  self.fitpoly(rx, ry) 

                # check if lanes parallel (fit first term close) 
                stop_optimization = self.is_parallel(l_fit[0], r_fit[0])

                if not stop_optimization:
                    # adjust cornerpoints until lanes parallel
                    stop_optimization, cornerpts, cornerpts_adj_ct = self.adjust_corner_pts(
                        cornerpts, l_fit, r_fit, cornerpts_adj_ct)
                    self.lanes.update_lanes(lx, ly, rx, ry)

                return (stop_optimization, cornerpts, cornerpts_adj_ct, warped)

            else:
                # stop optmization, True by default, if pixels on both lanes not found
                # set cornerpts to None to look for lane line pixels in next frame
                self.lanes.left_lane.detected = False
                self.lanes.right_lane.detected = False 
                self.lanes.current_cornerpts = None
                stop_optimization = True
                return (stop_optimization, cornerpts, cornerpts_adj_ct, warped)

        else:
            # stop optmization,
            # print("optimizing cornerpts stopped, 1 of 2 lanes not detected, cornerpts: ", cornerpts)

            if (self.lanes.n_lane_not_found) > self.max_iter_lanes_not_found: 
                # set cornerpts to None to look for lane line pixels in next frame
                self.lanes.n_lane_not_found = 0
            else:
                self.lanes.increment_n_lane_not_found()

            self.lanes.left_lane.detected = False
            self.lanes.right_lane.detected = False 
            self.lanes.current_cornerpts = None
            stop_optimization = True
            return (stop_optimization, cornerpts, cornerpts_adj_ct, warped)


    def constrain_cornerpts(self, cornerpts):
        """
        Constrains cornerpoints and ensures it is within image.

        Return
        ------
        List of 4 tuples that are (x,y) pts of 4 corners.
        """
        h = self.roi_params["img_height"]
        w = self.roi_params["img_width"]        
        within_img_pts = []
        for i, (x,y) in enumerate(cornerpts):
            if x < 0:
                x = 0
            if x > w - 1:
                x = w - 1
            if y < 0:
                y = 0
            if y > h - 1:
                y = h - 1
            within_img_pts.append((x,y))

        return within_img_pts