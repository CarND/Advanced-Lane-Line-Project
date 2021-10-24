
from operator import le

from numpy.lib.npyio import mafromtxt 
# Import everything needed to edit/save/watch video clips

from numpy.polynomial.polyutils import PolyError

import numpy as np 

from lane import Lane 

class LaneProcessor:

    # keeps track of number of times lane not found
    n_lane_not_found = 0
    
    # number of previous values to store 
    n = 10 

    # current index for update 
    i = 0

    # curved lane line - bird view
    left_lane = None
    right_lane = None


    # most recent cornerpts
    current_cornerpts = None

    # last n cornerpoints (lt, rt, rb, lb) of lanes in driver's view
    n_cornerpts = [None for _ in range(n)] #
    
    # average of last n cornerpts
    best_cornerpts = None # 4 tuples (x, y) which are ave of cornerpts 

    # undistorted image (or original if no calibration)
    img = None
    img_height = None 
    img_width = None

    # highest number of overlap pixels for left or right side (used to prioritize 
    # which lane to use as guide in ambiguous situations - higher overlap indicates 
    # more confidence that line is true lane, though not always)
    left_overlap_hi = 0
    right_overlap_hi = 0

    # distance from ego center to ego lane center
    distance_from_center = None
    
    def __init__(self) -> None:

        self.left_lane = Lane(self.n)
        self.right_lane = Lane(self.n) 

    def increment_n_lane_not_found(self):
        self.n_lane_not_found += 1

    def set_img(self, img):
        self.img = img 

    def set_img_dim(self, img):
        self.img_height, self.img_width = img.shape[:2]

    def set_y_eval(self, y_horizon, y_car):
        # set y values for evaluating 4 cornerpts (driver's view)
        self.y_car = y_car
        self.y_hor = y_horizon
        # set y value for evaluating  curvature (bird's eye view)
        self.left_lane.y_eval = self.img_height

    def set_overlap_hi(self, left_overlap_hi, right_overlap_hi):
        self.left_overlap_hi = left_overlap_hi
        self.right_overlap_hi = right_overlap_hi

    def radius_of_curvature(self, A, B, y_eval):
        # calculate radius of curvature
        return ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)

    def update_curvature(self, lx, ly, rx, ry, img=None):
        '''
        @ parameters
        leftx = x coordinates of left points about left_fitx
        rightx = x coordinates of right points about right_fitx
        img_shape = size of an input image
        
        @ return
        left_curvature = left radius of curvature(m)
        right_curvature = right radius of curvature(m) 
        '''

        if img is None:
            img = self.img 

        # Define conversions in x and y from pixels space to meters
        # We set this by requirement of U.S. regulations for a road
        ym_per_mix = 15./img.shape[0]
        xm_per_mix = 3.7/800
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        # Reverse to match top-to-bottom in y for left and right line
        leftx = lx[::-1]
        rightx = rx[::-1]
                
        # Convert polynomials to x,y in real world space
        # Define y-value where we want radius of curvature
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ly*ym_per_mix, leftx*xm_per_mix, 2)
        right_fit_cr = np.polyfit(ry*ym_per_mix, rightx*xm_per_mix, 2)
        
        # Implement the calculation of R_curve(radius of curvature)
        left_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_mix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_mix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        self.left_lane.radius_of_curvature = left_curvature
        self.right_lane.radius_of_curvature = right_curvature

    def get_curvature(self):
        # return left and right curvature
        if (self.left_lane.radius_of_curvature is None and 
            self.right_lane.radius_of_curvature is None):
            return (0, 0)
        elif self.left_lane.radius_of_curvature is None:
            return (0, self.right_lane.radius_of_curvature)
        elif self.right_lane.radius_of_curvature is None:
            return (self.left_lane.radius_of_curvature, 0)
        else:
            return (self.left_lane.radius_of_curvature, self.right_lane.radius_of_curvature) 


    def update_line_base_pos(self, cornerpts):

        if self.img_width is None:
            print("Need to call set_img_dim() Lane_Processor instance before calculating center offset")
            return None
        
        _, _, rb, lb = cornerpts
        car_center = self.img_width // 2
        lx = lb[0]
        rx = rb[0]
        mx = 3.7 / (rx - lx) 

        # distance from car center to left line
        self.left_lane.line_base_pos = (car_center - lx) * mx
        # distance from car center to right line
        self.right_lane.line_base_pos = (rx - car_center) * mx

        self.distance_from_center = (rx-lx-car_center) * mx
        

        return self.left_lane.line_base_pos, self.right_lane.line_base_pos, self.distance_from_center 

    def get_car_center_offset(self):
        # assuming lane width is 3.7m, expect that left and right lane is (3.7/2)m from center
        expt_distance_lane_from_center = 3.7 / 2 
        self.left_lane.line_base_pos + self.right_lane.line_base_pos

        if (self.left_lane.line_base_pos is None and 
            self.right_lane.line_base_pos is None):
            return (0, 0)
        elif self.left_lane.line_base_pos is None:
            return (0, self.right_lane.line_base_pos)
        elif self.right_lane.line_base_pos is None:
            return (self.left_lane.line_base_pos, 0)
        else:
            return (self.left_lane.line_base_pos, self.right_lane.line_base_pos) 

    def get_distance_from_center(self):
        return self.distance_from_center

    def update_cornerpts(self, cornerpts):
        # current (changeable) cornerpts 
        self.current_cornerpts = cornerpts
        self.n_cornerpts[self.i] = self.current_cornerpts

        # update best_cornerpts 
        ltx = lty = rtx = rty = rbx = rby =  lbx = lby = 0
        c = 0
        for cornerpts in self.n_cornerpts:
            if cornerpts is not None:
                # add all values if not None
                lt, rt, rb, lb = cornerpts
                ltx += lt[0]
                lty += lt[1]
                rtx += rt[0]
                rty += rt[1]
                rbx += rb[0]
                rby += rb[1]
                lbx += lb[0]
                lby += lb[1]
                c += 1
        # divide by count of values to get average 
        # use max() with 1 to prevent division by 0                
        ltx /= max(c, 1) 
        lty /= max(c, 1)
        rtx /= max(c, 1)
        rty /= max(c, 1)
        rbx /= max(c, 1)
        rby /= max(c, 1)
        lbx /= max(c, 1)
        lby /= max(c, 1)

        self.best_cornerpts = ((ltx, lty), (rtx, rty), (rbx, rby), (lbx, lby))


    def update_lanes(self, lx, ly, rx, ry):

        # y used to evaluate x (equivalent to position of car @ edge) 
        y = self.img_height - 1

        if lx is not None and rx is not None and len(lx) > 0 and len(rx) > 0:

            # increment i, keep within range 0 and n-1
            self.i = (self.i + 1) % self.n

            # flag that both lanes detected
            self.left_lane.detected = True
            self.right_lane.detected = True        

            # calculate 2d polyfit 
            lfit = np.polyfit(ly, lx, deg=2)
            rfit = np.polyfit(ry, rx, deg=2)            

            if self.left_lane.best_fit is None:
                lfit_is_reasonable = True
            else:
                # check if 1st term of fit is close to average, as it should be
                lfit_is_reasonable = np.isclose(lfit[0], self.left_lane.best_fit[0], .1)
            
            if self.right_lane.best_fit is None:
                rfit_is_reasonable = True    
            else:        
                # check if 1st term of fit is close to average, as it should be
                rfit_is_reasonable = np.isclose(rfit[0], self.right_lane.best_fit[0], .1)

            # short-circuit above check
            lfit_is_reasonable = True 
            rfit_is_reasonable = True 

            lbest_fit = self.left_lane.best_fit
            if lbest_fit is None:
                lbest_fit = lfit
            rbest_fit = self.right_lane.best_fit
            if rbest_fit is None:
                rbest_fit = rfit

            # update with new fit if reasonably close or use previous fit 
            # update fit difference (delta) 
            if self.left_lane.current_fit is not None:
                if lfit_is_reasonable:
                    if self.left_lane.best_fit is None:
                        self.left_lane.diffs = lfit - self.left_lane.current_fit
                    else:    
                        self.left_lane.diffs = lfit - self.left_lane.best_fit
                else:
                    self.left_lane.diffs = lbest_fit - self.left_lane.current_fit                    
            if self.right_lane.current_fit is not None:
                if rfit_is_reasonable:
                    if self.right_lane.best_fit is None:
                        self.right_lane.diffs = rfit - self.right_lane.current_fit                
                    else:
                        self.right_lane.diffs = rfit - self.right_lane.best_fit
                else:
                    self.right_lane.diffs = rbest_fit - self.right_lane.current_fit

            # update current fit 
            if lfit_is_reasonable:
                self.left_lane.current_fit = lfit 
            else:
                self.left_lane.current_fit = lbest_fit  

            if rfit_is_reasonable:            
                self.right_lane.current_fit = rfit 
            else:
                self.right_lane.current_fit = rbest_fit 

            # update recent_fit (last n fits)
            if lfit_is_reasonable:
                self.left_lane.recent_fit[self.i] = lfit
            else:
                self.left_lane.recent_fit[self.i] = lbest_fit

            if rfit_is_reasonable:
                self.right_lane.recent_fit[self.i] = rfit
            else:
                self.right_lane.recent_fit[self.i] = rbest_fit

            # update best fit
            n = 0
            sum_fits = [0,0,0]   
            for fit in self.left_lane.recent_fit:
                # print("calc best_fit l fit ", fit)
                if len(fit)==3:
                    sum_fits += fit
                    n += 1
            self.left_lane.best_fit = sum_fits / n
            # print("best_fit left ", self.left_lane.best_fit)

            n = 0
            sum_fits = [0,0,0]   
            for fit in self.right_lane.recent_fit:
                # print("calc best_fit r fit ", fit)
                if len(fit)==3:
                    sum_fits += fit
                    n += 1
            self.right_lane.best_fit = sum_fits / n
            # print("best_fit right ", self.right_lane.best_fit)

            if lfit_is_reasonable:
                # update recent_xfitted
                lxfit = lfit[2] * y**2 + lfit[1] * y + lfit[2]
                self.left_lane.recent_xfitted[self.i] = lxfit
            else:
                lxfit = lbest_fit[2] * y**2 + lbest_fit[1] * y + lbest_fit[2]
                self.left_lane.recent_xfitted[self.i] = lxfit
                                
            if rfit_is_reasonable:
                rxfit = rfit[2] * y**2 + rfit[1] * y + rfit[2]
                self.right_lane.recent_xfitted[self.i] = rxfit
            else:
                rxfit = rbest_fit[2] * y**2 + rbest_fit[1] * y + rbest_fit[2]
                self.right_lane.recent_xfitted[self.i] = rxfit

            # update best_x 
            n = 0
            sum_xfits = 0    
            for xfit in self.left_lane.recent_xfitted:
                if xfit is not None:
                    sum_xfits += xfit
                    n += 1    
            if n > 0:
                self.left_lane.bestx = sum_xfits / n

            n = 0
            sum_xfits = 0    
            for xfit in self.right_lane.recent_xfitted:
                if xfit is not None:
                    sum_xfits += xfit
                    n += 1    
            if n > 0:
                self.right_lane.bestx = sum_xfits / n
        elif lx is not None and rx is not None and len(lx) == 0 and len(rx) == 0:
            # flag that both lanes detected
            self.left_lane.detected = False
            self.right_lane.detected = False
        elif lx is not None and rx is not None and len(lx) > 0 and len(rx) == 0:
            # flag that both lanes detected
            self.left_lane.detected = True
            self.right_lane.detected = False
        elif lx is not None and rx is not None and len(lx) == 0 and len(rx) > 0:
            # flag that both lanes detected
            self.left_lane.detected = False
            self.right_lane.detected = True


    def get_best_fits(self):
        return (self.left_lane.best_fit, self.right_lane.best_fit)

    def get_current_fits(self):
        return (self.left_lane.current_fit, self.right_lane.current_fit)

