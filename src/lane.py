import numpy as np 

class Lane():
    # represents lane lines from bird's eye view - lanes should be parallel 
    # and is used to verify, not determine 4 cornerpts used to warp img 
    
    def __init__(self, n=10):
    
        # y value used to calculate curvature - update using set_y_eval() in lane_processor.py
        self.y_eval = None

        # degree for polyfit (2nd degree polynomial fit)
        self.degree = 2

        # was the line detected in the last iteration?
        self.detected = False  

        # # index at which to replace x_fitted value
        self.i = 0
    
        # x values of the last n fits of the line
        self.recent_xfitted = [None for _ in range(n)] 
    
        # average x values of the fitted line over the last n iterations
        self.bestx = None 

        # last n fits
        self.recent_fit = [np.array([False]) for _ in range(n)]
        
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 

        # distance in meters of vehicle center from the line
        self.line_base_pos = None 

        # x values for detected line pixels
        self.allx = None 
         
        # y values for detected line pixels
        self.ally = None 

