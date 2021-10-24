import cv2 
import numpy as np
import glob 
import os
import pickle
from image_historian import ImageHistorian 

class Calibrator:
    """ 
    Class to calculate and store distortion coefficients to correct 
    camera distortion.

    If mtx and dist coefficients available from previous calibrations,
    initialize instance with their values. 

    Method correct_img_distortion() can be called on a passed in image 
    to output an image with distortions corrected. 

    Parameters
    ----------
    mtx: numpy array of float64
        Generated when calibrate() is called. Required to run 
        corect_img_distortion().
    dist: numpy array of float64
        Generated when calibrate() is called. Required to run 
        corect_img_distortion().

    Methods
    -------
    calibrate_with_images(): instance method
        Calibrates camera and stores variables needed in cv2.undistort
        to fix image distortion.

    calibrate_with_pickle(pickle_path): instane method
        Unpickles mtx and dist variable and updates it in instance.

    pickle_calibration_variables(pickle_path): instance method
        Pickles self.mtx and self.dist.

    correct_img_distortion(): instance method
        Applies cv2.undistort to image and returns undistorted image. 

    """ 

    mtx = None
    dist = None
    
    def __init__(self) -> None:
        pass

    @staticmethod    
    def pickle_calibration_coeff(mtx, dist, pickle_path='../dist_pickle.p'):
        dist_pickle = {
            "mtx": mtx,
            "dist": dist 
            }

        # save as pickle file 'dist_pickle.p'
        pickle.dump(dist_pickle, open(pickle_path, 'wb'))


    def calibration_coeffs(self, chessboard_images_path, pickle_path, imshow=False):    
        """
        Calibrates camera using image path to chessboard images. 

        Calibrates camera and stores mtx and dist coefficients as pickle. 

        Returns:
        Distortion coefficient tuple (mtx, dist)
        """    
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        img = None

        images = glob.glob(chessboard_images_path)
        
        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname, 1)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                if imshow:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                    cv2.imshow('img',img)
                    cv2.waitKey(500)

        if imshow:
            cv2.destroyAllWindows()

        img_size = (img.shape[1], img.shape[0]) # (width, height)

        # calibrate and save variables for distortion correction later
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        # save coeff in pickle
        self.pickle_calibration_coeff(mtx, dist, pickle_path)

        return (mtx, dist)

    # # @staticmethod
    def unpickle_calibration_coeffs(self, pickle_path='../dist_pickle.p'):
        try:
            with open(pickle_path, 'rb') as f: 
                dist_coef_pickle = pickle.load(f)
                mtx = dist_coef_pickle['mtx']
                dist = dist_coef_pickle['dist']
                return (mtx, dist)

        except IOError:
            print("pickle file ", pickle_path, " not found.")
            return (None, None)


    def calibrate_camera(self, calibration_images_path = '../camera_cal/*.jpg', 
                        pickle_path='../dist_pickle.p'):
        """
        Convenience method to either find and use pickled distortion 
        coefficients or calculate new ones using chess images. 

        Return
        ------
        (mtx, dist) coefficients 
        """

        # use pickled distortion coefficients if available
        mtx, dist = self.unpickle_calibration_coeffs(pickle_path)

        # calculate distortion coefficients from images 
        if mtx is not None or dist is not None:
            mtx, dist = self.calibration_coeffs(calibration_images_path, 
                pickle_path)

        # update 
        self.mtx = mtx 
        self.dist = dist 

        return (mtx, dist)


