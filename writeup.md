from IPython import display

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)


[image1]: ./output_images/undistorted_chessboard.jpg "Undistorted Chessboard"
[image2]: ./output_images/undistorted.jpg "Undistorted Road"
[image3]: ./output_images/lane_optimized.jpg "Lane Optimized"
[image4]: ./output_images/cornerpoints.png "Source Cornerpoints"
[image5]: ./output_images/birds_eye_binary.jpg "Birds Eye Binary With Dst Lines"
[image6]: ./output_images/annotated_birds_eye.jpg "Annotated Birds Eye"
[image7]: ./output_images/painted_road.jpg "Painted Road"
[video1]: ./test_videos_output/painted_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Calibration was Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to calibrate the camera can be found in calibrator.py.
The calibrate_camera() method in the Calibrator class looks up 
the calibration coefficients 'mtx' and 'dst' from a pickle file or calculates it using chessboard images if the pickle file is not found.  

With the camera calibration and distortion coefficients, each frame's image distortion was corrected using the undistort_image() method in the PipelineHelper class ('f_pipeline_helper.py') which  ultimately called the `cv2.undistort()` function and obtained this result: 

![Undistorted Chessboard][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Undistorted Road][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image using the optimize_lane_lines() method from the PipelineHelper class.  Here's an example of my output for this step.  

![Lane Optimized][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function `find_cornerpts()` from 
the PipelineHelper class to find the source `src` points which are required for the perspective transform.  There is an optimization step (`optimize_cornerpts()` in PipelineHelper) in which the cornerpts are tweaked so that the resulting warped image yields parallel lane lines.  The destination (`dst`) points are calculated using a function `default_dst_pts()` from the PipelineHelper class.  The perspective transform is accomplished by calling the `birds_eye_transform()` method of the PipelineHelper class and passing the laneline optimized binary image and the `src` and `dst` points. 

The `src` points were verified by plotting them on the image and visually inspecting them to check that they do in fact all fall on and frame the road lane lines. 

![Source Cornerpoints][image4]

Perspective transform was verified by applying the `birds_eye_transform()` on the image and overlaying vertical lines that go through the destination points.  A visual checking indicates that the lines appear parallel and fall on or near the destination point lines as seen below.

![Birds Eye Binary With Dst Lines][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels were identified using the method `lane_inds_from_histogram()` from the PipelineHelper class which uses a histogram method to identify lane-line pixels.  If the previous polyfit is available, the method `lane_inds_from_polyfit()` is used to find lane-line pixels that are within a margin to the left and right of the previous polyfit.  To fit a polynomial, the `fitpoly()` method is applied to each set of lane indices on the left and right.  The resulting polyfits of the lane pixels in the warped image is seen below.

![Annotated Birds Eye][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in the file f_lane_processor.py in the `update_curvature()` method of the LaneProcessor class whereas the position of the vehicle is calculated in the `update_line_base_pos()` method.  

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This step was implemented in lines # 119 through # 128 in my code in `adv_lane_line_project.ipynb`.  Much of the work is actually done in the `paint_road()` method of the PipelineHelper class.  Here is an example of my result on a test image:

![Painted Road][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/painted_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There were many challenges.  One challenge was identifying the appropriate threshold to optimize lane line pixel detection in the binary.  The next challenge was identifying the optimal region of interest points. Being able to solve these two challenges greaty simplified our task of correctly finding the lanes and painting the road.  

The current algorithm doesn't handle the situation in which there is only one lane detected on the road.  Also, if the road changes too abruptly, the algorithm will consider the detected lanes errant and revert to the average values so this implementation will fail in a very sharp turn.  In addition, roads that have been redone (lane markers moved) and exhibit two sets of lane lines may lead this implementation to track the wrong (old) lane.  

To improve the current implementation, it may be helpful to implement a check on the width of the 4 source cornerpoints so that the top two and bottom two points should stay relatively consistent.  Another improvement would be to change our implementation to handle situations where only one lane is detected.  In cases where we only see one line, the other line could be created based on the current lane but shifted by the expected width to where the undetected lane is expected to be. Lastly, once we are fairly confident of where our lane lines will be, we can reduce noise by narrowing our region of interest to two lines with a small margin on each side. 

As the current implementation doesn't work on the more challenging conditions, it would be more fruitful to train a machine learning model to recognize the lane and roads using LaneNet, Enet-Label, or one of many other lane detection models.

