import unittest
import numpy as np 

import sys
sys.path.insert(0, 'src')

from pipeline_helper import PipelineHelper

import os

print(os.getcwd())
# os.chdir("src")
# print(os.getcwd())

# import sys
# sys.path.append('../..')

SQUARE_HEIGHT = 350
SQUARE_WIDTH = 800

class TestPipelineMethods(unittest.TestCase):

    def setUp(self):
        
        # create image with upside down 'V' to mimic road 
        # that intersects at vertical midpoint of image
        sq = (SQUARE_HEIGHT, SQUARE_WIDTH)
        topleft = np.zeros(sq)
        topright = np.zeros(sq)
        bottomright = np.eye(sq[0], sq[1])
        bottomleft = np.array([n for n in reversed(bottomright)])
        blk = np.block([[topleft, topright], [bottomleft, bottomright]])
        self.img = np.array(blk)

        self.pipeline = PipelineHelper()
        self.pipeline.set_roi_params(self.img)

    def test_region_of_interest(self):
        # check that area outside of roi sums to 0
        pass 

    def test_find_lines(self):
        # check that image passed into find_lines
        # turn img into gray - check that image has only 1 channel
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find edges with canny - should be same or similar to original
        # lo_thresh, hi_thresh = (10, 255)
        # canny_edges_img = cv2.Canny(blurred_img, lo_thresh, hi_thresh)  
        # roi - areas outside roi should sum to 0
        # self.region_of_interest(canny_edges_img, vertices=self.roi_vertices(img))
        # houghlines - should find at least 2 lines with slope = 1 and -1     
        # houghlines = cv2.HoughLinesP( roi_img, rho=1, theta=np.pi/180, threshold=2, minLineLength=5, maxLineGap=10 )

        pass 

    def test_categorize_lines_left_right(self):
        pass 


    def test_birds_eye_lane_width(self):
        line_width = self.pipeline.get_birds_eye_lane_width()
        expected_line_width = self.img.shape[1] - 2*self.pipeline.DST_OFFSET
        msg = "line width ", line_width, " is expected to be ", expected_line_width
        self.assertAlmostEqual(line_width, expected_line_width, 1, msg)

    def test_find_cornerpts(self):
        line_width = self.pipeline.get_birds_eye_lane_width()
        expected_line_width = self.img.shape[1] - 2*self.pipeline.DST_OFFSET
        msg = "line width ", line_width, " is expected to be ", expected_line_width
        self.assertAlmostEqual(line_width, expected_line_width, 1, msg)

    def test_lane_inds_from_histogram(self):
        pass
        w = self.img.shape[1]
        nz = np.nonzero(self.img)
        nzx = nz[1]
        nzy = nz[0]

        inds = nzx < w//2
        lnzx = nzx[inds]
        lnzy = nzy[inds]
        
        inds = nzx >= w//2
        rnzx = nzx[inds]
        rnzy = nzy[inds]

        minpx = 1
        lx, ly, rx, ry = self.pipeline.lane_inds_from_histogram(self.img, minpx)  
        np.testing.assert_array_equal(lnzx, lx, "left x lane inds from histogram off")
        # self.assertListEqual(lnzx, lx, "left x lane inds from histogram off")
        # self.assertListEqual(lnzy, ly, "left y lane inds from histogram off")
        # self.assertListEqual(rnzx, rx, "right x lane inds from histogram off")
        # self.assertListEqual(rnzy, ry, "right y lane inds from histogram off")

if __name__ == '__main__':
    unittest.main()