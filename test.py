import cv2 

target = __import__("pipeline_helper.py")
img = cv2.imread("/test_images/straight_lines1.jpg")
pipeline = target.PipelineHelper()
pipeline.set_img(img)
assert(img == pipeline.img)

