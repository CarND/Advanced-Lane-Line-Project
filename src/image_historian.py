import cv2 
import os 

class ImageHistorian:
    """
        Saves image and description at certain stages in a list
    
    Parameters
    ----------
        history: list
            List of tuple (image, description_string) that documents 
            image changes at various steps of image processing  

    Instance Method
    ---------------
        record(self, img, description)

    Static Method
    -------------
        save_image(img, description, path_name optional)

    """


    history = []

    def record(self, img=None, description=None):
        """ 
        Saves (appends) image and description in history list of instance.
        """
        if img is None or description is None:
            print("Image and description not saved either one is None")
        else:
            self.history.append((img, description))

    def view_history(self):
        for img, desc in self.history:
            cv2.imshow(desc, img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def clear_history(self):
        self.history = []
    
    @staticmethod
    def save_image(img, name, path_name=None):
        """
        Saves image to path by joining path and name. Default path '../output_images'

        """
        # save undistorted checkboard image for writeup
        if path_name is None:
            path_name = '../output_images'

        if not os.path.exists(path_name):
            os.mkdir(path_name)
        cv2.imwrite(os.path.join(path_name, name), img)
    
    