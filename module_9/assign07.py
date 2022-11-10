import numpy as np
import skimage.util as skut

class DL:
    def cropImg(self, img, width=299, height=299, offset=None):
        """Returns a cropped 2D array

        Args:
            img (np.ndarray): np 2D array
            width (int, optional): Width. Defaults to 299.
            height (int, optional): Height. Defaults to 299.
            offset ((int, int), optional): Offset coordinates. Defaults to None.
        """
        if not isinstance(img, np.ndarray):
            return Exception("Provided image is not a numpy array")

        if img.shape[0] < width:
            return Exception("Image width too small")
        if img.shape[1] < height:
            return Exception("Image height too small")

        if offset is None:
            x_mid, y_mid = img.shape[0]/2, img.shape[1]/2
            offset = (int(x_mid - width/2), int(y_mid - height/2))

        return skut.img_as_float(img[offset[0]: offset[0] + width, offset[1]: offset[1] + height, :])