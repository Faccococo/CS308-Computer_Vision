import numpy as np
from skimage.filters import scharr_h, scharr_v, sobel_h, sobel_v, gaussian
import cv2
# debug:
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """
    confidences, scales, orientations = None, None, None
    
    a = 0.06
    threshold = 0.005
    stride = 2
    sigma = 0.2
    rows = image.shape[0]
    cols = image.shape[1]
    xs = []
    ys = []
    print("R threshold: ", threshold, " Stride: ", stride, "Gaussian sigma: ", sigma, " Alpha: ", a)
    #get the gradients in the x and y directions using sobel filter
    # image = gaussian(image)
    USE_SOBEL = True 
    if USE_SOBEL:
        I_x = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
        I_y = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5)
    else:
        I_x = np.abs(cv2.Scharr(image, cv2.CV_32F, 1, 0))
        I_y = np.abs(cv2.Scharr(image, cv2.CV_32F, 0, 1))
    I_x = gaussian(I_x, sigma)
    I_y = gaussian(I_y, sigma)

    Ixx = I_x**2
    Ixy = I_y*I_x
    Iyy = I_y**2

    # find the sum squared difference (SSD)
    for y in range(0,rows-feature_width,stride):
        for x in range(0,cols-feature_width,stride):
            Sxx = np.sum(Ixx[y:y+feature_width+1, x:x+feature_width+1])
            Syy = np.sum(Iyy[y:y+feature_width+1, x:x+feature_width+1])
            Sxy = np.sum(Ixy[y:y+feature_width+1, x:x+feature_width+1])
            #Find determinant and trace, use to get corner response
            detH = (Sxx * Syy) - (Sxy**2)
            traceH = Sxx + Syy
            R = detH - a*(traceH**2)
            #If corner response is over threshold, it is a corner
            if R > threshold:
                xs.append(x + int(feature_width/2 -1))
                ys.append(y + int(feature_width/2 -1))
    return np.asarray(xs), np.asarray(ys), confidences, scales, orientations
