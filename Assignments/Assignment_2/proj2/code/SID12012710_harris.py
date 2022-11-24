import math
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    threshold = 1000
    step_size = 5
    x = []
    y = []
    confidences = []
    size = (image.shape[0], image.shape[1])
    DX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    DY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    DX = cv2.GaussianBlur(DX, (3, 3), 0.2) 
    DY = cv2.GaussianBlur(DY, (3, 3), 0.2)
    Ixx = DX ** 2
    Iyy = DY ** 2
    Ixy = DX * DY
    for i in range(0, size[0] - feature_width, step_size):
        for j in range(0, size[1] - feature_width, step_size):
                Ixx_sum = np.sum(Ixx[i:i+feature_width, j : j + feature_width + 1])
                Iyy_sum = np.sum(Iyy[i:i+feature_width, j : j + feature_width + 1])
                Ixy_sum = np.sum(Ixy[i:i+feature_width, j : j + feature_width + 1])
                Det = Ixx_sum*Iyy_sum - Ixy_sum**2
                Trace = Ixx_sum + Iyy_sum
                R = Det - 0.06 * (Trace**2)
                if R > threshold:
                        x.append(j + feature_width // 2)
                        y.append(i + feature_width // 2)
                        confidences.append(R)
    print(len(x))
    print(len(y))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################
    i = 0
    j = 0
    n = 1500
    points = []
    x_new, y_new = [], []
    confidences_new = []
    for i in range(len(x)):
        min_dis = sys.maxsize
        x_i, y_i = x[i], y[i]
        for j in range(len(x)):
            x_j, y_j = x[j], y[j]
            if x_i != x_j and y_i != y_j and confidences[i] < confidences[j]:
                dis = math.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
                if dis < min_dis:
                        min_dis = dis
        points.append([x_i, y_i, min_dis])

    points.sort(key = lambda x: x[2])
    points = points[- (n + 1) : -1]
    for point in points:
        x_new.append(point[0])
        y_new.append(point[1])
        confidences_new.append(point[2])
    
    x = np.asarray(x_new)
    y = np.asarray(y_new)
    confidences = np.asarray(confidences_new)
        
    # x = np.asarray(x)
    # y = np.asarray(y)
    # confidences = np.asarray(confidences)
            


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, confidences, scales, orientations
