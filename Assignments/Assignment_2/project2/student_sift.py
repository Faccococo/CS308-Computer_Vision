import numpy as np
from skimage.filters import scharr_h, scharr_v, sobel_h, sobel_v, gaussian
import cv2
# debug:
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
def get_features(image, xs, ys, feature_width, scales = None):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """
    # Convert to integers for indexing 
    xs = np.round(xs).astype(int)
    ys = np.round(ys).astype(int)

    # Define helper functions for readabilty and avoid copy-pasting
    def get_window(y, x):
        """
         Helper to get indices of the feature_width square
        """
        rows = (x - (feature_width/2 -1), x + feature_width/2)
        if rows[0] < 0:
            rows = (0, rows[1] - rows[0])
        if rows[1] >= image.shape[0]:
            rows = (rows[0]  + (image.shape[0] -1 - rows[1]), image.shape[0] - 1)
        cols = (y - (feature_width/2 -1), y + feature_width/2)
        if cols[0] < 0:
            cols = (0, cols[1] - cols[0])
        if cols[1] >= image.shape[1]:
            cols = (cols[0]  - (cols[1] + 1 - image.shape[1]), image.shape[1] - 1)
        return int(rows[0]), int(rows[1]) + 1, int(cols[0]), int(cols[1]) + 1

    def get_current_window(i, j, matrix):
        """
        Helper to get sub square of size feature_width/4 
        From the square matrix of size feature_width
        """
        return matrix[int(i*feature_width/4):
                    int((i+1)*feature_width/4),
                    int(j*feature_width/4):
                    int((j+1)*feature_width/4)]

    def rotate_by_dominant_angle(angles, grads):
        hist, bin_edges = np.histogram(angles, bins= 36, range=(0, 2*np.pi), weights=grads)
        angles -= bin_edges[np.argmax(hist)]
        angles[ angles < 0 ] += 2 * np.pi
    
    # Initialize features tensor, with an easily indexable shape
    features = np.zeros((len(xs), 4, 4, 8))
    # Get gradients and angles by filters (approximation)
    sigma = 0.8
    filtered_image = gaussian(image, sigma)
    dx = scharr_v(filtered_image)
    dy = scharr_h(filtered_image)
    gradient = np.sqrt(np.square(dx) + np.square(dy))
    angles = np.arctan2(dy, dx)
    angles[angles < 0 ] += 2*np.pi

    for n, (x, y) in enumerate(zip(xs, ys)):
        # Feature square 
        i1, i2, j1, j2 = get_window(x, y)
        grad_window = gradient[i1:i2, j1:j2]
        angle_window = angles[i1:i2, j1:j2]
        # Loop over sub feature squares 
        for i in range(int(feature_width/4)):
            for j in range(int(feature_width/4)):
                # Enhancement: a Gaussian fall-off function window
                current_grad = get_current_window(i, j, grad_window).flatten()
                current_angle = get_current_window(i, j, angle_window).flatten()
                features[n, i, j] = np.histogram(current_angle, bins=8,
                 range=(0, 2*np.pi), weights=current_grad)[0]
                
    features = features.reshape((len(xs), -1,))
    dividend = np.linalg.norm(features, axis=1).reshape(-1, 1)
    # Rare cases where the gradients are all zeros in the window
    # Results in np.nan from division by zero.
    dividend[dividend == 0 ] = 1
    features = features / dividend
    thresh = 0.25
    features[ features >= thresh ] = thresh
    features  = features ** 0.8
    # features = features / features.sum(axis = 1).reshape(-1, 1)
    return features