import math
import cv2
import numpy as np

def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

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

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    # fv = np.zeros((len(x), 128))
    fv = []
    for _ in range(len(x)):
        fv.append([])

    DX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 7)
    DY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 7)

    
    for n in range(len(x)):
        point_x = y[n]
        point_y = x[n]
        start_x = point_x - (feature_width // 2) + 1
        start_y = point_y - (feature_width // 2)
        for cell_start_x in range(start_x, start_x + (feature_width), feature_width // 4):
            for cell_start_y in range(start_y, start_y + (feature_width), feature_width // 4):
                
                angels = np.array([])
                for k in range(cell_start_x, cell_start_x + feature_width // 4):
                    for l in range(cell_start_y, cell_start_y + feature_width // 4):
                        angels = np.append(angels, np.arctan2(DX[k,l], DY[k,l]))
                angels = angels * (180/(math.pi))
                directions, _ = np.histogram(angels, bins = 8, range = (-180,180))
                for direction in directions:
                    fv[n].append(direction)

        fv_sum = np.sum(fv[n])
        fv[n] = fv[n] / fv_sum                


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return np.array(fv)
