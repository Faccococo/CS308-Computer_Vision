import numpy as np
from skimage.filters import scharr_h, scharr_v, sobel_h, sobel_v, gaussian
import cv2
# debug:
import matplotlib
import matplotlib.pyplot as plt


def match_features(im1_features, im2_features, x1, y1, x2, y2):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # Initialize variables
    matches = []
    confidences = []
    
    # Loop over the number of features in the first image
    for i in range(im1_features.shape[0]):
        # Calculate the euclidean distance between feature vector i in 1st image and all other feature vectors
        # second image
        distances = np.sqrt(((im1_features[i,:]-im2_features)**2).sum(axis = 1))

        # sort the distances in ascending order, while retaining the index of that distance
        ind_sorted = np.argsort(distances)
        # If the ratio between the 2 smallest distances is less than 0.8
        # add the smallest distance to the best matches
        if (distances[ind_sorted[0]] < 0.9 * distances[ind_sorted[1]]):
        # append the index of im1_feature, and its corresponding best matching im2_feature's index
            matches.append([i, ind_sorted[0]])
            confidences.append(1.0  - distances[ind_sorted[0]]/distances[ind_sorted[1]])
          # How can I measure confidence?
    confidences = np.asarray(confidences)
    confidences[np.isnan(confidences)] = np.min(confidences[~np.isnan(confidences)])     

    return np.asarray(matches), confidences