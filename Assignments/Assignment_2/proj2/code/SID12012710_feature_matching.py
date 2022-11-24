import math
import numpy as np
import cv2


def match_features(features1, features2, x1, y1, x2, y2):
    """
        This function does not need to be symmetric (e.g. it can produce
        different numbers of matches depending on the order of the arguments).

        To start with, simply implement the "ratio test", equation 4.18 in
        section 4.1.3 of Szeliski. There are a lot of repetitive features in
        these images, and all of their descriptors will look similar. The
        ratio test helps us resolve this issue (also see Figure 11 of David
        Lowe's IJCV paper).

        For extra credit you can implement various forms of spatial/geometric
        verification of matches, e.g. using the x and y locations of the features.

        Args:
        -   features1: A numpy array of shape (n,feat_dim) representing one set of
                features, where feat_dim denotes the feature dimensionality
        -   features2: A numpy array of shape (m,feat_dim) representing a second set
                features (m not necessarily equal to n)
        -   x1: A numpy array of shape (n,) containing the x-locations of features1
        -   y1: A numpy array of shape (n,) containing the y-locations of features1
        -   x2: A numpy array of shape (m,) containing the x-locations of features2
        -   y2: A numpy array of shape (m,) containing the y-locations of features2

        Returns:
        -   matches: A numpy array of shape (k,2), where k is the number of matches.
                The first column is an index in features1, and the second column is
                an index in features2
        -   confidences: A numpy array of shape (k,) with the real valued confidence for
                every match

        'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
        """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################
    matches = []
    confidences = []

    for i in range(len(features1)):
        distances = []
        for j in range(len(features2)):
            feature_in_img1 = features1[i]
            feature_in_img2 = features2[j]
            distance = math.sqrt(np.sum((feature_in_img1 - feature_in_img2)*(feature_in_img1 - feature_in_img2)))
            distances.append(distance)
        distances_idx = np.argsort(distances)
        best_match_idx = distances_idx[0]
        second_match_idx = distances_idx[1]
        ratio = distances[best_match_idx] / (distances[second_match_idx]+10**-11)
        if ratio < 0.90:
            matches.append([i, best_match_idx])
            confidences.append(ratio)
        # print(distances)
            
    new_match = []
    idx = np.argsort(confidences)
    for i in idx:
        new_match.append(matches[i])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return np.array(new_match), np.array(confidences)
