import numpy as np
from utils import im_range


def my_imfilter(image, filter):
    image1 = image.copy()

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    def Mul(A, B):  # A,B are 2 matrixes with same size, B need to be flipped
        return np.sum(A * np.flip(B))

    def getMat(Mat, i, j, filter):
        return Mat[i - filter.shape[0] // 2: i + filter.shape[0] // 2 + 1,
                   j - filter.shape[1] // 2: j + filter.shape[1] // 2 + 1]

    def ExtendMat(img, Filter):
        A = img.copy()
        M = np.zeros((A.shape[0] + Filter.shape[0] - 1,
                     A.shape[1] + Filter.shape[1] - 1))
        M[Filter.shape[0] // 2: M.shape[0] - (Filter.shape[0] // 2),
          Filter.shape[1] // 2: M.shape[1] - (Filter.shape[1] // 2)] = A
        return M

    def Cov(img, Filter):
        Mat = ExtendMat(img, Filter)
        img1 = ExtendMat(img, Filter)
        for i in range(Filter.shape[0] // 2, Filter.shape[0] // 2 + img.shape[0]):
            for j in range(Filter.shape[1] // 2, Filter.shape[1] // 2 + img.shape[1]):
                Mat[i, j] = Mul(getMat(img1, i, j, Filter), Filter)
        return Mat[Filter.shape[0] // 2: Mat.shape[0] - (Filter.shape[0] // 2), Filter.shape[1] // 2: Mat.shape[1] - (Filter.shape[1] // 2)]

    for i in range(len(image1[0][0])):
        image1[:, :, i] = Cov(image1[:, :, i], filter)

    return image1


def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
      as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### TODO: YOUR CODE HERE ###

    low_frequencies = im_range(my_imfilter(image1, filter))

    high_frequencies = im_range(image2 - my_imfilter(image2, filter))

    hybrid_image = im_range(low_frequencies + high_frequencies)

    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
