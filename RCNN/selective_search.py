'''
Computing hierachical grouping of similar regions
based on color, texture, size and shape
'''
from skimage.segmentation import felzenszwalb
from skimage.io import imread, imshow
import numpy as np
import matplotlib.pyplot as plt

def calculate_color_hist(mask, img):
    # mask is the corresponding region
    BINS = 25

    # If the image is only 2 dimensional, convert it to 3 dimension
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    
    hist = np.array([])

    for ch in range(img.shape[2]):
        # Take the corresponding mask layers
        layer = img[:, :, ch][mask]
    



def oversegmentation(img, k):

    img_seg = felzenszwalb(img, scale=k, sigma=0.8, min_size=100)

    return img_seg


if __name__ == "__main__":
    k = 3.0
    path = "./RCNN/dog.jpg"
    img = imread(path)
    imshow(img)
    # plt.show()
    segments = oversegmentation(img, k)
    imshow(segments)
    # plt.show()
    print(segments)
    print(segments.shape)

    # Number of unique elements is the number of regions
    # print(np.unique(segments))
    print(type(segments))
    # segment is a 2d array

    # seg_dict = {k:np.where(segments == k)[0] for k in np.unique(segments)}
    # seg_dict = {k:[] for k in np.unique(segments)}
    # for iy, ix in np.ndindex(segments.shape):
    #     temp = segments[iy, ix]
    #     # Append (row_number, col_number)
    #     seg_dict[temp].append([ix, iy])
    # print(seg_dict[0])


    mask_0 = segments == np.unique(segments)[0]
    # Take out the pixel value of the mask with 0 from the first channels
    layer_0 = img[:, :, 0][mask_0]
    hist_0 = np.histogram(layer_0, bins=25)[0]
    print(mask_0)
    print(layer_0)
    print(hist_0)