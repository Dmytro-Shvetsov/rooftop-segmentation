import numpy as np


def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    size = rgb_image.shape[:2]
    encoded_image = np.zeros((*size, len(colormap)), dtype=np.int8)
    pixels = rgb_image.reshape((-1,3))
    for i, (cid, code) in enumerate(colormap.items()):
        encoded_image[:, :, i] = np.all(pixels == code, axis=1).reshape(size)
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in colormap:
        output[single_layer==k] = colormap[k]
    return np.uint8(output)