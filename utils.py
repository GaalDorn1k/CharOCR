import numpy as np
import cv2
import os

from tqdm import tqdm


def get_probabilities(quality: int, path_to_masks: str) -> tuple:
    '''
    return probabilities classes
    class probability - percentage of pixels belonging to this class

    args:
    quality - number of images to analyze
    '''
    masks = [os.path.join(path_to_masks, m) for m in os.listdir(path_to_masks) if m.endswith('npy')]
    m = np.load(masks[0])
    classes = np.zeros(m.shape[1] + 1)

    for i in tqdm(range(quality)):
        cm = cv2.imread(masks[i])
        cm = cm[0]
        cm = np.round(cm)
        cm = cm.swapaxes(0, 2)
        cm = cm.swapaxes(0, 1)
        for y in range(cm.shape[0]):
            for x in range(cm.shape[1]):
                pix = cm[y][x]
                c = np.where(pix == 1)
                classes[c] += 1

    classes = classes / np.sum(classes)
    classes = np.array([1 - i for i in classes])
    classes = classes / np.sum(classes)

    return classes
