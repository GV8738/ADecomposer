import math
import torchvision
import numpy as np
from PIL import Image

def relabel(labels):
    labels[labels != 0] = 1

''''--------------------------------IMAGES FUNCTIONS-------------------------------------'''
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    ndarr = ndarr.astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(path)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

'''----------------------------------VAE FUNCTIONS---------------------------------------'''
def cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i+c*period)] = 0.5 - 0.5*math.cos(v*math.pi)
            v += step
            i += 1

    return L
