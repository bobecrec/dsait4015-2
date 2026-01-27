import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random

from keras.src.utils import load_img, img_to_array


######################## 1D RAMP SIGNAL OVER IMAGE ##############################

def create_1d_signal(shape, strength, epsilon):
    signal = np.zeros((shape[0], shape[1], 3), dtype=np.int32)

    if strength / 256 > epsilon:
        strength = epsilon * 256

    for i in range(shape[0]):
        for j in range(shape[1]):
            offset = int(strength * (shape[0] - i) / shape[0])
            signal[i, j] = offset
            # signal[i, j, 0] = offset # channel specific

    # cv2.imwrite(os.path.join(curr_dir, "images", "mod", "ramp.jpeg"), signal)

    return signal


######################## 2D SIN SIGNAL OVER IMAGE ##############################

# max value in signal will be = strength => max divergence from pxl value will be strength/256
# if divergence would be > epsilon, it gets capped
def create_2d_signal(shape, strength, freq, epsilon):
    signal = np.zeros((shape[0], shape[1], 3), dtype=np.int32)

    if strength / 255 > epsilon:
        strength = epsilon * 255

    for i in range(shape[0]):
        for j in range(shape[1]):
            offset = int (strength * np.sin(2 * np.pi * (i * j) * freq / shape[0]))
            # signal[i, j, :] = offset
            signal[i, j, np.random.randint(0, 3)] = offset # channel specific

    # cv2.imwrite(os.path.join(curr_dir, "images", "mod", "2dsignal.jpeg"), signal)

    return signal


######################## RANDOM NOISE OVER IMAGE ##############################

def create_noise(shape, intensity, fraction, epsilon):
    noise = np.zeros((shape[0] * shape[1], 3), dtype=np.int32)
    rnd_idx = random.sample(range(shape[0]*shape[1]), k=int(len(noise) * fraction))

    if intensity / 256 > epsilon:
        intensity = int(epsilon * 256)

    for i in rnd_idx:
        n = random.randint(-intensity, intensity)
        noise[i] = n

    noise = noise.reshape(shape)

    # cv2.imwrite(os.path.join(curr_dir, "images", "mod", "random_noise.jpeg"), noise)

    return noise


##################### NOISE OVER EDGE OF IMAGE ################################

def find_edges(image, intensity, threshold, epsilon):
    copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(copy, cv2.CV_32F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(copy, cv2.CV_32F, 0, 1, ksize=1)

    if intensity / 256 > epsilon:
        intensity = int(epsilon * 256)

    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    noise = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            # considered an edge
            if edges[i, j] > int(threshold * 256):
                noise[i, j, :] = random.randint(-intensity, intensity)

    # cv2.imwrite(os.path.join(curr_dir, "images", "mod", "edge.jpeg"), noise)

    return noise


######################## CHECKERBOARD PATH ON TOP LEFT CORNER ##############################

def create_patch(shape, size, epsilon):
    patch = np.zeros((shape[0], shape[1], 3), dtype=np.int32)

    for i in range(size // 2):
        for j in range(size // 2):
            patch[i*2, j*2] = np.floor(epsilon * 256)
        for j in range(size // 2):
            patch[i*2+1, j*2+1] = np.floor(epsilon * 256)


    # cv2.imwrite(os.path.join(curr_dir, "images", "mod", "patch.jpeg"), patch)

    return patch


def apply_noise(image, signal):
    return np.clip(image + signal, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    ########################### PIPELINE ###################################################
    curr_dir = os.getcwd()
    img_path = os.path.join(curr_dir, "images", "espresso.jpeg")

    img = img_to_array(load_img(img_path)).astype(np.uint8)
    # im_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_copy = img.astype(np.uint8)

    # signal = create_2d_signal(im_copy.shape, 100, 50, 0.1)  # SELECT METHOD, EXCEPT FOR EDGE DET. DONT REGENERATE SIGNAL PER IMAG
    signal = find_edges(im_copy, 1000, 0.1, 0.3)
    image = np.clip(im_copy + signal, 0, 255).astype(np.uint8)
    # cv2.imwrite(os.path.join(curr_dir, "images", "mod", "mod_im.jpeg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    plt.imshow(image)
    plt.show()