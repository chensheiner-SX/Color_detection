import cv2
import glob
import numpy as np
import shutil
import os


def color_identify(image):
    """
    Masking images per boundaries of colors in HSV color scheme.
    Returning the color with the maximum sum value of the masked image

    :param image: numpy.ndarray : the IMAGE in BGR color format
    :return: str: color of the car in the image. value options: ['red', 'white', 'black', 'gray']
    """
    # Static Variables
    boundaries = [
        ([114, 50, 50], [179, 255, 255]),  # Red
        ([60, 50, 193], [179, 255, 255]),  # White
        ([20, 100, 20], [179, 255, 90]),  # Black
        ([75, 80, 110], [179, 125, 190]),  # Gray
    ]

    colors = ['red', 'white', 'black', 'gray']

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output = np.empty([4, *image.shape])
    # loop over the boundaries
    for i, (lower, upper) in enumerate(boundaries):
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image_hsv, lower, upper)
        output[i] = cv2.bitwise_and(image, image, mask=mask)
    # Calculate the sum of all the pixel values.
    sums = np.array([im.sum() for im in output])
    return colors[sums.argmax()]


def main():
    """
    Categorize the image to specific color.
    Save it to the matching color folder in "results/"

    :return:
    """
    # Delete old results
    if os.path.isdir('results/'):
        shutil.rmtree('results/')
    for color in colors:
        os.makedirs(f"results/{color}", exist_ok=True)

    files = glob.glob("ogen_car_stamps/*.png")
    files = sorted(files)
    for file in files:
        img = cv2.imread(file)
        image_color = color_identify(img)
        # write the image to the color directory of the max sum value
        cv2.imwrite(f'results/{image_color}/{file.split("/")[-1]}', img)



if __name__ == "__main__":
    main()
