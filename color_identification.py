import cv2
import glob
import numpy as np
import shutil
import os
shutil.rmtree('results/')


boundaries=[
([114, 50, 50],[179,255,255]),  # Red
([60, 50, 193],[179,255,255]),  # White
([20, 100, 20],[179,255,90]),   # Black
([75, 80, 110],[179,125,190]),  # Gray
]
colors=['red', 'white', 'black', 'gray']

# Delete old results

shutil.rmtree('results/')
for color in colors:
    os.makedirs(f"results/{color}",exist_ok=True)

files = glob.glob("ogen_car_stamps/*.png")
files=sorted(files)
for file in files:
    img=cv2.imread(file)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    output=np.empty([4, *img.shape])
    # loop over the boundaries
    for i,(lower, upper) in enumerate(boundaries):
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image_hsv, lower, upper)
        output[i] = cv2.bitwise_and(img, img, mask = mask)
    # Calculate the sum of all the pixel values.
    sums=np.array([im.sum() for im in output])

    # write the image to the color directory of the max sum value
    cv2.imwrite(f'results/{colors[sums.argmax()]}/{file.split("/")[-1]}', img)
    # cv2.imshow('image', np.hstack([image, *output]))
    # cv2.waitKey(0)