import cv2 as cv
import numpy as np

img1 = [
    [[0, 0, 255], [0, 0, 0]],
    [[255, 0, 0], [0, 255, 0]]
]
img = np.array(img1)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()