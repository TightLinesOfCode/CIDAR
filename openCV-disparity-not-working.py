import cv2 as cv
import numpy as np

# Load the left and right images (grayscale)
imgL = cv.imread('cam1.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('cam2.png', cv.IMREAD_GRAYSCALE)

# Initialize the StereoBM object
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(imgL, imgR)

# Normalize the disparity map for visualization
min_val = disparity.min()
max_val = disparity.max()
normalized_disparity = np.uint8(255 * (disparity - min_val) / (max_val - min_val))

# Display the disparity map
cv.imshow('Disparity Map', normalized_disparity)
cv.waitKey(0)
cv.destroyAllWindows()