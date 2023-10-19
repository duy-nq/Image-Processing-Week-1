import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read dental.jpg
img = cv.imread('dental.jpg', cv.IMREAD_GRAYSCALE)

# Histogram equalization
equ = cv.equalizeHist(img)

# Using subplot to plot images and histograms
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(equ, cmap='gray'), plt.title('Equalized')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.hist(img.ravel(), 256, [0, 256]), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(equ.ravel(), 256, [0, 256]), plt.title('Equalized')
plt.xticks([]), plt.yticks([])
plt.show()

# Wait for user input
cv.waitKey(0)
cv.destroyAllWindows()

