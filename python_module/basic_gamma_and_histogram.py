import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

image = cv2.imread('data/data/frame_001780.jpg')

# Histogram Equalization
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
# cv2.imshow('Histogram Equalization', equalized)

# White Balance
balanced = white_balance(image)
# cv2.imshow('White Balance', balanced)

# Gamma Correction
gamma_corrected = adjust_gamma(image, gamma=2)
# cv2.imshow('Gamma Correction', gamma_corrected)


stacked_image = np.hstack((image, equalized, balanced, gamma_corrected))
cv2.imshow('Images', stacked_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
