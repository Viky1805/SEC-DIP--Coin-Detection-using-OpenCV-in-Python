# SEC-DIP--Coin-Detection-using-OpenCV-in-Python
# DIPT-WORKSHOP-4
# Coin-Detection-using-OpenCV-in-Python
## Name : Vignesh S
## Reg.no : 212224110061

### PROGRAM :

```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read image
image = cv2.imread('CoinsA.png')
imageCopy = image.copy()

# Display original image
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.show()

# Convert to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(imageGray, cmap='gray')
plt.title("Grayscale Image")
plt.show()


# Split BGR channels
imageB, imageG, imageR = cv2.split(image)

plt.figure(figsize=(20, 12))

plt.subplot(141)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")

plt.subplot(142)
plt.imshow(imageB, cmap='gray')
plt.title("Blue Channel")

plt.subplot(143)
plt.imshow(imageG, cmap='gray')
plt.title("Green Channel")

plt.subplot(144)
plt.imshow(imageR, cmap='gray')
plt.title("Red Channel")

plt.show()


# Thresholding
thresh = 20
maxValue = 255

th, dst_bin_inv = cv2.threshold(
    imageG,
    thresh,
    maxValue,
    cv2.THRESH_BINARY_INV
)

plt.imshow(dst_bin_inv, cmap='gray', vmin=0, vmax=255)
plt.title("Threshold Binary Inverse")
plt.show()


# Dilation
kSize = (5, 5)
kernel2 = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    kSize
)

imageDilated2 = cv2.dilate(
    dst_bin_inv,
    kernel2,
    iterations=2
)

plt.imshow(imageDilated2, cmap='gray')
plt.title("Dilated Image Iteration 2")
plt.show()


# Erosion
kSize = (11, 11)

kernel1 = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    kSize
)

imageEroded = cv2.erode(
    imageDilated2,
    kernel1
)

plt.imshow(imageEroded, cmap='gray')
plt.title("Eroded Image")
plt.show()


# Simple Blob Detector
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0
params.minDistBetweenBlobs = 2

# Filter by Area
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8


# Create detector
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(imageEroded)

# Draw detected coins
imageWithKeypoints = cv2.drawKeypoints(
    image,
    keypoints,
    np.array([]),
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display result
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(imageWithKeypoints, cv2.COLOR_BGR2RGB))
plt.title("Detected Coins")
plt.axis("off")
plt.show()

# Count coins
print("Number of coins detected:", len(keypoints))

```

## OUTPUT :
<img width="538" height="513" alt="image" src="https://github.com/user-attachments/assets/683849d0-9c42-4132-ad9c-064940fd07e8" />

<img width="1297" height="657" alt="image" src="https://github.com/user-attachments/assets/1f8f0eac-4ca8-4a3a-995a-97db07175d42" />

<img width="772" height="233" alt="image" src="https://github.com/user-attachments/assets/26a098cc-4744-4a3a-a9fb-c768d6c320dc" />

<img width="620" height="537" alt="image" src="https://github.com/user-attachments/assets/3e299fc6-891d-4e8d-b282-afa58b26681c" />

<img width="630" height="537" alt="image" src="https://github.com/user-attachments/assets/d67a878c-7fa9-4d9b-af13-5e3c75518a3e" />

<img width="597" height="551" alt="image" src="https://github.com/user-attachments/assets/1c351f30-369d-4a4b-ae6a-4f752fd214fc" />

<img width="797" height="820" alt="image" src="https://github.com/user-attachments/assets/fc1b5994-778f-4345-8548-0a07da76005d" />
