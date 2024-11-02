import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the image in grayscale
image = cv2.imread("document.png", cv2.IMREAD_GRAYSCALE)

# giai doan 1
# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(image, (3, 3), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour by area
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour to a polygon and draw a bounding rectangle
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
x, y, w, h = cv2.boundingRect(approx)

# Draw the bounding rectangle on the original image (converted to RGB for visualization)
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Approximate the contour to a polygon and draw a bounding rectangle
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
x, y, w, h = cv2.boundingRect(approx)

# Crop the image to keep only the content within the bounding rectangle
cropped_image = image[y:y+h, x:x+w]

# giai doan 2

# Load and preprocess the image
blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Apply Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# Detect angles of the lines
angles = []
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        angle = math.degrees(theta)  # Convert theta from radians to degrees
        angles.append(angle)

# Select the angle closest to 90 degrees for rotation
# Calculate the rotation angle to make the line vertical
if angles:
    target_angle = min(angles, key=lambda x: abs(x - 90))  # Closest to 90 degrees
    rotation_angle =  np.abs(90 - target_angle) # Angle to rotate to make it vertical

    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # Display the rotated image
    plt.subplot(131)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title("Contours")
    plt.imshow(image_rgb, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title("Rotate Document")
    plt.imshow(rotated_image, cmap='gray')
    plt.axis('off')
    plt.show()
else:
    print("Không tìm thấy đường để xoay.")
