import cv2
import numpy as np

import image_extract_points_json as extract

# Load the image
image = cv2.imread('/home/johnbosco/soybean/dataset/original/train/DJI_0075_1.bmp')
points = extract.extract_points_from_json("/home/johnbosco/soybean/dataset/original/json/DJI_0075_1.json")


def perform_contour_box_process_using_dots(image_received, points):
        # List of known soybean pod coordinates
        image = image_received
        dot_points = points

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask for extracted pods
        mask = np.zeros_like(image)

        # Define function to calculate Euclidean distance
        def euclidean_distance(pt1, pt2):
            return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

        # Match contours to dot points
        matched_contours = []
        for dot in dot_points:
            closest_contour = None
            min_distance = float('inf')
            for contour in contours:
                # Calculate contour centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:  # Avoid division by zero
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    distance = euclidean_distance(dot, (cx, cy))
                    if distance < min_distance and distance < 20:  # Threshold for distance
                        min_distance = distance
                        closest_contour = contour
            if closest_contour is not None:
                matched_contours.append(closest_contour)

        # Create a copy of the original image for visualization
        visualized_image = image.copy()

        # Draw matched contours and dot points on the visualization image
        for contour in matched_contours:
            cv2.drawContours(visualized_image, [contour], -1, (0, 255, 0), 2)  # Green for contours
        for dot in dot_points:
            cv2.circle(visualized_image, dot, 5, (255, 0, 0), -1)  # Blue for dots

        # Draw matched contours on the mask
        for contour in matched_contours:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

        # Extract pods using the mask
        extracted_pods = cv2.bitwise_and(image, mask)

        #list = []
        return visualized_image, extracted_pods, matched_contours, dot_points


'''
#visualized_image, extracted_pods, matched_contours, dot_points = perform_contour_box_process_using_dots(image, points)
result = perform_contour_box_process_using_dots(image, points)

# Save and display the two results
cv2.imshow('Contours and Dots', result[0])
cv2.imwrite('visualized_contours_and_dots.bmp',result[0])

cv2.imshow('Extracted Soybean Pods', result[1])
cv2.imwrite('refined_extracted_soybean_pods.bmp', result[1])

num_matched_contours = len(result[2])
num_points = len(result[3])
print("The number of contours: ", num_matched_contours)
print("The number of points: ", num_points)
'''