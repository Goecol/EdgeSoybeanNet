import cv2
import numpy as np

# Load the image
image = cv2.imread('dataset/cropped_DJI_0073_1_crop_265_760.jpg')

def perform_contour_box_process(image_received):
        image = image_received

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding to separate soybean pods from the background
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Apply morphological operations to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # Find contours of the soybean pods
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank image to store the soybean pods
        image_with_pods = np.zeros_like(image)

        # Define a margin to increase the bounding box size
        margin = 10  # Increase the margin to make the bounding box bigger

        # Iterate over the contours and copy the soybean pods inside the bounding box region
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Set an appropriate area threshold to avoid small noise
                # Get bounding box coordinates (x, y, width, height)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Increase the bounding box size by adding a margin
                x, y, w, h = x - margin, y - margin, w + 2 * margin, h + 2 * margin
                
                # Copy the soybean pods inside the bounding box region
                image_with_pods[y:y + h, x:x + w] = image[y:y + h, x:x + w]

        return image_with_pods


'''
image_with_pods = perform_contour_box_process(image)
# Save the final image with soybean pods (no bounding box borders)
cv2.imwrite('soybean_pods_without_borders_5.bmp', image_with_pods)

# Display the result
cv2.imshow('Soybean Pods Without Borders', image_with_pods)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''