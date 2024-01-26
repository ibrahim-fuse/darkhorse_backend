import cv2
import numpy as np

class NearestObjectFinder:
    def __init__(self, image_path, bbox1, bbox2):
        self.image_path = image_path
        self.bbox1 = bbox1
        self.bbox2 = bbox2

    def find_nearest_object(self):
        # Load the image
        image = cv2.imread(self.image_path)

        # Calculate the center of the image
        image_height, image_width, _ = image.shape
        image_center = (image_width // 2, image_height // 2)

        # Calculate the centers of the bounding boxes
        center1 = ((self.bbox1[0] + self.bbox1[2]) // 2, (self.bbox1[1] + self.bbox1[3]) // 2)
        center2 = ((self.bbox2[0] + self.bbox2[2]) // 2, (self.bbox2[1] + self.bbox2[3]) // 2)

        # Calculate distances from image center to bounding box centers
        distance1 = np.sqrt((center1[0] - image_center[0])**2 + (center1[1] - image_center[1])**2)
        distance2 = np.sqrt((center2[0] - image_center[0])**2 + (center2[1] - image_center[1])**2)

        # Determine which object is closer
        if distance1 < distance2:
            nearest_bbox = self.bbox1
        else:
            nearest_bbox = self.bbox2

        # Draw bounding boxes on the image
        image_with_boxes = image.copy()
        cv2.rectangle(image_with_boxes, (self.bbox1[0], self.bbox1[1]), (self.bbox1[2], self.bbox1[3]), (0, 255, 0), 2)
        cv2.rectangle(image_with_boxes, (self.bbox2[0], self.bbox2[1]), (self.bbox2[2], self.bbox2[3]), (0, 0, 255), 2)

        return image_with_boxes, nearest_bbox

if __name__ == "__main__":
    image_path = "path_to_image.jpg"  # Provide the path to your image
    bbox1 = (100, 100, 300, 300)  # Format: (left, top, right, bottom)
    bbox2 = (400, 400, 600, 600)

    finder = NearestObjectFinder(image_path, bbox1, bbox2)
    result_image, nearest_bbox = finder.find_nearest_object()

    # Display the result
    cv2.imshow("Nearest Object", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Nearest Bounding Box:", nearest_bbox)
