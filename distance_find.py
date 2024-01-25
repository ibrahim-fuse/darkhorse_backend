import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageDistancePlotter:
    def __init__(self, image_location, object_bbox):
        self.image_location = image_location
        self.object_bbox = object_bbox
        self.image = cv2.imread(image_location)

    def calculate_distance(self):
        if self.image is None:
            print("Image not found.")
            return None

        # Get image dimensions
        height, width, _ = self.image.shape

        # Calculate center of the image
        image_center = (width // 2, height // 2)

        # Calculate center of the object bounding box
        object_center = (
            (self.object_bbox[0] + self.object_bbox[2]) // 2,
            (self.object_bbox[1] + self.object_bbox[3]) // 2
        )

        # Calculate Euclidean distance between the centers
        distance = np.sqrt((image_center[0] - object_center[0])**2 + (image_center[1] - object_center[1])**2)

        return distance

    def plot_distance_on_image(self, distance):
        if distance is None:
            return

        # Draw the distance on the image
        cv2.putText(self.image, f'Distance: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def save_image_with_distance(self, output_location):
        if self.image is None:
            print("Image not found.")
            return

        self.plot_distance_on_image(self.calculate_distance())
        cv2.imwrite(output_location, self.image)

if __name__ == "__main__":
    image_location = 'your_image.jpg'  # Replace with the path to your image
    object_bbox = [100, 100, 300, 300]  # Replace with your object's bounding box [x_min, y_min, x_max, y_max]
    output_location = 'output_image_with_distance.jpg'  # Replace with the desired output path

    plotter = ImageDistancePlotter(image_location, object_bbox)
    plotter.save_image_with_distance(output_location)
    
    print(f'Distance between center and object: {plotter.calculate_distance()} pixels')
