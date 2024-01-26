import torch
import torchvision
import time

class ObjectDetector:
    def __init__(self, model_name):
        # Load the pre-trained object detection model
        self.model = torchvision.models.detection.__dict__[model_name](pretrained=True)
        self.model.eval()  # Set the model to evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def inference(self, dummy_input):
        # Convert the dummy input to a PyTorch tensor and move it to the device
        dummy_input = torch.from_numpy(dummy_input).to(self.device)

        # Measure inference time
        start_time = time.time()

        # Run inference
        with torch.no_grad():
            output = self.model(dummy_input)

        end_time = time.time()
        inference_time = end_time - start_time

        return output, inference_time

if __name__ == "__main__":
    # Example usage
    model_name = "fasterrcnn_resnet50_fpn"  # You can use other models like "ssd300", "ssdlite320", etc.
    dummy_input = torch.rand(1, 3, 224, 224).float().cpu().numpy()  # Replace with your input shape

    object_detector = ObjectDetector(model_name)
    output, inference_time = object_detector.inference(dummy_input)

    print(f"Inference Time: {inference_time} seconds")
    print(output)
