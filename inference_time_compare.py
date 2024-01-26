import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import time

class ObjectDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model.to(self.device)
        return model

    def preprocess_input(self, image_path):
        transform = T.Compose([T.ToPILImage(), T.ToTensor()])
        image = Image.open(image_path).convert("RGB")
        image = transform(image).to(self.device)
        return image.unsqueeze(0)

    def detect_objects(self, image_path):
        image = self.preprocess_input(image_path)
        
        # Measure inference time on CPU
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(image)
        cpu_inference_time = time.time() - start_time

        # Move the model to GPU (if available) and measure inference time
        self.model.to("cuda")
        image = image.to("cuda")
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(image)
        gpu_inference_time = time.time() - start_time

        # Return the inference times
        return cpu_inference_time, gpu_inference_time

if __name__ == "__main__":
    model_path = "path_to_pretrained_model.pth"  # Provide the path to the pre-trained model
    image_path = "path_to_dummy_input_image.jpg"  # Provide the path to a sample input image
    
    detector = ObjectDetector(model_path)
    
    cpu_time, gpu_time = detector.detect_objects(image_path)
    
    print(f"Inference time on CPU: {cpu_time:.4f} seconds")
    print(f"Inference time on GPU: {gpu_time:.4f} seconds")
