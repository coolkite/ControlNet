import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from annotator.util import HWC3, resize_image
from annotator.midas import MidasDetector


class DepthEstimator:
    def __init__(self, model_name="midas", detect_resolution=384, image_resolution=512):
        self.model_name = model_name.lower()
        self.detect_resolution = detect_resolution
        self.image_resolution = image_resolution
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model_name == "midas":
            self.model = MidasDetector()
        elif self.model_name == "zoe":
            repo = "isl-org/ZoeDepth"
            self.model = torch.hub.load(repo, "ZoeD_N", pretrained=True).to(self.device)
        else:
            checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
            self.model = pipeline("depth-estimation", model=checkpoint, device=self.device)

    def estimate_depth(self, input_image):
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        
        input_image = HWC3(input_image)
        
        if self.model_name == "midas":
            return self._estimate_depth_midas(input_image)
        elif self.model_name == "zoe":
            return self._estimate_depth_zoe(input_image)
        else:
            return self._estimate_depth_hub(input_image)

    def _estimate_depth_midas(self, input_image):
        detected_map, _ = self.model(resize_image(input_image, self.detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, self.image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        return self._normalize_depth_map(detected_map)

    def _estimate_depth_zoe(self, input_image):
        detected_map = self.model.infer_pil(Image.fromarray(resize_image(input_image, self.detect_resolution)))
        detected_map = HWC3(np.array(detected_map))
        img = resize_image(input_image, self.image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        return self._normalize_depth_map(detected_map)

    def _estimate_depth_hub(self, input_image):
        img = resize_image(input_image, self.image_resolution)
        predictions = self.model(Image.fromarray(img))
        detected_map = predictions["depth"]
        return self._normalize_depth_map(np.array(detected_map))

    def _normalize_depth_map(self, depth_map):
        depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return depth_map