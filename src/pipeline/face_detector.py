"""
Face Detection Module
Wrapper for SCRFD (Sample and Computation Redistribution for Face Detection)
or RetinaFace for high-speed face detection with landmarks.

This module provides a unified interface for face detection regardless of the
underlying detector implementation.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Face detector wrapper
    
    Detects faces and 5 facial landmarks (eyes, nose, mouth corners)
    Supports both SCRFD and RetinaFace backends
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = 'opencv',
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize face detector
        
        Args:
            model_path: Path to ONNX model file (if using ONNX backend)
            backend: 'opencv' (uses OpenCV DNN) or 'onnxruntime'
            conf_threshold: Confidence threshold for detection
            nms_threshold: NMS threshold
            input_size: Model input size (width, height)
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.backend = backend
        
        # For now, we'll use a simple fallback detector (Haar Cascade)
        # In production, replace with SCRFD/RetinaFace ONNX model
        self.use_simple_detector = model_path is None
        
        if self.use_simple_detector:
            # Fallback to OpenCV's Haar Cascade for demo purposes
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("Using OpenCV Haar Cascade detector (fallback)")
                print("For production, provide SCRFD/RetinaFace ONNX model path")
            except Exception as e:
                print(f"Warning: Could not load face detector: {e}")
                self.face_cascade = None
        else:
            # Load ONNX model for SCRFD/RetinaFace
            if backend == 'onnxruntime':
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']
                )
                self.input_name = self.session.get_inputs()[0].name
            elif backend == 'opencv':
                self.net = cv2.dnn.readNetFromONNX(model_path)
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image
        
        Args:
            image: BGR image array
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection score
                - landmarks: 5 facial landmarks (2x5 array) [optional]
        """
        if self.use_simple_detector:
            return self._detect_simple(image)
        else:
            return self._detect_onnx(image)
    
    def _detect_simple(self, image: np.ndarray) -> List[dict]:
        """Simple detection using Haar Cascade (fallback)"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            # Generate approximate landmarks (for demo)
            landmarks = self._generate_approximate_landmarks(x, y, w, h)
            
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.9,  # Haar cascade doesn't provide confidence
                'landmarks': landmarks
            })
        
        return detections
    
    def _generate_approximate_landmarks(self, x, y, w, h) -> np.ndarray:
        """
        Generate approximate 5-point landmarks based on bbox
        Order: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        landmarks = np.array([
            [x + w * 0.3, y + h * 0.35],  # left eye
            [x + w * 0.7, y + h * 0.35],  # right eye
            [x + w * 0.5, y + h * 0.55],  # nose
            [x + w * 0.35, y + h * 0.75], # left mouth
            [x + w * 0.65, y + h * 0.75], # right mouth
        ], dtype=np.float32)
        return landmarks
    
    def _detect_onnx(self, image: np.ndarray) -> List[dict]:
        """
        Detection using ONNX model (SCRFD/RetinaFace)
        
        This is a placeholder for the actual ONNX inference.
        Replace with proper preprocessing and postprocessing for your model.
        """
        # Preprocess
        img_resized = cv2.resize(image, self.input_size)
        img_normalized = (img_resized - 127.5) / 128.0
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        
        # Inference
        if self.backend == 'onnxruntime':
            outputs = self.session.run(None, {self.input_name: img_batch})
        else:  # opencv
            self.net.setInput(img_batch)
            outputs = self.net.forward()
        
        # Postprocess (model-specific)
        # This is a placeholder - implement based on your specific model
        detections = self._postprocess_onnx(outputs, image.shape[:2])
        
        return detections
    
    def _postprocess_onnx(self, outputs, original_shape):
        """
        Postprocess ONNX outputs
        This should be implemented based on the specific model used
        """
        # Placeholder implementation
        # Real implementation depends on SCRFD/RetinaFace output format
        return []
    
    def draw_detections(self, image: np.ndarray, detections: List[dict], 
                       color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        Draw bounding boxes and landmarks on image
        
        Args:
            image: Input image
            detections: List of detection dicts
            color: Box color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with drawn detections
        """
        img_draw = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            conf = det.get('confidence', 0.0)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence
            label = f"{conf:.2f}"
            cv2.putText(img_draw, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw landmarks if available
            if 'landmarks' in det and det['landmarks'] is not None:
                landmarks = det['landmarks']
                for (lx, ly) in landmarks:
                    cv2.circle(img_draw, (int(lx), int(ly)), 2, (0, 0, 255), -1)
        
        return img_draw


if __name__ == "__main__":
    """Test face detector"""
    import time
    
    print("=" * 80)
    print("Face Detector Test")
    print("=" * 80)
    
    # Initialize detector (fallback mode)
    detector = FaceDetector()
    
    # Create a test image with a synthetic face region
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw a simple face-like structure
    cv2.circle(test_img, (320, 240), 80, (100, 100, 100), -1)  # Face
    cv2.circle(test_img, (290, 220), 10, (50, 50, 50), -1)      # Left eye
    cv2.circle(test_img, (350, 220), 10, (50, 50, 50), -1)      # Right eye
    cv2.ellipse(test_img, (320, 260), (30, 15), 0, 0, 180, (50, 50, 50), 2)  # Mouth
    
    print("\nRunning face detection...")
    start_time = time.time()
    detections = detector.detect(test_img)
    elapsed = time.time() - start_time
    
    print(f"Detected {len(detections)} face(s) in {elapsed*1000:.1f}ms")
    
    for i, det in enumerate(detections):
        print(f"\nFace {i+1}:")
        print(f"  Bbox: {det['bbox']}")
        print(f"  Confidence: {det['confidence']:.3f}")
        if 'landmarks' in det:
            print(f"  Landmarks shape: {det['landmarks'].shape}")
    
    # Draw detections
    img_with_dets = detector.draw_detections(test_img, detections)
    
    print("\n" + "=" * 80)
    print("Integration Notes:")
    print("=" * 80)
    print("""
    Current Implementation:
    - Uses OpenCV Haar Cascade as fallback
    - For PRODUCTION, replace with SCRFD or RetinaFace ONNX model
    
    To use SCRFD/RetinaFace:
    1. Download pretrained ONNX model (e.g., scrfd_10g_bnkps.onnx)
    2. Initialize: detector = FaceDetector(model_path='path/to/model.onnx')
    3. The detector will automatically use ONNX runtime
    
    Recommended Models:
    - SCRFD-10G: High accuracy, ~10ms on CPU
    - SCRFD-2.5G: Balanced, ~5ms on CPU
    - RetinaFace-MobileNet0.25: Ultra-fast, ~3ms on CPU
    
    All models should output:
    - Bounding boxes
    - 5 facial landmarks (eyes, nose, mouth corners)
    - Confidence scores
    """)
