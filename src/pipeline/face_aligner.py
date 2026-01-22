"""
Face Alignment Module
Performs affine transformation to normalize face pose based on facial landmarks.

Standard alignment for face recognition:
- Input: Face image with 5 landmarks
- Output: 112x112 aligned face (canonical pose)

The alignment ensures:
1. Eyes are horizontally aligned
2. Face is centered
3. Consistent scale across all faces
"""

import numpy as np
import cv2
from typing import Tuple


# Standard reference landmarks for 112x112 face alignment
# These are empirically determined ideal positions for face recognition
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)


class FaceAligner:
    """
    Face alignment using similarity/affine transformation
    
    Aligns face based on 5 facial landmarks to a canonical pose.
    This normalization is crucial for face recognition accuracy.
    """
    
    def __init__(self, output_size: Tuple[int, int] = (112, 112)):
        """
        Initialize face aligner
        
        Args:
            output_size: Size of output aligned face (width, height)
        """
        self.output_size = output_size
        
        # Scale reference landmarks to output size
        scale = output_size[0] / 112.0
        self.reference_landmarks = REFERENCE_LANDMARKS_112 * scale
    
    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        bbox: np.ndarray = None
    ) -> np.ndarray:
        """
        Align face using similarity transformation
        
        Args:
            image: Input image (BGR)
            landmarks: 5 facial landmarks, shape (5, 2)
                      Order: left_eye, right_eye, nose, left_mouth, right_mouth
            bbox: Optional bounding box [x1, y1, x2, y2] for cropping before alignment
        
        Returns:
            Aligned face image of size output_size
        """
        # Compute similarity transformation matrix
        # This preserves angles and scales uniformly (rotation + translation + scaling)
        tform = self._estimate_transform(landmarks, self.reference_landmarks)
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            image,
            tform,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return aligned_face
    
    def _estimate_transform(
        self,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Estimate similarity transformation matrix
        
        Args:
            src_landmarks: Source landmarks (5, 2)
            dst_landmarks: Destination landmarks (5, 2)
        
        Returns:
            2x3 transformation matrix
        """
        # Use OpenCV's estimateAffinePartial2D for similarity transform
        # This finds optimal rotation, translation, and uniform scaling
        tform, _ = cv2.estimateAffinePartial2D(
            src_landmarks,
            dst_landmarks,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        if tform is None:
            # Fallback: use simple affine transform
            tform = cv2.getAffineTransform(
                src_landmarks[:3],
                dst_landmarks[:3]
            )
        
        return tform
    
    def align_batch(
        self,
        image: np.ndarray,
        landmarks_list: list,
        bboxes_list: list = None
    ) -> list:
        """
        Align multiple faces from the same image
        
        Args:
            image: Input image
            landmarks_list: List of landmark arrays
            bboxes_list: Optional list of bboxes
        
        Returns:
            List of aligned face images
        """
        aligned_faces = []
        
        for i, landmarks in enumerate(landmarks_list):
            bbox = bboxes_list[i] if bboxes_list is not None else None
            aligned_face = self.align(image, landmarks, bbox)
            aligned_faces.append(aligned_face)
        
        return aligned_faces
    
    def visualize_alignment(
        self,
        original: np.ndarray,
        aligned: np.ndarray,
        src_landmarks: np.ndarray = None
    ) -> np.ndarray:
        """
        Create visualization showing original and aligned face side by side
        
        Args:
            original: Original face crop
            aligned: Aligned face
            src_landmarks: Original landmarks for visualization
        
        Returns:
            Combined visualization image
        """
        # Resize original to match aligned for comparison
        original_resized = cv2.resize(original, self.output_size)
        
        # Draw landmarks on both
        if src_landmarks is not None:
            # Transform src landmarks to resized space
            h_orig, w_orig = original.shape[:2]
            scale_x = self.output_size[0] / w_orig
            scale_y = self.output_size[1] / h_orig
            
            for (x, y) in src_landmarks:
                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)
                cv2.circle(original_resized, (x_scaled, y_scaled), 2, (0, 255, 0), -1)
        
        # Draw reference landmarks on aligned
        for (x, y) in self.reference_landmarks:
            cv2.circle(aligned, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Concatenate horizontally
        combined = np.hstack([original_resized, aligned])
        
        # Add labels
        cv2.putText(combined, "Original", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Aligned", (self.output_size[0] + 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return combined


def align_face_simple(
    image: np.ndarray,
    bbox: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
    margin: float = 0.2
) -> np.ndarray:
    """
    Simple face alignment using only bounding box (no landmarks)
    
    This is a fallback when landmarks are not available.
    Less accurate than landmark-based alignment.
    
    Args:
        image: Input image
        bbox: Face bounding box [x1, y1, x2, y2]
        output_size: Output size
        margin: Extra margin around face (0.2 = 20%)
    
    Returns:
        Cropped and resized face
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Calculate width and height
    w = x2 - x1
    h = y2 - y1
    
    # Add margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(image.shape[1], x2 + margin_w)
    y2 = min(image.shape[0], y2 + margin_h)
    
    # Crop and resize
    face_crop = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, output_size)
    
    return face_resized


if __name__ == "__main__":
    """Test face alignment"""
    print("=" * 80)
    print("Face Alignment Test")
    print("=" * 80)
    
    # Create test image
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Simulate detected landmarks (slightly rotated face)
    # In a real scenario, these come from the face detector
    detected_landmarks = np.array([
        [250, 180],  # left eye
        [350, 175],  # right eye
        [300, 240],  # nose
        [270, 300],  # left mouth
        [330, 295],  # right mouth
    ], dtype=np.float32)
    
    # Draw original landmarks on image
    for (x, y) in detected_landmarks:
        cv2.circle(test_img, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Initialize aligner
    aligner = FaceAligner(output_size=(112, 112))
    
    print(f"\nReference landmarks (112x112):")
    print(aligner.reference_landmarks)
    
    print(f"\nDetected landmarks:")
    print(detected_landmarks)
    
    # Perform alignment
    aligned_face = aligner.align(test_img, detected_landmarks)
    
    print(f"\nAligned face shape: {aligned_face.shape}")
    
    # Create visualization
    bbox = [200, 150, 400, 350]  # Approximate bbox
    face_crop = test_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    vis = aligner.visualize_alignment(face_crop, aligned_face, detected_landmarks - [bbox[0], bbox[1]])
    
    print("\n" + "=" * 80)
    print("Alignment Process:")
    print("=" * 80)
    print("""
    1. INPUT: Raw face image with detected landmarks
       - Landmarks from detector (eyes, nose, mouth)
       - Face may be rotated, scaled, or off-center
    
    2. TRANSFORMATION: Similarity transform (rotation + translation + scaling)
       - Aligns eyes horizontally
       - Centers the face
       - Scales to 112x112 (standard for face recognition)
    
    3. OUTPUT: Normalized face
       - Consistent pose across all faces
       - Ready for embedding extraction
       - Critical for recognition accuracy
    
    Why alignment matters:
    - Face recognition models are trained on aligned faces
    - Alignment reduces pose variation
    - Improves feature extraction and matching
    - 10-20% accuracy improvement over unaligned faces
    
    Standard sizes:
    - 112x112: Most common for face recognition
    - 224x224: For models pretrained on ImageNet
    - 96x96: Lightweight models
    """)
