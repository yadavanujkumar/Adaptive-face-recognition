"""
Simple example demonstrating the complete face recognition pipeline

This script shows:
1. How to initialize the pipeline
2. How to register faces
3. How to perform recognition
4. How to interpret results

Note: Requires models to be trained/downloaded first
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from pipeline import FaceRecognitionPipeline


def demo_pipeline():
    """Demonstrate the face recognition pipeline"""
    
    print("=" * 80)
    print("Adaptive Face Recognition - Simple Demo")
    print("=" * 80)
    
    # Step 1: Initialize Pipeline
    print("\n[1] Initializing pipeline...")
    pipeline = FaceRecognitionPipeline(
        detector_path=None,  # Will use fallback detector for demo
        liveness_path=None,   # Set to your ONNX model path
        recognition_path=None, # Set to your ONNX model path
        liveness_threshold=0.7,
        recognition_threshold=0.6,
    )
    print("✓ Pipeline initialized")
    
    # Step 2: Register Faces
    print("\n[2] Registering sample faces...")
    
    # In a real scenario, load actual images
    # For demo, we'll create synthetic images
    sample_image1 = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.circle(sample_image1, (320, 240), 80, (100, 100, 100), -1)
    
    sample_image2 = np.ones((480, 640, 3), dtype=np.uint8) * 180
    cv2.circle(sample_image2, (320, 240), 85, (120, 120, 120), -1)
    
    # Register faces
    success1 = pipeline.register_face(sample_image1, "Alice", {"department": "Engineering"})
    success2 = pipeline.register_face(sample_image2, "Bob", {"department": "Marketing"})
    
    if success1:
        print("✓ Registered: Alice")
    if success2:
        print("✓ Registered: Bob")
    
    # Step 3: Check Database
    print("\n[3] Database statistics:")
    db = pipeline.face_database
    print(f"  Registered identities: {len(set(db['identities']))}")
    print(f"  Total embeddings: {len(db['embeddings'])}")
    print(f"  Identities: {set(db['identities'])}")
    
    # Step 4: Perform Recognition
    print("\n[4] Performing recognition...")
    
    # Create a test image (simulating a camera capture)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 190
    cv2.circle(test_image, (320, 240), 82, (110, 110, 110), -1)
    
    # Process the frame
    results = pipeline.process_frame(test_image)
    
    print(f"  Detected {len(results)} face(s)")
    
    for i, result in enumerate(results):
        print(f"\n  Face {i+1}:")
        print(f"    Liveness: {'LIVE' if result['is_live'] else 'SPOOF'} ({result['liveness_score']:.3f})")
        print(f"    Identity: {result.get('identity', 'Unknown')}")
        print(f"    Similarity: {result.get('similarity', 0.0):.3f}")
        print(f"    Bbox: {result['bbox']}")
    
    # Step 5: Visualize Results
    print("\n[5] Generating visualization...")
    output_image = pipeline.draw_results(test_image, results)
    
    # Save output
    cv2.imwrite('output_demo.jpg', output_image)
    print("✓ Saved visualization to 'output_demo.jpg'")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("""
Next Steps:
1. Train or download models (see TRAINING.md)
2. Place ONNX models in weights/ directory
3. Update pipeline initialization with model paths
4. Run with real camera: python examples/demo.py --mode webcam
5. Start API server: python src/api/server.py
6. Launch frontend: streamlit run src/frontend/app.py
    """)


def webcam_demo():
    """Demo with live webcam (requires models)"""
    
    print("=" * 80)
    print("Webcam Face Recognition Demo")
    print("=" * 80)
    print("Press 'q' to quit, 'r' to register face")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 3rd frame for performance
        if frame_count % 3 == 0:
            results = pipeline.process_frame(frame)
            frame = pipeline.draw_results(frame, results)
        
        # Display frame
        cv2.imshow('Face Recognition', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Register mode
            identity = input("Enter identity name: ")
            success = pipeline.register_face(frame, identity)
            if success:
                print(f"✓ Registered {identity}")
            else:
                print("✗ Registration failed")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition Demo")
    parser.add_argument(
        '--mode',
        type=str,
        default='pipeline',
        choices=['pipeline', 'webcam'],
        help='Demo mode: pipeline (simple) or webcam (live)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'pipeline':
        demo_pipeline()
    elif args.mode == 'webcam':
        webcam_demo()
