"""
Streamlit Frontend for Face Recognition System

Features:
- Live video feed with real-time face recognition
- Face registration interface
- Database management
- Bounding box visualization (green=live, red=spoof)
- Identity display with confidence scores
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline import FaceRecognitionPipeline


# Page configuration
st.set_page_config(
    page_title="Adaptive Face Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #ff7f0e;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .live-indicator {
        background-color: #28a745;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .spoof-indicator {
        background-color: #dc3545;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load and cache the face recognition pipeline"""
    detector_path = os.environ.get('DETECTOR_MODEL_PATH')
    liveness_path = os.environ.get('LIVENESS_MODEL_PATH')
    recognition_path = os.environ.get('RECOGNITION_MODEL_PATH')
    
    pipeline = FaceRecognitionPipeline(
        detector_path=detector_path,
        liveness_path=liveness_path,
        recognition_path=recognition_path,
        liveness_threshold=0.7,
        recognition_threshold=0.6,
        use_onnx=True
    )
    
    return pipeline


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🎭 Adaptive Face Recognition</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        app_mode = st.selectbox(
            "Choose Mode",
            ["Live Recognition", "Upload Image", "Register Face", "Database Management"]
        )
        
        st.markdown("---")
        st.markdown("## 📊 System Info")
        
        # Liveness threshold
        liveness_threshold = st.slider("Liveness Threshold", 0.0, 1.0, 0.7, 0.05)
        
        # Recognition threshold
        recognition_threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.6, 0.05)
        
        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.info("""
        **Features:**
        - Real-time face detection
        - Liveness detection (anti-spoofing)
        - Face recognition with AdaFace
        - Quality-adaptive matching
        
        **Technologies:**
        - PyTorch & ONNX Runtime
        - SCRFD/RetinaFace detector
        - MobileFaceNet backbone
        - Streamlit UI
        """)
    
    # Load pipeline
    try:
        pipeline = load_pipeline()
        pipeline.liveness_threshold = liveness_threshold
        pipeline.recognition_threshold = recognition_threshold
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()
    
    # Main content based on mode
    if app_mode == "Live Recognition":
        live_recognition_mode(pipeline)
    elif app_mode == "Upload Image":
        upload_image_mode(pipeline)
    elif app_mode == "Register Face":
        register_face_mode(pipeline)
    elif app_mode == "Database Management":
        database_management_mode(pipeline)


def live_recognition_mode(pipeline):
    """Live video recognition mode"""
    st.markdown('<p class="sub-header">📹 Live Recognition</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Video Feed")
        
        # Placeholder for video
        video_placeholder = st.empty()
        
        # Camera controls
        start_camera = st.button("Start Camera")
        stop_camera = st.button("Stop Camera")
        
        if start_camera:
            st.session_state['camera_active'] = True
        if stop_camera:
            st.session_state['camera_active'] = False
        
        # Video capture
        if st.session_state.get('camera_active', False):
            cap = cv2.VideoCapture(0)
            
            frame_count = 0
            while st.session_state.get('camera_active', False):
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break
                
                # Process every 3rd frame for performance
                if frame_count % 3 == 0:
                    results = pipeline.process_frame(frame)
                    frame_annotated = pipeline.draw_results(frame, results)
                else:
                    frame_annotated = frame
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                frame_count += 1
            
            cap.release()
    
    with col2:
        st.markdown("### Statistics")
        
        # Database stats
        db = pipeline.face_database
        st.metric("Registered Identities", len(set(db['identities'])))
        st.metric("Total Embeddings", len(db['embeddings']))
        
        st.markdown("---")
        st.markdown("### Recent Detections")
        st.info("Start camera to see live detections")


def upload_image_mode(pipeline):
    """Upload and process image mode"""
    st.markdown('<p class="sub-header">📤 Upload & Recognize</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("### Recognition Results")
            
            with st.spinner("Processing..."):
                results, timings = pipeline.process_frame(image_bgr, return_debug_info=True)
                result_image = pipeline.draw_results(image_bgr, results)
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            st.image(result_rgb, use_column_width=True)
        
        # Display detailed results
        st.markdown("---")
        st.markdown("### Detailed Results")
        
        if len(results) == 0:
            st.warning("No faces detected")
        else:
            for i, result in enumerate(results):
                with st.expander(f"Face {i+1}: {result.get('identity', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if result['is_live']:
                            st.markdown('<div class="live-indicator">✓ LIVE</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="spoof-indicator">✗ SPOOF</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Liveness Score", f"{result['liveness_score']:.3f}")
                    
                    with col3:
                        st.metric("Similarity", f"{result.get('similarity', 0.0):.3f}")
        
        # Timing information
        if timings:
            st.markdown("---")
            st.markdown("### Performance")
            
            cols = st.columns(len(timings))
            for col, (step, time_ms) in zip(cols, timings.items()):
                with col:
                    st.metric(step.title(), f"{time_ms:.1f}ms")


def register_face_mode(pipeline):
    """Face registration mode"""
    st.markdown('<p class="sub-header">➕ Register New Face</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Face Image")
        
        uploaded_file = st.file_uploader("Choose a clear face image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, width=400)
    
    with col2:
        st.markdown("### Identity Information")
        
        identity = st.text_input("Name/ID", placeholder="Enter person's name")
        
        metadata_json = st.text_area("Metadata (JSON)", placeholder='{"department": "Engineering", "employee_id": "12345"}')
        
        register_button = st.button("Register Face", type="primary")
        
        if register_button:
            if not identity:
                st.error("Please enter a name/ID")
            elif uploaded_file is None:
                st.error("Please upload an image")
            else:
                # Read image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Parse metadata
                import json
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}
                
                # Register
                with st.spinner("Registering face..."):
                    success = pipeline.register_face(image_bgr, identity, metadata)
                
                if success:
                    st.success(f"✓ Successfully registered {identity}!")
                else:
                    st.error("Registration failed. Ensure face is visible and live.")


def database_management_mode(pipeline):
    """Database management mode"""
    st.markdown('<p class="sub-header">🗄️ Database Management</p>', unsafe_allow_html=True)
    
    db = pipeline.face_database
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Unique Identities", len(set(db['identities'])))
    
    with col2:
        st.metric("Total Embeddings", len(db['embeddings']))
    
    with col3:
        st.metric("Avg. Embeddings/Identity", 
                 f"{len(db['embeddings']) / max(len(set(db['identities'])), 1):.1f}")
    
    st.markdown("---")
    
    # Identity list
    st.markdown("### Registered Identities")
    
    if len(db['identities']) == 0:
        st.info("No faces registered yet. Use 'Register Face' mode to add identities.")
    else:
        identities = set(db['identities'])
        
        for identity in sorted(identities):
            count = db['identities'].count(identity)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{identity}**")
            
            with col2:
                st.write(f"{count} embedding(s)")
            
            with col3:
                if st.button("Delete", key=f"del_{identity}"):
                    # Delete identity
                    indices = [i for i, id_name in enumerate(db['identities']) if id_name == identity]
                    for idx in sorted(indices, reverse=True):
                        del db['embeddings'][idx]
                        del db['identities'][idx]
                        del db['metadata'][idx]
                    
                    st.success(f"Deleted {identity}")
                    st.rerun()
    
    st.markdown("---")
    
    # Clear database button
    if st.button("⚠️ Clear Entire Database", type="secondary"):
        if st.checkbox("I understand this will delete all registered faces"):
            pipeline.face_database = {
                'embeddings': [],
                'identities': [],
                'metadata': []
            }
            st.success("Database cleared")
            st.rerun()


if __name__ == "__main__":
    main()
