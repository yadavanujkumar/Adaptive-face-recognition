# Deployment Guide

This guide covers deploying the Adaptive Face Recognition system in various scenarios.

## Table of Contents
1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Edge Device Deployment](#edge-device-deployment)
5. [Production Checklist](#production-checklist)

---

## 1. Local Deployment

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download/Train Models

Option A: Use pre-trained models
```bash
# Download pre-trained models
mkdir -p weights
# Place ONNX models in weights/ directory:
# - weights/scrfd_10g.onnx (face detector)
# - weights/liveness_net.onnx (anti-spoofing)
# - weights/mobilefacenet.onnx (face recognition)
```

Option B: Train your own models
```bash
# See TRAINING.md for detailed instructions
python train_recognition.py
python train_liveness.py
python src/utils/export_onnx.py
```

### Step 3: Configure System

Edit `configs/config.yaml`:
```yaml
models:
  detector:
    path: "weights/scrfd_10g.onnx"
  liveness:
    path: "weights/liveness_net.onnx"
  recognition:
    path: "weights/mobilefacenet.onnx"
```

### Step 4: Start Services

Terminal 1 - API Server:
```bash
python src/api/server.py
# Access at http://localhost:8000
```

Terminal 2 - Frontend:
```bash
streamlit run src/frontend/app.py
# Opens in browser at http://localhost:8501
```

### Step 5: Test System

```bash
# Run tests
python test_system.py

# Test with sample image
curl -X POST http://localhost:8000/api/recognize \
  -F "image=@sample_face.jpg"
```

---

## 2. Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Start services
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t adaptive-face-recognition .

# Run container
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/weights:/app/weights \
  adaptive-face-recognition
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./weights:/app/weights
      - ./data:/app/data
    environment:
      - DETECTOR_MODEL_PATH=/app/weights/scrfd_10g.onnx
      - LIVENESS_MODEL_PATH=/app/weights/liveness_net.onnx
      - RECOGNITION_MODEL_PATH=/app/weights/mobilefacenet.onnx
  
  frontend:
    build: .
    command: streamlit run src/frontend/app.py
    ports:
      - "8501:8501"
    depends_on:
      - api
  
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
```

Run with docker-compose:
```bash
docker-compose up -d
```

---

## 3. Cloud Deployment

### AWS Deployment

#### Using EC2

1. Launch EC2 instance (t3.medium or larger)
2. Install dependencies:
```bash
sudo apt update
sudo apt install -y python3-pip
git clone https://github.com/yadavanujkumar/Adaptive-face-recognition.git
cd Adaptive-face-recognition
pip install -r requirements.txt
```

3. Configure security groups:
   - Port 8000 (API)
   - Port 8501 (Frontend)

4. Start services:
```bash
# Use screen or tmux for persistent sessions
screen -S api
python src/api/server.py

# Detach: Ctrl+A, D
# Start frontend in another screen
screen -S frontend
streamlit run src/frontend/app.py
```

#### Using Lambda + API Gateway

For serverless deployment:
1. Package application with dependencies
2. Create Lambda function
3. Configure API Gateway
4. Deploy with SAM or Serverless Framework

### Google Cloud Deployment

#### Using Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/[PROJECT-ID]/face-recognition

# Deploy to Cloud Run
gcloud run deploy face-recognition \
  --image gcr.io/[PROJECT-ID]/face-recognition \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Deployment

#### Using App Service

```bash
# Create resource group
az group create --name FaceRecognitionRG --location eastus

# Create App Service plan
az appservice plan create --name FaceRecognitionPlan \
  --resource-group FaceRecognitionRG --sku B1 --is-linux

# Deploy application
az webapp create --resource-group FaceRecognitionRG \
  --plan FaceRecognitionPlan --name face-recognition-app \
  --runtime "PYTHON|3.9"
```

---

## 4. Edge Device Deployment

### Raspberry Pi

System Requirements:
- Raspberry Pi 4 (4GB+ RAM recommended)
- Camera module or USB webcam

Installation:
```bash
# Update system
sudo apt update && sudo apt upgrade

# Install dependencies
sudo apt install -y python3-opencv python3-numpy
pip3 install -r requirements.txt

# Optimize for ARM
pip3 install onnxruntime-arm  # ARM-optimized runtime
```

Performance Optimization:
- Use INT8 quantized models
- Reduce input resolution
- Process every N-th frame
- Use hardware acceleration (if available)

### NVIDIA Jetson

```bash
# Install JetPack SDK
# Then install dependencies
pip3 install -r requirements.txt

# Use TensorRT for optimization
pip3 install tensorrt

# Convert ONNX to TensorRT
trtexec --onnx=weights/mobilefacenet.onnx \
  --saveEngine=weights/mobilefacenet.trt \
  --fp16
```

### Intel NUC / Edge PC

```bash
# Install OpenVINO for Intel optimization
pip install openvino-dev

# Convert models to OpenVINO IR format
mo --input_model weights/mobilefacenet.onnx \
   --output_dir weights/openvino/
```

---

## 5. Production Checklist

### Security
- [ ] Enable HTTPS/TLS
- [ ] Implement authentication (JWT, OAuth)
- [ ] Rate limiting on API endpoints
- [ ] Input validation and sanitization
- [ ] Secure storage of face embeddings
- [ ] Regular security audits

### Performance
- [ ] Use ONNX Runtime with optimizations
- [ ] Enable model quantization (INT8)
- [ ] Configure proper batch sizes
- [ ] Set up load balancing
- [ ] Implement caching strategies
- [ ] Monitor and optimize bottlenecks

### Reliability
- [ ] Error handling and logging
- [ ] Health check endpoints
- [ ] Automatic restarts on failure
- [ ] Database backups
- [ ] Monitoring and alerts
- [ ] Fallback mechanisms

### Scalability
- [ ] Horizontal scaling (multiple instances)
- [ ] Vector database optimization
- [ ] CDN for static assets
- [ ] Asynchronous processing
- [ ] Queue system for bulk processing

### Compliance
- [ ] GDPR compliance (data privacy)
- [ ] Biometric data regulations
- [ ] User consent management
- [ ] Data retention policies
- [ ] Audit trails
- [ ] Documentation

### Monitoring
- [ ] Application metrics (Prometheus)
- [ ] Logging (ELK stack)
- [ ] Error tracking (Sentry)
- [ ] Performance monitoring (New Relic/DataDog)
- [ ] Uptime monitoring
- [ ] Resource utilization

### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Load testing
- [ ] Security testing
- [ ] Cross-browser testing (frontend)
- [ ] Mobile responsiveness

---

## Environment Variables

For production deployment, use environment variables:

```bash
# Model paths
export DETECTOR_MODEL_PATH=/path/to/detector.onnx
export LIVENESS_MODEL_PATH=/path/to/liveness.onnx
export RECOGNITION_MODEL_PATH=/path/to/recognition.onnx

# Database
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# API settings
export API_HOST=0.0.0.0
export API_PORT=8000
export SECRET_KEY=your-secret-key

# Feature flags
export ENABLE_LIVENESS=true
export ENABLE_LOGGING=true
```

---

## Performance Tuning

### CPU Optimization

```python
# Use all CPU cores
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4  # Adjust based on CPU
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    model_path,
    session_options,
    providers=['CPUExecutionProvider']
)
```

### GPU Optimization

```python
# Use CUDA
session = ort.InferenceSession(
    model_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### Model Quantization

```python
from onnxruntime.quantization import quantize_dynamic

# Quantize model to INT8
quantize_dynamic(
    model_input='weights/mobilefacenet.onnx',
    model_output='weights/mobilefacenet_int8.onnx',
    weight_type=QuantType.QInt8
)
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use model quantization
   - Process frames sequentially

2. **Slow Inference**
   - Check ONNX Runtime optimizations
   - Use GPU if available
   - Reduce input resolution

3. **Poor Recognition Accuracy**
   - Ensure proper face alignment
   - Check lighting conditions
   - Retrain with domain-specific data

4. **High False Positives (Liveness)**
   - Adjust liveness threshold
   - Collect more spoof samples
   - Retrain liveness model

---

## Support

For deployment assistance:
- GitHub Issues: Report bugs and issues
- Documentation: Refer to README.md and TRAINING.md
- Community: Join discussions

---

## Conclusion

This deployment guide covers various scenarios from local testing to production deployment. Choose the approach that best fits your infrastructure and requirements.

For questions or improvements, please contribute to the project!
