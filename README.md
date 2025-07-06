# DLABS - Deep Learning Laboratory

This repository is a comprehensive, beginner-friendly deep learning codebase supporting Computer Vision, Natural Language Processing, Handwritten Text Recognition, and Production-Ready ML Systems.

## 🚀 Quick Start

```bash
# Clone and setup in one command
git clone <your-repo-url> dlabs && cd dlabs
python -m venv venv && source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Test installation
python quick_test.py
```

Expected output:
```
✅ PyTorch installed successfully
✅ Computer Vision modules ready
✅ NLP modules ready  
✅ Handwriting Recognition modules ready
🎉 DLABS setup complete!
```

## 📋 Table of Contents

- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Development Workflow](#development-workflow)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+ (recommended: 3.9 or 3.10)
- CUDA 11.8+ (for GPU support, optional but recommended)
- Git

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv dlabs_env

# Activate environment
# On macOS/Linux:
source dlabs_env/bin/activate
# On Windows:
dlabs_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Core Dependencies

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Project Structure

```
dlabs/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── quick_test.py                     # Installation verification
├── setup.py                         # Package setup
├── .gitignore                        # Git ignore rules
├── .pre-commit-config.yaml          # Code formatting hooks
│
├── computer_vision/                  # Computer Vision modules
│   ├── __init__.py
│   ├── image_classification/         # Image classification models
│   ├── object_detection/            # YOLO, R-CNN implementations
│   ├── image_segmentation/          # Semantic/instance segmentation
│   ├── utils/                       # CV utility functions
│   └── examples/                    # CV example notebooks
│
├── nlp/                             # Natural Language Processing
│   ├── __init__.py
│   ├── text_classification/         # Sentiment analysis, topic classification
│   ├── language_models/             # BERT, GPT implementations
│   ├── text_generation/             # Text generation models
│   ├── utils/                       # NLP utility functions
│   └── examples/                    # NLP example notebooks
│
├── handwriting_recognition/         # Handwritten Text Recognition
│   ├── __init__.py
│   ├── ocr/                        # Optical Character Recognition
│   ├── text_detection/             # Text detection in images
│   ├── preprocessing/              # Image preprocessing for OCR
│   ├── utils/                      # OCR utility functions
│   └── examples/                   # OCR example notebooks
│
├── utils/                          # Common utilities
│   ├── __init__.py
│   ├── data_loader.py             # Data loading utilities
│   ├── model_utils.py             # Model saving/loading
│   ├── visualization.py           # Plotting and visualization
│   └── metrics.py                 # Evaluation metrics
│
├── models/                         # Pre-trained and custom models
│   ├── pretrained/                # Downloaded pre-trained models
│   ├── custom/                    # Your custom trained models
│   └── configs/                   # Model configuration files
│
├── data/                          # Dataset management
│   ├── raw/                       # Raw, unprocessed data
│   ├── processed/                 # Cleaned and preprocessed data
│   ├── external/                  # External datasets
│   └── interim/                   # Intermediate processing results
│
├── experiments/                   # Research and experimentation
│   ├── notebooks/                 # Jupyter notebooks
│   ├── scripts/                   # Experiment scripts
│   └── results/                   # Experiment results and logs
│
├── production/                    # Production deployment
│   ├── api/                       # FastAPI application
│   ├── docker/                    # Docker configurations
│   ├── monitoring/                # Logging and monitoring
│   └── deployment/                # Deployment scripts
│
└── tests/                         # Unit and integration tests
    ├── test_computer_vision/
    ├── test_nlp/
    ├── test_handwriting/
    └── test_utils/
```

## 🎯 Usage Examples

### Computer Vision - Image Classification

```python
from computer_vision.image_classification import ImageClassifier
from PIL import Image

# Load pre-trained model
classifier = ImageClassifier(model_name='resnet50', pretrained=True)

# Classify an image
image = Image.open('path/to/your/image.jpg')
prediction = classifier.predict(image)
print(f"Predicted class: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.2f}")
```

### Natural Language Processing - Sentiment Analysis

```python
from nlp.text_classification import SentimentAnalyzer

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer(model_name='bert-base-uncased')

# Analyze sentiment
text = "I love this deep learning framework!"
result = analyzer.analyze(text)
print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['score']:.2f}")
```

### Handwriting Recognition - OCR

```python
from handwriting_recognition.ocr import HandwritingOCR
from PIL import Image

# Initialize OCR model
ocr = HandwritingOCR(model_name='trocr-base-handwritten')

# Extract text from handwritten image
image = Image.open('handwritten_text.jpg')
extracted_text = ocr.extract_text(image)
print(f"Extracted text: {extracted_text}")
```

### Multi-Modal Example - Document Analysis

```python
from handwriting_recognition.ocr import HandwritingOCR
from nlp.text_classification import SentimentAnalyzer
from computer_vision.utils import preprocess_image

# Complete pipeline: Image → Text → Analysis
def analyze_handwritten_document(image_path):
    # Step 1: Preprocess image
    image = preprocess_image(image_path)
    
    # Step 2: Extract text using OCR
    ocr = HandwritingOCR()
    text = ocr.extract_text(image)
    
    # Step 3: Analyze extracted text
    analyzer = SentimentAnalyzer()
    sentiment = analyzer.analyze(text)
    
    return {
        'extracted_text': text,
        'sentiment': sentiment['sentiment'],
        'confidence': sentiment['score']
    }

# Usage
result = analyze_handwritten_document('document.jpg')
print(result)
```

## 🔧 Development Workflow

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code formatting
black --check .
flake8 .
```

### Creating a New Module

1. Create module directory in appropriate section (cv/nlp/handwriting)
2. Add `__init__.py` file
3. Implement your model class
4. Add example usage in `examples/` directory
5. Write unit tests in `tests/`
6. Update documentation

### Code Style Guidelines

- Use **Black** for code formatting
- Follow **PEP 8** naming conventions
- Add **type hints** for function parameters
- Write **docstrings** for all public functions
- Keep functions **small and focused**

## 🚀 Production Deployment

### API Server Setup

```bash
# Navigate to production directory
cd production/api

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Test API endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}'
```

### Docker Deployment

```bash
# Build Docker image
docker build -t dlabs:latest .

# Run container
docker run -p 8000:8000 dlabs:latest

# Docker Compose (with GPU support)
docker-compose up -d
```

### Model Serving Best Practices

1. **Model Versioning**: Use MLflow for model tracking
2. **Caching**: Implement Redis for model caching
3. **Monitoring**: Set up Prometheus + Grafana
4. **Load Balancing**: Use nginx for multiple instances
5. **Health Checks**: Implement `/health` endpoint

## 🔍 Troubleshooting

### Common Installation Issues

**Issue**: `torch` installation fails
```bash
# Solution: Install specific version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

**Issue**: CUDA not detected
```bash
# Check CUDA installation
nvidia-smi
# Reinstall PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue**: Out of memory errors
```python
# Solution: Reduce batch size or use gradient accumulation
# In your training script:
batch_size = 16  # Reduce from 32
torch.cuda.empty_cache()  # Clear GPU memory
```

### Performance Optimization

1. **GPU Utilization**: Monitor with `nvidia-smi`
2. **Memory Management**: Use `torch.cuda.empty_cache()`
3. **Data Loading**: Increase `num_workers` in DataLoader
4. **Mixed Precision**: Use `torch.cuda.amp` for faster training

### Debugging Tips

```python
# Enable detailed error messages
import torch
torch.autograd.set_detect_anomaly(True)

# Profile your code
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your code here
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 📚 Learning Resources

### Recommended Reading
- [Deep Learning with PyTorch](https://pytorch.org/tutorials/)
- [Hugging Face Course](https://huggingface.co/course/chapter1/1)
- [Computer Vision with PyTorch](https://pytorch.org/vision/stable/index.html)

### Useful Datasets
- **Computer Vision**: CIFAR-10, ImageNet, COCO
- **NLP**: IMDB Reviews, AG News, SQuAD
- **Handwriting**: IAM Database, RIMES, CVL

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup for Contributors

```bash
# Clone your fork
git clone https://github.com/yourusername/dlabs.git
cd dlabs

# Add upstream remote
git remote add upstream https://github.com/original/dlabs.git

# Install in development mode
pip install -e .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- Hugging Face for transformers library
- OpenCV community for computer vision tools
- All contributors and the open-source community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Geoelycom/dlabs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Geoelycom/dlabs/discussions)
- **Email**: ekenimohelyanpheta@gmail.com

---

**Happy Deep Learning! 🧠✨**

> "The best way to learn deep learning is by doing. Start with the examples, experiment, and build amazing things!"
