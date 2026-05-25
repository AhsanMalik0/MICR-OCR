# MICR Cheque OCR API

A production-ready Flask-based OCR API for cheque processing using:

- YOLO Object Detection
- MICR OCR
- Date Extraction
- Amount Extraction
- Authorizer Detection

This API detects and extracts:

- MICR Code
- Cheque Number
- Routing Number
- Account Number
- Date
- Amount
- Maker / Authorizer Name

---

# Features

- MICR detection
- OCR extraction
- Date & amount recognition
- REST API endpoints
- Rate limiting
- API key security
- Production-ready Flask server
- Docker support
- Logging support
- Error handling

---

# Project Structure

```bash
project/
│
├── app.py
├── MICR_.py
├── Authorizer.py
├── classes.py
├── Date.py
├── requirements.txt
│
├── weights/
│   └── best.pt
│
├── Handwritten/
│
├── train/
│
├── Prediction/
│
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd project
```

---

## 2. Create Virtual Environment

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Install Tesseract OCR

## Ubuntu / Debian

```bash
sudo apt update
sudo apt install tesseract-ocr
```

---

## Windows

Download:

- https://github.com/UB-Mannheim/tesseract/wiki

After installation, add Tesseract to PATH.

Example:

```bash
C:\Program Files\Tesseract-OCR
```

---

# Environment Variables

Create a `.env` file:

```env
API_KEY=your_secret_api_key
```

---

# Running the API

## Development

```bash
python app.py
```

Server starts at:

```bash
http://localhost:5000
```

---

# Production Deployment

Use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

# API Authentication

All endpoints require API key.

Add header:

```http
x-api-key: your_secret_api_key
```

---

# API Endpoints

---

# 1. Health Check

## Endpoint

```http
GET /health
```

## Response

```json
{
  "status": "healthy"
}
```

---

# 2. OCR Endpoint

Extract:

- MICR
- Date
- Amount
- Maker

## Endpoint

```http
POST /ocr
```

## Form Data

| Key   | Type | Required |
|------|------|------|
| image | File | Yes |

---

## cURL Example

```bash
curl -X POST http://localhost:5000/ocr \
-H "x-api-key: your_secret_api_key" \
-F "image=@cheque.jpg"
```

---

## Response

```json
{
  "success": true,
  "mode": "OCR",
  "result": {
    "Date": "01/01/2025",
    "Cheque_Number": "123456",
    "Account_Number": "1234567890",
    "Routing_Number": "021000021",
    "Amount": "5000",
    "Maker": "John Doe"
  }
}
```

---

# 3. MICR Endpoint

Extract only:

- Cheque Number
- Account Number
- Routing Number

## Endpoint

```http
POST /micr
```

## cURL Example

```bash
curl -X POST http://localhost:5000/micr \
-H "x-api-key: your_secret_api_key" \
-F "image=@cheque.jpg"
```

---

# 4. Date & Amount Endpoint

Extract:

- Date
- Amount
- Maker

## Endpoint

```http
POST /date-amount
```

## cURL Example

```bash
curl -X POST http://localhost:5000/date-amount \
-H "x-api-key: your_secret_api_key" \
-F "image=@cheque.jpg"
```

---

# Supported Image Formats

- JPG
- JPEG
- PNG
- BMP
- TIFF

---

# API Limits

Default limits:

- 20 requests/minute per endpoint
- Max upload size: 10MB

---

# Error Responses

## Unauthorized

```json
{
  "error": "Unauthorized"
}
```

---

## Invalid File

```json
{
  "error": "Unsupported file type"
}
```

---

## File Too Large

```json
{
  "error": "File too large"
}
```

---

## Internal Error

```json
{
  "error": "Internal server error"
}
```

---

# Model Files

Place YOLO weights here:

```bash
weights/best.pt
```

---

# Prediction Outputs

Failed OCR predictions are automatically saved in:

```bash
Prediction/
```

Folders:

- MICR
- Date
- Amount
- Maker

---

# Security Features

- API Key Authentication
- Rate Limiting
- File Validation
- Secure Upload Handling
- Error Protection
- Input Validation

---

# Docker Deployment

## Build Docker Image

```bash
docker build -t micr-api .
```

---

## Run Container

```bash
docker run -p 5000:5000 \
-e API_KEY=your_secret_api_key \
micr-api
```

---

# Recommended Production Stack

- Flask
- Gunicorn
- Nginx
- Docker
- Redis
- Prometheus
- Grafana

---

# Future Improvements

- FastAPI migration
- Async inference
- GPU serving
- Batch processing
- JWT authentication
- Kubernetes deployment
- Triton inference support

---

# Troubleshooting

---

## Tesseract Not Found

Error:

```bash
tesseract is not installed
```

Fix:

Install Tesseract and add it to PATH.

---

## Model Not Found

Error:

```bash
Unable to load model
```

Fix:

Ensure model exists:

```bash
weights/best.pt
```

---

## CUDA Issues

If GPU fails:

```python
device='cpu'
```

---

# License

MIT License

---

# Author

MICR OCR API
Production-ready cheque OCR and MICR extraction system.