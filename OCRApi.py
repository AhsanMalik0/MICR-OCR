import os
import logging
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
# from werkzeug.utils import secure_filename

from MICR_ import load_model, prediction


# CONFIG
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
API_KEY = os.getenv("API_KEY", "change_this_in_production")

# APP SETUP
app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

CORS(
    app,
    resources={r"/*": {"origins": "*"}},  # Restrict in production
)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["60 per minute"]
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# LOAD MODEL
try:
    model = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.exception("Failed to load model")
    raise RuntimeError(f"Model loading failed: {str(e)}")


# HELPERS
def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def validate_api_key():
    api_key = request.headers.get("x-api-key")

    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    return None


def process_image(file_storage):
    # Safely process uploaded image.
    if not file_storage:
        raise ValueError("No file provided")

    filename = secure_filename(file_storage.filename)

    if filename == "":
        raise ValueError("Empty filename")

    if not allowed_file(filename):
        raise ValueError("Unsupported file type")

    try:
        image = Image.open(file_storage.stream)
        # Security check
        image.verify()
        file_storage.stream.seek(0)
        image = Image.open(file_storage.stream).convert("RGB")
        image_np = np.array(image)
        return image_np

    except UnidentifiedImageError:
        raise ValueError("Invalid image file")

    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

def run_prediction(mode):
    # Shared prediction handler.
    auth_error = validate_api_key()
    if auth_error:
        return auth_error

    if "image" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    file = request.files["image"]

    try:
        # image = process_image(file)

        result = prediction(model, image, mode=mode)

        return jsonify({
            "success": True,
            "mode": mode,
            "result": result
        }), 200

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


# ROUTES
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    }), 200


@app.route("/ready", methods=["GET"])
def readiness():
    if model is None:
        return jsonify({
            "status": "model_not_loaded"
        }), 503

    return jsonify({
        "status": "ready"
    }), 200


@app.route("/ocr", methods=["POST"])
@limiter.limit("20/minute")
def ocr():
    return run_prediction("OCR")


@app.route("/micr", methods=["POST"])
@limiter.limit("20/minute")
def micr():
    return run_prediction("MICR")


@app.route("/date-amount", methods=["POST"])
@limiter.limit("20/minute")
def date_amount():
    return run_prediction("DM")


# ERROR HANDLERS
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "success": False,
        "error": "File too large"
    }), 413


@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify({
        "success": False,
        "error": "Rate limit exceeded"
    }), 429


@app.errorhandler(Exception)
def global_exception_handler(e):
    logger.exception("Unhandled exception")

    return jsonify({
        "success": False,
        "error": "Unexpected server error"
    }), 500


# MAIN
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )