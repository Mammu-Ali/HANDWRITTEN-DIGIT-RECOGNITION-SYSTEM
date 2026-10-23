# Handwritten Digit Recognition ✍️

A beginner-friendly deep learning web application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset, served with Flask and an interactive web interface.

## Tech Stack

- **Backend**: Python 3.12+, Flask
- **Machine Learning**: TensorFlow / Keras, NumPy, Pillow
- **Frontend**: HTML5, CSS3, Bootstrap 5, Vanilla JavaScript
- **Testing**: pytest

---

## Project Structure

```text
Handwritten Digit Recognition/
├── .venv/                   # Python virtual environment
├── dataset/                 # User sample digit images partitioned by digit (0-9)
├── samples/                 # 10 pre-loaded MNIST sample digits for gallery testing
├── static/
│   ├── css/
│   │   └── style.css        # Custom CSS styling (responsive Bootstrap theme)
│   └── js/
│       └── script.js        # Frontend JavaScript (canvas, upload, gallery, history, CSV)
├── templates/
│   ├── index.html           # Main responsive prediction UI
│   └── model_info.html      # Model architecture & evaluation dashboard
├── tests/
│   ├── __init__.py          # Test package marker
│   ├── test_app.py          # Pytest suite for app routes & health check
│   ├── test_canvas.py       # Canvas interaction, scaling & inference tests (Phase 9)
│   ├── test_csv_export.py   # Prediction history CSV export & escaping tests (Phase 15/16)
│   ├── test_enhanced_results.py # Top-3 rankings & probability consistency tests (Phase 10)
│   ├── test_gallery.py      # Sample digit gallery & prediction tests (Phase 13/16)
│   ├── test_history.py      # Browser session prediction history tests (Phase 12)
│   ├── test_integration.py  # End-to-end integration & dynamic inference tests (Phase 6)
│   ├── test_model.py        # Model loading & shape tests (Phase 2)
│   ├── test_model_info.py   # Model information & evaluation artifact tests (Phase 14/16)
│   ├── test_preprocessing.py# Preprocessing & inference tests (Phase 3)
│   └── test_preprocessing_visualization.py # Preprocessing pipeline visualization tests (Phase 11)
├── .gitignore               # Git ignore rules for venv, caches, & bytecode
├── app.py                   # Flask web application & REST API
├── digit_model.keras        # Trained CNN model weights (Phase 2)
├── evaluation_metrics.json  # Genuine MNIST evaluation results, confusion matrix & metrics
├── generate_evaluation_metrics.py # Script generating evaluation metrics artifact
├── image_processing.py      # Preprocessing, normalization & centering (Phase 3)
├── predictor.py             # Model inference & probability scoring (Phase 3)
├── README.md                # Project documentation & instructions
├── requirements.txt         # Pinned project dependencies
└── train_model.py           # MNIST CNN training script (Phase 2)
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git (optional)

### 2. Set Up Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (Phase 2)
```powershell
python train_model.py
```
Outputs `digit_model.keras` with:
- **Test Accuracy**: 99.10%
- **Test Loss**: 0.0262

### 5. Run the Flask Server
```powershell
python app.py
```
Open your browser and navigate to:
- **Recognizer Interface**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **Model Architecture & Evaluation Dashboard**: [http://127.0.0.1:5000/model-info](http://127.0.0.1:5000/model-info)
- **Health Check**: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)
- **Sample Metadata API**: [http://127.0.0.1:5000/api/samples](http://127.0.0.1:5000/api/samples)

### 6. Prediction API

#### `POST /predict`
Accepts either `multipart/form-data` with an image file or `application/json` with a base64 Data URL.

**Request (JSON):**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAA..."
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "predicted_digit": 7,
  "confidence": 0.9985,
  "probabilities": {
    "0": 0.0,
    "1": 0.0,
    "2": 0.0,
    "3": 0.0,
    "4": 0.0,
    "5": 0.0,
    "6": 0.0,
    "7": 0.9985,
    "8": 0.0,
    "9": 0.0001
  },
  "top_three": [
    {"digit": 7, "probability": 0.9985},
    {"digit": 1, "probability": 0.0001},
    {"digit": 2, "probability": 0.0}
  ],
  "preprocessing": {
    "grayscale_preview": "data:image/png;base64,...",
    "tensor_preview": "data:image/png;base64,...",
    "steps": [
      "Converted color and transparency to grayscale.",
      "Normalized background polarity to match MNIST format.",
      "Cropped bounding box and resized to 20×20 preserving aspect ratio.",
      "Centered in 28×28 field using Center of Mass translation.",
      "Normalized pixel values to [0.0, 1.0] for CNN input."
    ]
  }
}
```

### 7. Run Automated Tests
```powershell
pytest -v
```

---

## Development Roadmap

- [x] **Phase 1: Project Setup & Baseline Web App**
  - Project directory structure & virtual environment
  - Dependencies installed & configured
  - Minimal Flask app with Bootstrap 5 responsive UI
  - `/health` endpoint & unit test verification
- [x] **Phase 2: Model Training & Evaluation**
  - Download & verify MNIST dataset (labels 0–9, 60,000 train / 10,000 test)
  - Normalize pixel values and reshape tensors to `(28, 28, 1)`
  - Build and train CNN model (`Conv2D`, `MaxPooling2D`, `Dropout`, `Dense`)
  - Evaluate on held-out test set (Achieved **99.10%** accuracy, **0.0262** loss)
  - Save trained model weights as `digit_model.keras`
  - Verify model reload and 10-class probability distribution
- [x] **Phase 3: Image Preprocessing & Prediction Pipeline**
  - Reusable image preprocessing module (`image_processing.py`)
  - Format validation (PNG, JPG, JPEG) and corrupt byte rejection
  - Automatic background polarity detection (light-on-dark & dark-on-light) without blind inversion
  - Digit bounding box detection, aspect-ratio preserving resize (20x20), and center-of-mass centering (28x28)
  - Blank and negligible-noise rejection (`BlankImageError`)
  - Inference engine (`predictor.py`) returning predicted digit, confidence, and 10-class probability distribution
- [x] **Phase 4: Flask Backend & Prediction API**
  - Model preloaded once during application initialization
  - `POST /predict` supporting both multipart uploads and base64 JSON
  - Upload size limit enforcement (2MB max) with 413 error handler
  - Strict input validation with descriptive HTTP status codes (400, 404, 405, 413, 500)
  - Information disclosure protection (no stack traces or local paths exposed)
  - Comprehensive integration test suite (27 passing tests)
- [x] **Phase 5: Responsive Frontend & Prediction Interface**
  - Minimal, accessible UI with white/light-neutral background and restrained blue accent
  - Drag-and-drop file upload zone + file browser button (PNG, JPG, JPEG up to 2MB)
  - Image preview with file metadata before submission
  - Functional Predict Digit and Clear buttons with loading spinner
  - Hero predicted digit display and confidence score badge
  - Complete 10-class probability distribution progress bars with winning digit highlighting
  - Responsive empty, loading, success, and error states
  - Academic mini project identification footer
- [x] **Phase 6: End-to-End API Integration & Verification**
  - Connected frontend to live `/predict` API via `fetch` and `FormData`
  - Active request lifecycle management (disabled buttons, loading indicator, preventing duplicates)
  - Strict client-side validation of API response payload before rendering
  - Seamless repeat predictions without full-page reloads
  - Complete state reset on Clear action
  - Touch and mobile responsive optimizations
  - Automated integration tests verifying dynamic non-hardcoded model inference
- [x] **Phase 7: Complete Quality Assurance & Automated Testing**
  - Expanded test coverage across home, health, PNG/JPEG predictions, errors, and blank inputs
  - Added test coverage for missing/unreadable model files and HTTP 503 fallback
  - Verified 10-class probability ranges and argmax alignment
  - Regression testing for transparent RGBA images, reverse polarity, and information disclosure suppression
  - Template structure testing verifying all required frontend DOM elements and static assets
- [x] **Phase 9: Interactive Drawing Canvas**
  - Integrated 280x280 interactive drawing canvas with Mouse, Touch, and Stylus support
  - Dynamic brush-size control slider (8px - 28px)
  - Full Undo stroke history and Clear canvas actions
  - Preserved existing drag-and-drop file upload via responsive navigation tabs
  - Coordinate scaling to eliminate drawing distortion across devices and mobile viewports
  - Unified prediction API consumption with real-time probability visualization
- [x] **Phase 10: Enhanced Prediction Results**
  - Prominent hero prediction display with top-class confidence percentage
  - Identification and ranking of the Top 3 candidate digit classes sorted by probability
  - Full probability distribution chart for all 10 digits (0 to 9) with primary prediction highlighted
  - Seamless "Predict Again" action resetting input and focus for rapid testing
  - Consistent probability formatting across both canvas drawings and file uploads
- [x] **Phase 11: Image Preprocessing Visualization**
  - Side-by-side comparison previews: Original Input, Grayscale & Polarity normalized, and the exact 28×28 Model Tensor
  - Pixelated CSS rendering for the 28×28 tensor allowing visual inspection of MNIST cell values
  - Detailed breakdown of pipeline steps (grayscale conversion, polarity detection, bounding box extraction, center of mass translation, normalization)
  - Optional collapsible section via "Show Processing Details" toggle button
  - Unified across both canvas drawings and image file uploads
  - Comprehensive automated test suite with 54 passing tests
- [x] **Phase 12: Simple Prediction History**
  - Session-scoped prediction history using browser `sessionStorage` (no databases or authentication required)
  - Records predicted digit, confidence percentage, input mode (Canvas ✏️, Upload 📁, or Sample 🖼️), timestamp, and tensor preview
  - Compact responsive history card layout with active selection highlighting
  - Full inspection capability: clicking any previous history item restores the complete result view (Hero digit, Top-3 candidates, probability distribution, and preprocessing visualization)
  - Clear History action with confirmation dialog
  - Defensive data validation filtering invalid or corrupted session entries
  - Automatic FIFO history limiting (capped at 10 most recent predictions)
  - Comprehensive automated test suite with 59 passing tests
- [x] **Phase 13: Pre-loaded MNIST Sample Digit Gallery**
  - Integrated Sample Gallery tab containing 10 pre-loaded MNIST digits (0–9)
  - Ground truth label clearly separated from dynamic model prediction
  - Image serving endpoint `/samples/<filename>` and JSON metadata endpoint `/api/samples`
  - Seamless switching between Drawing Canvas, File Upload, and Sample Gallery
- [x] **Phase 14: Model Architecture & Evaluation Dashboard**
  - Dedicated `/model-info` page presenting genuine evaluation results on the official 10,000-image MNIST test set
  - Accurate layer breakdown, parameter count (225,034 parameters), input tensor shapes
  - Complete 10×10 Confusion Matrix with ground truth vs. predicted counts and True Positive diagonal highlights
  - Per-class precision, recall, F1-score, and sample support table
  - Training progression table across epochs (loss, accuracy, validation loss, validation accuracy)
  - Zero-latency page load using pre-computed genuine evaluation artifact (`evaluation_metrics.json`)
- [x] **Phase 15: CSV Prediction History Export**
  - Instant one-click CSV export of session prediction history
  - Defensive CSV character escaping (handles commas, quotes, special characters)
  - Comprehensive columns: ID, Timestamp, Input Mode, Predicted Digit, Confidence, Top 2nd & 3rd Candidates
  - Non-mutating operation preserving underlying session history
- [x] **Phase 16: Final Integration & Release Verification**
  - Comprehensive QA and automated verification across entire stack
  - 100% test pass rate across 70 unit and integration tests
  - End-to-end integration verified: Canvas, Upload, Gallery, Predictions, Visualizations, History, CSV Export, Model Info

---

## Known Limitations

1. **Single Digit Focus**: The model is trained on isolated MNIST digits (0–9). Multi-digit strings (e.g., "42") or equations are not segmented.
2. **Extreme Stroke Variation**: Extremely thin, fragmented, or faint pencil strokes may fall below the 0.05 energy threshold and trigger blank-canvas rejection.
3. **Session-Scoped Storage**: Prediction history is saved in browser `sessionStorage`. It is scoped to the current browser tab and clears when the tab is closed, ensuring zero database overhead.
4. **Out-of-Distribution Inputs**: Drawing non-digit doodles (such as smiley faces or letters) will still produce a softmax distribution over digits 0–9; low confidence or ambiguous top-3 scores indicate uncertainty.

---

## Demonstration Checklist (Lecturer / Examiner Walkthrough)

To deliver a reliable 5-minute demonstration of the project:

1. **Start the Application**:
   ```powershell
   python app.py
   ```
   Open `http://127.0.0.1:5000/` in Google Chrome, Edge, or Firefox.

2. **Demonstrate Drawing Canvas (Interactive AI)**:
   - Click the **Draw Digit** tab (active by default).
   - Use your mouse or touchscreen to draw a clear digit **8**.
   - Click **Predict Digit**.
   - Show the prominent hero digit **8**, high confidence score, and 10-class probability bars.
   - Click **Show Processing Details** to show the 3-step pipeline: Original &rarr; Grayscale & Polarity &rarr; 28×28 Centered Tensor.

3. **Demonstrate Sample Gallery**:
   - Switch to the **Sample Gallery** tab.
   - Click on **Digit 3** (notice the ground-truth badge).
   - Click **Predict Sample Digit** to show real-time inference on pre-loaded MNIST data.

4. **Demonstrate File Upload**:
   - Switch to the **Upload Image** tab.
   - Upload any sample image from the `samples/` directory or an external PNG/JPEG.
   - Confirm upload validation, file size handling, and prediction output.

5. **Demonstrate Prediction History & CSV Export**:
   - Scroll down to **Recent Predictions History**.
   - Click any previously predicted card to demonstrate full result re-inspection.
   - Click **Export CSV** and open the generated file to show formatted timestamps, modes, digits, and candidate confidence scores.

6. **Demonstrate Model Info Dashboard**:
   - Click **Model Info** in the top navigation bar (`/model-info`).
   - Highlight the 99.10% test accuracy, 225,034 parameters, 10×10 Confusion Matrix, and per-class precision/recall/F1 metrics.

7. **Demonstrate Automated Quality Assurance**:
   - Open a terminal and run `pytest -v` to confirm **70 tests passing** in under 9 seconds.
