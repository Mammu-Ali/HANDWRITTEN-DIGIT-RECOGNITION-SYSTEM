/**
 * Handwritten Digit Recognition - Complete Frontend Controller
 * Features:
 * 1. Enhanced Canvas: Mouse, Touch, Stylus, Brush/Eraser modes, Undo, Redo, Keyboard Shortcuts.
 * 2. Smart Digit Prediction: Manual and Auto-predict (~700ms stroke debounce).
 * 3. Interactive Probability Chart: Chart.js bar chart with winning class highlight.
 * 4. Image Upload & Validation: Drag-and-drop, client-side & server-side validation.
 * 5. Preprocessing Inspector: 6-stage visualization with detailed educational explanations.
 * 6. Sample Digit Gallery: 10 MNIST pre-loaded samples with instant testing.
 * 7. Persistent History: SQLite backend sync + sessionStorage, search, filtering, and sorting.
 * 8. CSV Export: Secure formula-injection-safe export.
 * 9. Prediction Verification Feedback: Thumbs up/down, actual digit correction, and live stats.
 * 10. Digit Challenge Mode: Interactive learning game tracking score, attempts, and accuracy.
 * 11. Accessibility & UX: Toast alerts, dark/light theme, and focus indications.
 */

document.addEventListener("DOMContentLoaded", () => {
    // =========================================================================
    // DOM Element References
    // =========================================================================

    // --- Drawing Canvas References ---
    const canvas = document.getElementById("drawing-canvas");
    const ctx = canvas ? canvas.getContext("2d", { willReadFrequently: true }) : null;
    const brushSizeInput = document.getElementById("brush-size");
    const brushSizeVal = document.getElementById("brush-size-val");
    const canvasPredictBtn = document.getElementById("canvas-predict-btn");
    const canvasPredictBtnText = document.getElementById("canvas-predict-btn-text");
    const canvasPredictSpinner = document.getElementById("canvas-predict-spinner");
    const canvasUndoBtn = document.getElementById("canvas-undo-btn");
    const canvasRedoBtn = document.getElementById("canvas-redo-btn");
    const canvasClearBtn = document.getElementById("canvas-clear-btn");
    const canvasCornerUndoBtn = document.getElementById("canvas-corner-undo-btn");
    const canvasCornerClearBtn = document.getElementById("canvas-corner-clear-btn");
    const brushDotBtns = document.querySelectorAll(".brush-dot-btn");
    const btnToolBrush = document.getElementById("btn-tool-brush");
    const btnToolEraser = document.getElementById("btn-tool-eraser");
    const autoPredictToggle = document.getElementById("auto-predict-toggle");

    // --- File Upload References ---
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const dropPrompt = document.getElementById("drop-prompt");
    const previewWrapper = document.getElementById("preview-wrapper");
    const imagePreview = document.getElementById("image-preview");
    const fileDetails = document.getElementById("file-details");
    const predictBtn = document.getElementById("predict-btn");
    const predictBtnText = document.getElementById("predict-btn-text");
    const predictSpinner = document.getElementById("predict-spinner");
    const clearBtn = document.getElementById("clear-btn");
    const apiStatusBadge = document.getElementById("api-status-badge");

    // --- Results & States References ---
    const stateEmpty = document.getElementById("state-empty");
    const stateLoading = document.getElementById("state-loading");
    const stateError = document.getElementById("state-error");
    const stateSuccess = document.getElementById("state-success");
    const errorMessage = document.getElementById("error-message");
    const resultDigit = document.getElementById("result-digit");
    const resultConfidence = document.getElementById("result-confidence");
    const resultStatusBadge = document.getElementById("result-status-badge");
    const topThreeList = document.getElementById("top-three-list");
    const probabilitiesList = document.getElementById("probabilities-list");
    const verticalChartArea = document.getElementById("vertical-chart-area");
    const verticalChartXLabels = document.getElementById("vertical-chart-x-labels");
    const predictAgainBtn = document.getElementById("predict-again-btn");
    const predictionChartCanvas = document.getElementById("prediction-chart");

    // --- Prediction Feedback References ---
    const feedbackStatusBadge = document.getElementById("feedback-status-badge");
    const feedbackCorrectBtn = document.getElementById("feedback-correct-btn");
    const feedbackIncorrectBtn = document.getElementById("feedback-incorrect-btn");
    const feedbackCorrectionPanel = document.getElementById("feedback-correction-panel");
    const feedbackActualDigit = document.getElementById("feedback-actual-digit");
    const feedbackSubmitBtn = document.getElementById("feedback-submit-btn");

    // --- Preprocessing Details References ---
    const detailOriginalImg = document.getElementById("detail-original-img");
    const detailGrayscaleImg = document.getElementById("detail-grayscale-img");
    const detailTensorImg = document.getElementById("detail-tensor-img");
    const pipelineStepsList = document.getElementById("pipeline-steps-list");
    const preprocessingCollapse = document.getElementById("preprocessing-collapse");
    const toggleDetailsLabel = document.getElementById("toggle-details-label");

    // --- Prediction History References ---
    const historyCountBadge = document.getElementById("history-count-badge");
    const clearHistoryBtn = document.getElementById("clear-history-btn");
    const exportCsvBtn = document.getElementById("export-csv-btn");
    const historyEmpty = document.getElementById("history-empty");
    const historyItemsContainer = document.getElementById("history-items-container");
    const historyTableBody = document.getElementById("history-table-body");
    const historySearchInput = document.getElementById("history-search-input");
    const historyFilterDigit = document.getElementById("history-filter-digit");
    const historyFilterType = document.getElementById("history-filter-type");
    const historySortSelect = document.getElementById("history-sort-select");
    const historyStatsIndicator = document.getElementById("history-stats-indicator");
    const historyDetailModal = document.getElementById("historyDetailModal");
    const historyDetailModalBody = document.getElementById("history-detail-modal-body");
    const historyDetailDeleteBtn = document.getElementById("history-detail-delete-btn");

    // --- Sample Gallery References ---
    const gallerySelectedBadge = document.getElementById("gallery-selected-badge");
    const galleryPredictBtn = document.getElementById("gallery-predict-btn");
    const galleryPredictSpinner = document.getElementById("gallery-predict-spinner");
    const galleryPredictBtnText = document.getElementById("gallery-predict-btn-text");
    const galleryClearBtn = document.getElementById("gallery-clear-btn");
    const sampleReloadBtn = document.getElementById("sample-reload-btn");
    const sampleCards = document.querySelectorAll(".sample-digit-card");

    // --- Digit Challenge References ---
    const challengeTargetDigit = document.getElementById("challenge-target-digit");
    const challengeScoreEl = document.getElementById("challenge-score");
    const challengeAttemptsEl = document.getElementById("challenge-attempts");
    const challengeAccuracyEl = document.getElementById("challenge-accuracy");
    const challengeFeedbackBox = document.getElementById("challenge-feedback-box");
    const challengeSubmitBtn = document.getElementById("challenge-submit-btn");
    const challengeNextBtn = document.getElementById("challenge-next-btn");
    const challengeResetBtn = document.getElementById("challenge-reset-btn");

    // --- Theme & Toast References ---
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeIconSun = document.getElementById("theme-icon-sun");
    const themeIconMoon = document.getElementById("theme-icon-moon");
    const toastContainer = document.getElementById("toast-container");

    // =========================================================================
    // State Management
    // =========================================================================
    const HISTORY_STORAGE_KEY = "mnist_prediction_history";
    const MAX_HISTORY_ITEMS = 25;

    let isDrawing = false;
    let hasDrawn = false;
    let isEraserMode = false;
    let brushSize = parseInt(brushSizeInput ? brushSizeInput.value : 16, 10);
    let canvasHistory = [];
    let canvasRedoStack = [];
    let autoPredictTimer = null;
    let selectedFile = null;
    let isRequestActive = false;
    let currentOriginalDataUrl = "";
    let currentPredictionId = null;
    let selectedHistoryId = null;
    let selectedSample = null;
    let chartInstance = null;

    // Challenge Mode State
    let challengeTarget = 7;
    let challengeScore = 0;
    let challengeAttempts = 0;
    let challengeCorrect = 0;

    const MAX_FILE_SIZE = 2 * 1024 * 1024; // 2 MB
    const ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg"];
    const ALLOWED_MIME_TYPES = ["image/png", "image/jpeg", "image/jpg"];

    // =========================================================================
    // UI Helpers: Notifications, States & Formatting
    // =========================================================================

    /**
     * Display a non-blocking floating toast notification.
     */
    function showToast(message, type = "info") {
        if (!toastContainer) return;
        const toast = document.createElement("div");
        toast.className = `toast-custom border-start border-4 ${
            type === "success" ? "border-success" : type === "danger" ? "border-danger" : "border-primary"
        }`;

        const iconClass =
            type === "success" ? "bi-check-circle-fill text-success" :
            type === "danger" ? "bi-exclamation-triangle-fill text-danger" :
            "bi-info-circle-fill text-primary";

        toast.innerHTML = `
            <i class="bi ${iconClass} fs-5"></i>
            <div class="flex-grow-1">${message}</div>
            <button type="button" class="btn-close btn-close-sm ms-2" aria-label="Close"></button>
        `;

        const closeBtn = toast.querySelector(".btn-close");
        closeBtn.addEventListener("click", () => toast.remove());

        toastContainer.appendChild(toast);
        setTimeout(() => {
            if (toast.parentElement) toast.remove();
        }, 4000);
    }

    /**
     * Switch visible UI state in results column.
     */
    function showState(state) {
        stateEmpty.classList.add("d-none");
        stateLoading.classList.add("d-none");
        stateError.classList.add("d-none");
        stateSuccess.classList.add("d-none");

        switch (state) {
            case "empty":
                stateEmpty.classList.remove("d-none");
                if (resultStatusBadge) {
                    resultStatusBadge.textContent = "Ready to Recognize";
                    resultStatusBadge.className = "badge bg-light text-secondary border extra-small";
                }
                break;
            case "loading":
                stateLoading.classList.remove("d-none");
                if (resultStatusBadge) {
                    resultStatusBadge.textContent = "Analyzing Digit...";
                    resultStatusBadge.className = "badge bg-warning-subtle text-warning border extra-small";
                }
                break;
            case "error":
                stateError.classList.remove("d-none");
                if (resultStatusBadge) {
                    resultStatusBadge.textContent = "Prediction Error";
                    resultStatusBadge.className = "badge bg-danger-subtle text-danger border extra-small";
                }
                break;
            case "success":
                stateSuccess.classList.remove("d-none");
                if (resultStatusBadge) {
                    resultStatusBadge.textContent = "✓ Prediction Complete";
                    resultStatusBadge.className = "badge bg-success-subtle text-success border extra-small";
                }
                break;
        }
    }

    function formatBytes(bytes) {
        if (!bytes || bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }

    function displayError(msg) {
        errorMessage.textContent = msg;
        showState("error");
        showToast(msg, "danger");
    }

    // =========================================================================
    // SECTION 1: Enhanced Interactive Canvas (Feature 1 & Feature 2)
    // =========================================================================

    function initCanvas() {
        if (!ctx) return;
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        canvasHistory = [ctx.getImageData(0, 0, canvas.width, canvas.height)];
        canvasRedoStack = [];
        hasDrawn = false;
        updateCanvasActionButtons();
    }

    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function updateCanvasActionButtons() {
        const canUndo = canvasHistory.length > 1;
        const canRedo = canvasRedoStack.length > 0;

        if (canvasUndoBtn) canvasUndoBtn.disabled = !canUndo;
        if (canvasCornerUndoBtn) canvasCornerUndoBtn.disabled = !canUndo;
        if (canvasRedoBtn) canvasRedoBtn.disabled = !canRedo;
    }

    function pushCanvasHistory() {
        if (!ctx) return;
        if (canvasHistory.length > 30) {
            canvasHistory.shift();
        }
        canvasHistory.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
        canvasRedoStack = []; // New stroke invalidates redo stack
        hasDrawn = true;
        updateCanvasActionButtons();
    }

    function undoLastStroke() {
        if (!ctx || canvasHistory.length <= 1) return;
        const currentSnapshot = canvasHistory.pop();
        canvasRedoStack.push(currentSnapshot);

        const previousSnapshot = canvasHistory[canvasHistory.length - 1];
        ctx.putImageData(previousSnapshot, 0, 0);

        if (canvasHistory.length <= 1) {
            hasDrawn = false;
            showState("empty");
        }
        updateCanvasActionButtons();
    }

    function redoStroke() {
        if (!ctx || canvasRedoStack.length === 0) return;
        const redoSnapshot = canvasRedoStack.pop();
        canvasHistory.push(redoSnapshot);
        ctx.putImageData(redoSnapshot, 0, 0);
        hasDrawn = true;
        updateCanvasActionButtons();
    }

    function clearCanvas(withConfirmation = false) {
        if (withConfirmation && hasDrawn) {
            if (!window.confirm("Clear the current drawing canvas?")) return;
        }
        initCanvas();
        showState("empty");
    }

    function setDrawingTool(tool) {
        if (tool === "eraser") {
            isEraserMode = true;
            btnToolEraser?.classList.add("active");
            btnToolBrush?.classList.remove("active");
            if (canvas) canvas.style.cursor = "cell";
        } else {
            isEraserMode = false;
            btnToolBrush?.classList.add("active");
            btnToolEraser?.classList.remove("active");
            if (canvas) canvas.style.cursor = "crosshair";
        }
    }

    // --- Pointer Events for Canvas (Mouse, Touch, Stylus) ---
    if (canvas && ctx) {
        initCanvas();

        canvas.addEventListener("pointerdown", (e) => {
            if (autoPredictTimer) {
                clearTimeout(autoPredictTimer);
                autoPredictTimer = null;
            }

            isDrawing = true;
            try {
                canvas.setPointerCapture(e.pointerId);
            } catch (_) {}

            const { x, y } = getCanvasCoordinates(e);
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.strokeStyle = isEraserMode ? "#ffffff" : "#0f172a";
            ctx.lineWidth = isEraserMode ? brushSize * 1.6 : brushSize;

            ctx.lineTo(x + 0.1, y + 0.1);
            ctx.stroke();
        });

        canvas.addEventListener("pointermove", (e) => {
            if (!isDrawing) return;
            const { x, y } = getCanvasCoordinates(e);
            ctx.lineTo(x, y);
            ctx.stroke();
            hasDrawn = true;
        });

        const stopDrawing = (e) => {
            if (!isDrawing) return;
            isDrawing = false;
            ctx.closePath();
            try {
                canvas.releasePointerCapture(e.pointerId);
            } catch (_) {}
            pushCanvasHistory();

            // Auto-Predict functionality (~700ms debounce)
            if (autoPredictToggle && autoPredictToggle.checked && hasDrawn && !isEraserMode) {
                autoPredictTimer = setTimeout(() => {
                    submitCanvasPrediction();
                }, 700);
            }
        };

        canvas.addEventListener("pointerup", stopDrawing);
        canvas.addEventListener("pointercancel", stopDrawing);
        canvas.addEventListener("pointerleave", (e) => {
            if (isDrawing && !e.buttons) stopDrawing(e);
        });

        // Brush tool listeners
        btnToolBrush?.addEventListener("click", () => setDrawingTool("brush"));
        btnToolEraser?.addEventListener("click", () => setDrawingTool("eraser"));

        // Brush size preset buttons
        brushDotBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const size = parseInt(btn.dataset.size, 10);
                brushSize = size;
                if (brushSizeInput) brushSizeInput.value = size;
                if (brushSizeVal) brushSizeVal.textContent = `${size}px`;
                brushDotBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
            });
        });

        // Brush slider listener
        brushSizeInput?.addEventListener("input", (e) => {
            brushSize = parseInt(e.target.value, 10);
            if (brushSizeVal) brushSizeVal.textContent = `${brushSize}px`;
            brushDotBtns.forEach(b => {
                b.classList.toggle("active", parseInt(b.dataset.size, 10) === brushSize);
            });
        });

        // Action button listeners
        canvasUndoBtn?.addEventListener("click", undoLastStroke);
        canvasCornerUndoBtn?.addEventListener("click", undoLastStroke);
        canvasRedoBtn?.addEventListener("click", redoStroke);
        canvasClearBtn?.addEventListener("click", () => clearCanvas(false));
        canvasCornerClearBtn?.addEventListener("click", () => clearCanvas(false));
        canvasPredictBtn?.addEventListener("click", submitCanvasPrediction);

        // Keyboard Shortcuts (Feature 1 & Feature 12)
        document.addEventListener("keydown", (e) => {
            // Ignore keystrokes when typing in inputs/textareas
            if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;

            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
                e.preventDefault();
                if (e.shiftKey) {
                    redoStroke();
                } else {
                    undoLastStroke();
                }
            } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
                e.preventDefault();
                redoStroke();
            } else if (e.key.toLowerCase() === "b") {
                setDrawingTool("brush");
            } else if (e.key.toLowerCase() === "e") {
                setDrawingTool("eraser");
            } else if (e.key.toLowerCase() === "c") {
                clearCanvas(true);
            } else if (e.key === "Enter") {
                const drawTabBtn = document.getElementById("tab-draw-btn");
                if (drawTabBtn && drawTabBtn.classList.contains("active")) {
                    submitCanvasPrediction();
                }
            }
        });
    }

    // =========================================================================
    // SECTION 2: Image File Upload Engine (Feature 4)
    // =========================================================================

    function handleFile(file) {
        if (!file) return;

        const lowerName = (file.name || "").toLowerCase();
        const hasValidExt = ALLOWED_EXTENSIONS.some(ext => lowerName.endsWith(ext));
        const hasValidMime = ALLOWED_MIME_TYPES.includes(file.type);

        if (!hasValidExt && !hasValidMime) {
            displayError("Unsupported file format. Please upload a valid PNG, JPG, or JPEG image.");
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            displayError("File exceeds the 2MB limit. Please upload a smaller image.");
            return;
        }

        selectedFile = file;
        const reader = new FileReader();

        reader.onload = (e) => {
            currentOriginalDataUrl = e.target.result;
            imagePreview.src = e.target.result;
            fileDetails.textContent = `${file.name} (${formatBytes(file.size)})`;
            dropPrompt.classList.add("d-none");
            previewWrapper.classList.remove("d-none");

            predictBtn.disabled = false;
            clearBtn.disabled = false;

            showState("empty");
        };

        reader.onerror = () => {
            displayError("Failed to read the selected file. Please try another image.");
        };

        reader.readAsDataURL(file);
    }

    function resetForm() {
        if (isRequestActive) return;

        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "#";
        fileDetails.textContent = "";

        previewWrapper.classList.add("d-none");
        dropPrompt.classList.remove("d-none");

        predictBtn.disabled = true;
        clearBtn.disabled = true;

        showState("empty");
    }

    fileInput?.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    browseBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput?.click();
    });

    if (dropZone && fileInput) {
        dropZone.addEventListener("click", () => fileInput.click());
        dropZone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                fileInput.click();
            }
        });

        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("drag-active");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("drag-active");
            });
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files[0]) {
                handleFile(dt.files[0]);
            }
        });
    }

    predictBtn?.addEventListener("click", submitUploadPrediction);
    clearBtn?.addEventListener("click", resetForm);

    // =========================================================================
    // SECTION 3: Unified Prediction & Analytics Engine (Feature 2 & 3)
    // =========================================================================

    function isValidPredictionResponse(data) {
        if (!data || typeof data !== "object") return false;
        if (data.success !== true) return false;
        if (typeof data.predicted_digit !== "number" || data.predicted_digit < 0 || data.predicted_digit > 9) {
            return false;
        }
        if (typeof data.confidence !== "number" || data.confidence < 0.0 || data.confidence > 1.0) {
            return false;
        }
        if (!data.probabilities || typeof data.probabilities !== "object") {
            return false;
        }

        for (let digit = 0; digit < 10; digit++) {
            const val = data.probabilities[String(digit)];
            if (typeof val !== "number" || val < 0.0 || val > 1.0) {
                return false;
            }
        }
        return true;
    }

    function renderTopThree(backendTopThree, probabilities, winningDigit) {
        if (!topThreeList) return;
        topThreeList.innerHTML = "";

        let topCandidates = [];
        if (backendTopThree && Array.isArray(backendTopThree) && backendTopThree.length >= 3) {
            topCandidates = backendTopThree.slice(0, 3);
        } else {
            const all = [];
            for (let d = 0; d < 10; d++) {
                all.push({
                    digit: d,
                    probability: probabilities[String(d)] !== undefined ? probabilities[String(d)] : 0.0,
                });
            }
            all.sort((a, b) => b.probability - a.probability);
            topCandidates = all.slice(0, 3);
        }

        const rankLabels = ["1", "2", "3"];

        topCandidates.forEach((candidate, idx) => {
            const isWinner = candidate.digit === winningDigit;
            const pct = (candidate.probability * 100).toFixed(2);

            const row = document.createElement("div");
            row.className = "top-candidate-row";

            row.innerHTML = `
                <span class="rank-circle ${isWinner ? 'winner' : ''}">${rankLabels[idx]}</span>
                <span class="digit-name">${candidate.digit}</span>
                <div class="candidate-bar-wrapper">
                    <div class="candidate-bar-fill ${isWinner ? 'winner' : ''}" style="width: ${pct}%"></div>
                </div>
                <span class="candidate-pct ${isWinner ? 'winner' : ''}">${pct}%</span>
            `;

            topThreeList.appendChild(row);
        });
    }

    /**
     * Render the 10-digit probability distribution with Chart.js and fallback DOM list.
     */
    function renderProbabilities(probabilities, winningDigit) {
        const labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
        const values = labels.map(d => {
            const p = probabilities[d] !== undefined ? probabilities[d] : 0.0;
            return parseFloat((p * 100).toFixed(2));
        });

        // 1. Chart.js Interactive Bar Chart
        if (predictionChartCanvas && window.Chart) {
            const bgColors = labels.map(d =>
                parseInt(d, 10) === winningDigit ? "#2563eb" : "#cbd5e1"
            );
            const borderColors = labels.map(d =>
                parseInt(d, 10) === winningDigit ? "#1d4ed8" : "#94a3b8"
            );

            if (chartInstance) {
                chartInstance.data.datasets[0].data = values;
                chartInstance.data.datasets[0].backgroundColor = bgColors;
                chartInstance.data.datasets[0].borderColor = borderColors;
                chartInstance.update();
            } else {
                chartInstance = new Chart(predictionChartCanvas, {
                    type: "bar",
                    data: {
                        labels: labels,
                        datasets: [{
                            label: "Probability (%)",
                            data: values,
                            backgroundColor: bgColors,
                            borderColor: borderColors,
                            borderWidth: 1,
                            borderRadius: 4,
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: (ctx) => `Probability: ${ctx.parsed.y.toFixed(2)}%`
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: {
                                    stepSize: 25,
                                    callback: (val) => `${val}%`,
                                    font: { size: 10 }
                                },
                                grid: { color: "rgba(0,0,0,0.06)" }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { font: { weight: "bold", size: 11 } }
                            }
                        }
                    }
                });
            }
        }

        // 2. Preserved #probabilities-list for test verification
        if (probabilitiesList) {
            probabilitiesList.innerHTML = "";

            for (let digit = 0; digit < 10; digit++) {
                const digitKey = String(digit);
                const prob = probabilities[digitKey] !== undefined ? probabilities[digitKey] : 0.0;
                const percentage = (prob * 100).toFixed(2);
                const isWinner = digit === winningDigit;

                const row = document.createElement("div");
                row.className = `prob-row d-flex align-items-center mb-1 ${isWinner ? "active" : ""}`;

                row.innerHTML = `
                    <div class="prob-label">Digit ${digit}</div>
                    <div class="prob-bar-wrapper me-2">
                        <div class="progress" role="progressbar" aria-label="Digit ${digit} probability" aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
                            <div class="progress-bar" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                    <div class="prob-value small">${percentage}%</div>
                `;

                probabilitiesList.appendChild(row);
            }
        }
    }

    /**
     * Shared prediction execution pipeline.
     */
    async function executePrediction(payload, isJsonPayload, buttonEl, spinnerEl, textEl, inputTypeOverride = null) {
        if (isRequestActive) return;

        isRequestActive = true;
        showState("loading");

        if (buttonEl) buttonEl.disabled = true;
        if (spinnerEl) spinnerEl.classList.remove("d-none");
        if (textEl) textEl.textContent = "Analyzing...";

        try {
            const fetchOptions = { method: "POST" };
            if (isJsonPayload) {
                fetchOptions.headers = { "Content-Type": "application/json" };
                fetchOptions.body = JSON.stringify(payload);
            } else {
                fetchOptions.body = payload;
            }

            const response = await fetch("/predict", fetchOptions);

            let data;
            try {
                data = await response.json();
            } catch (_) {
                throw new Error("Unable to parse server response as JSON.");
            }

            if (response.ok && data.success) {
                if (!isValidPredictionResponse(data)) {
                    displayError("Received an invalid response format from the prediction service.");
                    return;
                }

                currentPredictionId = data.prediction_id || `pred_${Date.now()}`;

                // Populate Result Hero
                resultDigit.textContent = data.predicted_digit;
                resultConfidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

                // Reset feedback controls
                resetFeedbackUI();

                // Populate Top 3 Candidates & Chart
                renderTopThree(data.top_three, data.probabilities, data.predicted_digit);
                renderProbabilities(data.probabilities, data.predicted_digit);

                // Populate Preprocessing Visualization Details (Feature 5)
                if (detailOriginalImg && currentOriginalDataUrl) {
                    detailOriginalImg.src = currentOriginalDataUrl;
                }
                if (data.preprocessing) {
                    if (detailGrayscaleImg && data.preprocessing.grayscale_preview) {
                        detailGrayscaleImg.src = data.preprocessing.grayscale_preview;
                    }
                    if (detailTensorImg && data.preprocessing.tensor_preview) {
                        detailTensorImg.src = data.preprocessing.tensor_preview;
                    }
                    if (pipelineStepsList && Array.isArray(data.preprocessing.steps)) {
                        pipelineStepsList.innerHTML = "";
                        data.preprocessing.steps.forEach(step => {
                            const li = document.createElement("li");
                            li.textContent = step;
                            pipelineStepsList.appendChild(li);
                        });
                    }
                }

                showState("success");

                // Save prediction to SQLite & session history
                savePredictionToHistory(data, isJsonPayload, currentOriginalDataUrl, inputTypeOverride);

                // Update challenge mode if active
                if (challengeTarget !== null) {
                    handleChallengeEvaluation(data.predicted_digit, data.confidence);
                }
            } else {
                const errText = data.error || `Prediction request failed (HTTP ${response.status}).`;
                displayError(errText);
            }
        } catch (err) {
            displayError("Network error. Please ensure the Flask backend is running.");
        } finally {
            isRequestActive = false;
            if (spinnerEl) spinnerEl.classList.add("d-none");
            if (textEl) textEl.textContent = "Predict Digit";
            if (buttonEl) buttonEl.disabled = false;
        }
    }

    function submitCanvasPrediction() {
        if (!hasDrawn) {
            displayError("Please draw a digit (0–9) on the canvas before predicting.");
            return;
        }

        const dataUrl = canvas.toDataURL("image/png");
        currentOriginalDataUrl = dataUrl;
        executePrediction(
            { image: dataUrl, input_type: "Canvas" },
            true,
            canvasPredictBtn,
            canvasPredictSpinner,
            canvasPredictBtnText,
            "Canvas"
        );
    }

    function submitUploadPrediction() {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("input_type", "Upload");

        executePrediction(
            formData,
            false,
            predictBtn,
            predictSpinner,
            predictBtnText,
            "Upload"
        );
    }

    // Alias for automated tests compatibility
    function submitPrediction() {
        submitUploadPrediction();
    }

    async function submitSamplePrediction() {
        if (!selectedSample) {
            displayError("Please select a sample digit from the gallery first.");
            return;
        }

        try {
            const res = await fetch(selectedSample.src);
            const blob = await res.blob();
            currentOriginalDataUrl = selectedSample.src;

            const formData = new FormData();
            formData.append("file", blob, `sample_digit_${selectedSample.digit}.png`);
            formData.append("input_type", "Sample");

            executePrediction(
                formData,
                false,
                galleryPredictBtn,
                galleryPredictSpinner,
                galleryPredictBtnText,
                "Sample"
            );
        } catch (err) {
            displayError("Failed to load sample digit image for prediction.");
        }
    }

    function clearSampleSelection() {
        selectedSample = null;
        sampleCards.forEach(c => c.classList.remove("active-sample-card"));
        if (gallerySelectedBadge) gallerySelectedBadge.textContent = "No sample selected";
        if (galleryPredictBtn) galleryPredictBtn.disabled = true;
        if (galleryClearBtn) galleryClearBtn.disabled = true;
        showState("empty");
    }

    sampleCards.forEach(card => {
        const onSelect = () => {
            const digit = parseInt(card.dataset.digit, 10);
            const src = card.dataset.src;

            sampleCards.forEach(c => {
                c.classList.toggle("active-sample-card", parseInt(c.dataset.digit, 10) === digit);
            });

            selectedSample = { digit, src };
            currentOriginalDataUrl = src;

            if (gallerySelectedBadge) {
                gallerySelectedBadge.textContent = `Selected: Digit ${digit} (Ground Truth: ${digit})`;
            }
            if (galleryPredictBtn) galleryPredictBtn.disabled = false;
            if (galleryClearBtn) galleryClearBtn.disabled = false;
            showState("empty");
        };

        card.addEventListener("click", onSelect);
        card.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect();
            }
        });
    });

    galleryClearBtn?.addEventListener("click", clearSampleSelection);
    galleryPredictBtn?.addEventListener("click", submitSamplePrediction);
    sampleReloadBtn?.addEventListener("click", clearSampleSelection);

    function handlePredictAgain() {
        const drawTabBtn = document.getElementById("tab-draw-btn");
        const galleryTabBtn = document.getElementById("tab-gallery-btn");

        if (drawTabBtn && drawTabBtn.classList.contains("active")) {
            clearCanvas(false);
        } else if (galleryTabBtn && galleryTabBtn.classList.contains("active")) {
            clearSampleSelection();
        } else {
            resetForm();
        }
        showState("empty");
    }

    predictAgainBtn?.addEventListener("click", handlePredictAgain);

    // Preprocessing collapsible listener
    if (preprocessingCollapse && toggleDetailsLabel) {
        preprocessingCollapse.addEventListener("show.bs.collapse", () => {
            toggleDetailsLabel.textContent = "Hide Details ▲";
        });
        preprocessingCollapse.addEventListener("hide.bs.collapse", () => {
            toggleDetailsLabel.textContent = "Show Details ▼";
        });
    }

    // =========================================================================
    // SECTION 4: Prediction Feedback (Feature 9)
    // =========================================================================

    function resetFeedbackUI() {
        if (!feedbackStatusBadge) return;
        feedbackStatusBadge.textContent = "Unreviewed";
        feedbackStatusBadge.className = "badge bg-light text-muted border extra-small";
        feedbackCorrectBtn?.classList.remove("active");
        feedbackIncorrectBtn?.classList.remove("active");
        feedbackCorrectionPanel?.classList.add("d-none");
    }

    async function sendFeedback(feedbackType, actualDigit = null) {
        if (!currentPredictionId) return;

        try {
            const res = await fetch("/api/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prediction_id: currentPredictionId,
                    feedback: feedbackType,
                    actual_digit: actualDigit,
                })
            });
            const data = await res.json();
            if (res.ok && data.success) {
                if (feedbackType === "correct") {
                    feedbackStatusBadge.textContent = "✓ Verified Correct";
                    feedbackStatusBadge.className = "badge bg-success-subtle text-success border extra-small";
                    feedbackCorrectBtn.classList.add("active");
                    feedbackIncorrectBtn.classList.remove("active");
                    feedbackCorrectionPanel?.classList.add("d-none");
                    showToast("Feedback recorded: Verified correct!", "success");
                } else {
                    const label = actualDigit !== null ? `Incorrect (Actual: ${actualDigit})` : "Reported Incorrect";
                    feedbackStatusBadge.textContent = `✗ ${label}`;
                    feedbackStatusBadge.className = "badge bg-danger-subtle text-danger border extra-small";
                    feedbackIncorrectBtn.classList.add("active");
                    feedbackCorrectBtn.classList.remove("active");
                    showToast("Feedback recorded. Thank you!", "info");
                }
                loadFeedbackStats();
                fetchHistoryFromBackend();
            }
        } catch (_) {
            showToast("Failed to save feedback.", "danger");
        }
    }

    feedbackCorrectBtn?.addEventListener("click", () => sendFeedback("correct"));
    feedbackIncorrectBtn?.addEventListener("click", () => {
        feedbackCorrectionPanel?.classList.remove("d-none");
    });
    feedbackSubmitBtn?.addEventListener("click", () => {
        const actual = parseInt(feedbackActualDigit.value, 10);
        sendFeedback("incorrect", actual);
    });

    async function loadFeedbackStats() {
        if (!historyStatsIndicator) return;
        try {
            const res = await fetch("/api/feedback/stats");
            const data = await res.json();
            if (res.ok && data.success && data.stats) {
                const s = data.stats;
                if (s.total_feedback > 0 && s.observed_accuracy !== null) {
                    historyStatsIndicator.textContent = `Observed Feedback Accuracy: ${s.observed_accuracy}% (${s.correct_count}/${s.total_feedback})`;
                } else {
                    historyStatsIndicator.textContent = "Observed Feedback Accuracy: No feedback yet";
                }
            }
        } catch (_) {}
    }

    // =========================================================================
    // SECTION 5: Digit Challenge Mode (Feature 10)
    // =========================================================================

    function generateNewChallenge() {
        let nextTarget;
        do {
            nextTarget = Math.floor(Math.random() * 10);
        } while (nextTarget === challengeTarget);

        challengeTarget = nextTarget;
        if (challengeTargetDigit) challengeTargetDigit.textContent = challengeTarget;
        if (challengeFeedbackBox) {
            challengeFeedbackBox.classList.add("d-none");
            challengeFeedbackBox.textContent = "";
        }
    }

    function updateChallengeScoreboard() {
        if (challengeScoreEl) challengeScoreEl.textContent = challengeScore;
        if (challengeAttemptsEl) challengeAttemptsEl.textContent = challengeAttempts;
        if (challengeAccuracyEl) {
            const acc = challengeAttempts > 0 ? ((challengeCorrect / challengeAttempts) * 100).toFixed(0) : 0;
            challengeAccuracyEl.textContent = `${acc}%`;
        }
    }

    function handleChallengeEvaluation(predictedDigit, confidence) {
        const challengePane = document.getElementById("tab-challenge-pane");
        const isChallengeActive = challengePane && challengePane.classList.contains("active");
        if (!isChallengeActive) return;

        challengeAttempts += 1;
        const isMatch = predictedDigit === challengeTarget;

        if (challengeFeedbackBox) {
            challengeFeedbackBox.classList.remove("d-none");
            if (isMatch) {
                challengeScore += 10;
                challengeCorrect += 1;
                challengeFeedbackBox.className = "challenge-feedback-msg bg-success-subtle text-success";
                challengeFeedbackBox.innerHTML = `
                    <i class="bi bi-trophy-fill me-1"></i> Success! CNN recognized your digit <strong>${challengeTarget}</strong> with ${(confidence * 100).toFixed(1)}% confidence! (+10 pts)
                `;
                showToast(`Challenge Matched! +10 points`, "success");
            } else {
                challengeFeedbackBox.className = "challenge-feedback-msg bg-warning-subtle text-danger";
                challengeFeedbackBox.innerHTML = `
                    <i class="bi bi-x-circle me-1"></i> The model classified this as <strong>${predictedDigit}</strong> (${(confidence * 100).toFixed(1)}% conf). Target was <strong>${challengeTarget}</strong>. Try again!
                `;
            }
        }
        updateChallengeScoreboard();
    }

    challengeSubmitBtn?.addEventListener("click", () => {
        if (!hasDrawn) {
            displayError("Please draw your target digit on the canvas first!");
            return;
        }
        submitCanvasPrediction();
    });

    challengeNextBtn?.addEventListener("click", () => {
        generateNewChallenge();
        clearCanvas(false);
    });

    challengeResetBtn?.addEventListener("click", () => {
        if (window.confirm("Reset your challenge score and attempts?")) {
            challengeScore = 0;
            challengeAttempts = 0;
            challengeCorrect = 0;
            updateChallengeScoreboard();
            generateNewChallenge();
            clearCanvas(false);
        }
    });

    // =========================================================================
    // SECTION 6: Persistent History & CSV Export (Feature 7 & 8)
    // =========================================================================

    function isValidHistoryItem(item) {
        if (!item || typeof item !== "object") return false;
        if (typeof item.id !== "string" || !item.id) return false;
        if (typeof item.digit !== "number" || item.digit < 0 || item.digit > 9) return false;
        if (typeof item.confidence !== "number" || item.confidence < 0 || item.confidence > 1) return false;
        if (typeof item.inputType !== "string" || !["Canvas", "Upload", "Sample"].includes(item.inputType)) return false;
        if (typeof item.timestamp !== "string" || !item.timestamp) return false;
        if (!item.data || typeof item.data !== "object") return false;
        if (!item.data.probabilities || typeof item.data.probabilities !== "object") return false;
        return true;
    }

    function getStoredHistory() {
        try {
            const raw = sessionStorage.getItem(HISTORY_STORAGE_KEY);
            if (!raw) return [];
            const parsed = JSON.parse(raw);
            if (!Array.isArray(parsed)) return [];
            return parsed.filter(isValidHistoryItem);
        } catch (_) {
            return [];
        }
    }

    function setStoredHistory(items) {
        try {
            sessionStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items));
        } catch (_) {}
    }

    async function fetchHistoryFromBackend() {
        const search = historySearchInput?.value.trim() || "";
        const digit = historyFilterDigit?.value || "";
        const inputType = historyFilterType?.value || "";
        const sortVal = historySortSelect?.value || "date_desc";

        let sortBy = "date";
        let sortOrder = "desc";
        if (sortVal === "date_asc") {
            sortOrder = "asc";
        } else if (sortVal === "conf_desc") {
            sortBy = "confidence";
            sortOrder = "desc";
        } else if (sortVal === "conf_asc") {
            sortBy = "confidence";
            sortOrder = "asc";
        }

        const params = new URLSearchParams();
        if (search) params.append("search", search);
        if (digit !== "") params.append("digit", digit);
        if (inputType !== "") params.append("input_type", inputType);
        params.append("sort_by", sortBy);
        params.append("sort_order", sortOrder);

        try {
            const res = await fetch(`/api/history?${params.toString()}`);
            const data = await res.json();
            if (res.ok && data.success && Array.isArray(data.history)) {
                renderHistoryRows(data.history);
            } else {
                renderHistory();
            }
        } catch (_) {
            renderHistory();
        }
    }

    function renderHistoryRows(items) {
        if (!historyItemsContainer || !historyEmpty || !historyCountBadge) return;

        const count = items.length;
        historyCountBadge.textContent = `${count} ${count === 1 ? 'entry' : 'entries'}`;

        if (count === 0) {
            historyEmpty.classList.remove("d-none");
            historyItemsContainer.classList.add("d-none");
            clearHistoryBtn?.classList.add("d-none");
            exportCsvBtn?.classList.add("d-none");
            if (historyTableBody) historyTableBody.innerHTML = "";
            return;
        }

        historyEmpty.classList.add("d-none");
        historyItemsContainer.classList.remove("d-none");
        clearHistoryBtn?.classList.remove("d-none");
        exportCsvBtn?.classList.remove("d-none");

        if (historyTableBody) {
            historyTableBody.innerHTML = "";

            items.forEach((item, idx) => {
                const digit = item.predicted_digit !== undefined ? item.predicted_digit : item.digit;
                const conf = item.confidence;
                const pct = (conf * 100).toFixed(1);
                const inputType = item.input_type || item.inputType || "Canvas";
                const icon = inputType === "Canvas" ? "bi-brush" : (inputType === "Sample" ? "bi-grid-3x3" : "bi-cloud-arrow-up");

                let feedbackBadge = `<span class="badge bg-light text-muted border py-0">none</span>`;
                if (item.feedback === "correct") {
                    feedbackBadge = `<span class="badge bg-success-subtle text-success border py-0">✓ correct</span>`;
                } else if (item.feedback === "incorrect") {
                    feedbackBadge = `<span class="badge bg-danger-subtle text-danger border py-0">✗ actual: ${item.actual_digit ?? '?'}</span>`;
                }

                const tr = document.createElement("tr");
                tr.setAttribute("data-id", item.id);
                tr.setAttribute("role", "button");
                tr.setAttribute("tabindex", "0");
                tr.setAttribute("title", `Click to inspect: Digit ${digit} (${pct}%)`);

                tr.innerHTML = `
                    <td class="text-muted extra-small">#${idx + 1}</td>
                    <td>
                        <span class="fw-bold text-navy" style="font-size: 0.95rem;">${digit}</span>
                    </td>
                    <td>
                        <span class="badge bg-success-subtle text-success border border-success-subtle extra-small py-0">
                            ${pct}%
                        </span>
                    </td>
                    <td>
                        <span class="extra-small text-secondary"><i class="bi ${icon} me-1"></i>${inputType}</span>
                    </td>
                    <td class="text-muted extra-small">${item.timestamp}</td>
                    <td>${feedbackBadge}</td>
                    <td class="text-center">
                        <button type="button" class="btn btn-sm btn-link p-0 text-primary me-2 inspect-btn" title="Inspect">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button type="button" class="btn btn-sm btn-link p-0 text-danger delete-btn" title="Delete record">
                            <i class="bi bi-trash"></i>
                        </button>
                    </td>
                `;

                // Inspect action
                const inspect = (e) => {
                    e?.stopPropagation();
                    inspectPrediction(item);
                };

                // Delete action
                const deleteRecord = async (e) => {
                    e.stopPropagation();
                    if (window.confirm(`Delete prediction record #${idx + 1} (Digit ${digit})?`)) {
                        try {
                            const res = await fetch(`/api/history/${item.id}`, { method: "DELETE" });
                            if (res.ok) {
                                showToast("Record deleted", "info");
                                fetchHistoryFromBackend();
                            }
                        } catch (_) {
                            showToast("Failed to delete record.", "danger");
                        }
                    }
                };

                tr.addEventListener("click", inspect);
                tr.querySelector(".inspect-btn")?.addEventListener("click", inspect);
                tr.querySelector(".delete-btn")?.addEventListener("click", deleteRecord);

                historyTableBody.appendChild(tr);
            });
        }
    }

    function renderHistory(items) {
        if (!items) {
            items = getStoredHistory();
        }
        renderHistoryRows(items);
    }

    function inspectHistoryItem(item) {
        inspectPrediction(item);
    }

    function inspectPrediction(item) {
        if (!item) return;
        const digit = item.predicted_digit !== undefined ? item.predicted_digit : item.digit;
        const probs = item.probabilities || (item.data ? item.data.probabilities : {});
        const topThree = item.top_three || (item.data ? item.data.top_three : []);

        resultDigit.textContent = digit;
        resultConfidence.textContent = `Confidence: ${(item.confidence * 100).toFixed(2)}%`;
        currentPredictionId = item.id;

        renderTopThree(topThree, probs, digit);
        renderProbabilities(probs, digit);
        showState("success");

        showToast(`Inspecting Digit ${digit} from history`, "info");
    }

    function savePredictionToHistory(data, isJsonPayload, originalDataUrl, inputTypeOverride = null) {
        const inputType = inputTypeOverride || (isJsonPayload ? "Canvas" : "Upload");
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        const historyItem = {
            id: data.prediction_id || ("hist_" + Date.now()),
            digit: data.predicted_digit,
            confidence: data.confidence,
            inputType: inputType,
            timestamp: timeStr,
            originalPreview: originalDataUrl || "",
            data: {
                predicted_digit: data.predicted_digit,
                confidence: data.confidence,
                probabilities: data.probabilities,
                top_three: data.top_three || [],
                preprocessing: data.preprocessing || null
            }
        };

        let items = getStoredHistory();
        items.unshift(historyItem);
        if (items.length > MAX_HISTORY_ITEMS) {
            items = items.slice(0, MAX_HISTORY_ITEMS);
        }
        setStoredHistory(items);

        // Fetch fresh records from SQLite backend
        fetchHistoryFromBackend();
    }

    async function clearHistory() {
        if (window.confirm("Are you sure you want to clear your prediction history? This will delete all records permanently.")) {
            try {
                await fetch("/api/history", { method: "DELETE" });
            } catch (_) {}
            sessionStorage.removeItem(HISTORY_STORAGE_KEY);
            selectedHistoryId = null;
            showToast("Prediction history cleared", "info");
            fetchHistoryFromBackend();
        }
    }

    function exportHistoryCSV() {
        // Backend SQLite export with formula injection safety and fallback
        try {
            window.location.href = "/api/history/export";
            showToast("Exporting prediction history to CSV...", "info");
        } catch (_) {
            const items = getStoredHistory();
            if (!items.length) return;
            const headers = ["Record ID", "Predicted Digit", "Confidence (%)", "Input Type", "Timestamp"];
            const rows = items.map(i => [
                `"${i.id}"`,
                i.digit,
                (i.confidence * 100).toFixed(2),
                `"${i.inputType}"`,
                `"${i.timestamp}"`
            ]);
            const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\r\n");
            const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
            const link = document.createElement("a");
            const now = new Date().toISOString().slice(0, 10);
            link.href = URL.createObjectURL(blob);
            link.download = `mnist_predictions_${now}.csv`;
            link.click();
        }
    }

    exportCsvBtn?.addEventListener("click", exportHistoryCSV);
    clearHistoryBtn?.addEventListener("click", clearHistory);

    // Filter listeners with debounce
    let searchDebounce = null;
    historySearchInput?.addEventListener("input", () => {
        clearTimeout(searchDebounce);
        searchDebounce = setTimeout(fetchHistoryFromBackend, 250);
    });
    historyFilterDigit?.addEventListener("change", fetchHistoryFromBackend);
    historyFilterType?.addEventListener("change", fetchHistoryFromBackend);
    historySortSelect?.addEventListener("change", fetchHistoryFromBackend);

    // Initial history & feedback fetch
    fetchHistoryFromBackend();
    loadFeedbackStats();

    // =========================================================================
    // SECTION 7: Health & Theme Initializer
    // =========================================================================

    fetch("/health")
        .then(res => res.json())
        .then(data => {
            const apiStatusText = document.getElementById("api-status-text");
            if (data.status === "ok" && data.model_loaded) {
                if (apiStatusText) apiStatusText.textContent = "Model Ready";
            } else if (data.status === "ok") {
                if (apiStatusText) apiStatusText.textContent = "Model Standby";
            }
        })
        .catch(() => {
            const apiStatusText = document.getElementById("api-status-text");
            if (apiStatusText) apiStatusText.textContent = "Backend Offline";
            apiStatusBadge?.classList.add("text-danger");
        });

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        try {
            localStorage.setItem("mnist_theme", theme);
        } catch (_) {}
        if (theme === "dark") {
            themeIconSun?.classList.add("d-none");
            themeIconMoon?.classList.remove("d-none");
        } else {
            themeIconSun?.classList.remove("d-none");
            themeIconMoon?.classList.add("d-none");
        }
    }

    const savedTheme = (() => {
        try {
            return localStorage.getItem("mnist_theme") || "light";
        } catch (_) {
            return "light";
        }
    })();
    applyTheme(savedTheme);

    themeToggleBtn?.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") || "light";
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next);
        showToast(`Theme changed to ${next} mode`, "info");
    });

    // Initialize challenge target
    generateNewChallenge();
});
