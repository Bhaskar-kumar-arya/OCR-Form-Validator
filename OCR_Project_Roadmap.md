
# Project Goal & Core Technologies

The primary objective is to create an OCR solution to **extract text**, **auto-fill forms**, and **verify the data** for enhanced reliability. The solution will be built as two separate **RESTful APIs**—one for extraction and one for verification. The entire system must run locally without cloud services, leveraging open-source libraries.  

Your PC (Ryzen 7 4800H, RTX 3050 4GB VRAM, 16GB RAM) is well-suited for this.

---

## Phase 0: Foundation & Environment Setup

**Goal:** Prepare your development environment.

### Set Up Python Environment:
- Install Python (**3.8+ recommended**).  
- Create and activate a virtual environment (e.g., `ocr-env`) to manage dependencies.  

### Install Core Libraries:
- **PyTorch & Transformers**: For TrOCR and LayoutLMv3 models.  
- **OpenCV (`opencv-python`)**: For image preprocessing (grayscale, thresholding, deskew, denoising).  
- **FastAPI & Uvicorn**: For serving APIs.  
- **MMOCR (or standalone CRAFT detector)**: For robust text detection beyond EAST.  
- **RapidFuzz**: For string similarity in verification.  

### Download Models:
- From Hugging Face:  
  - TrOCR for printed text (`microsoft/trocr-base-printed`).  
  - TrOCR for handwritten text (`microsoft/trocr-base-handwritten`).  
  - LayoutLMv3 (for key-value field understanding).  

---

## Phase 1: The Intelligent OCR Extraction API (API 1)

**Goal:** Build the extraction API to output structured data from a scanned document. This API must support English.

### API Endpoint:
- `/extract` → accepts scanned PDF/image via `POST`.

### Image Preprocessing Pipeline:
- Grayscale → Adaptive thresholding → Deskew → Noise removal.  
- Optional: Blur/lighting quality check for feedback.  

### Text Detection & Recognition:
- Integrated **CRAFT** for robust text detection, providing bounding boxes for text regions.
- For each detected bounding box, the corresponding region is recognized using TrOCR, and the bounding box coordinates and recognized text are stored.

### Key-Value Pair Extraction:
- Instead of only rules, integrate **LayoutLMv3** (document understanding model).  
  - Inputs: OCR’d text + bounding box coordinates.  
  - Output: structured key-value mapping (`Name → Bhaskar`).  
- Rules (like nearest-neighbor) can still serve as fallback for unmapped fields.  

### Return JSON Output:
```json
{
  "data": {
    "Name": "Ananya Sharma",
    "Age": "29",
    "Gender": "Female",
    "Email Id": "ananya.sharma@example.com"
  },
  "field_coordinates": {
    "Name": [x, y, w, h],
    "Ananya Sharma": [x, y, w, h]
  }
}
```

---

## Phase 2: The Data Verification API (API 2)

**Goal:** Verify user-submitted data against the original document.

### API Endpoint:
- `/verify` → accepts form JSON + original scanned file.  

### Internal Ground Truth Extraction:
- Re-run the **full extraction pipeline (Phase 1)** on the scanned document.  

### Comparison & Scoring:
- Compare extracted values with submitted values.  
- Use **string similarity (Jaro-Winkler / RapidFuzz)** to handle small OCR errors.  
- For critical fields (e.g., DOB, ID numbers), apply stricter matching rules.  

### Return JSON Output:
```json
{
  "verification_summary": [
    { "field": "Name", "status": "Match", "score": 1.0 },
    { "field": "Email Id", "status": "Mismatch", "score": 0.85,
      "submitted": "ananya.sharma@example.co",
      "extracted": "ananya.sharma@example.com" }
  ]
}
```

---

## Phase 3: Interactive Frontend Demo

**Goal:** Showcase end-to-end workflow.

- **Upload:** User uploads document.  
- **Extract:** Calls `/extract` → auto-fills form fields.  
- **Visualize:** Draw bounding boxes on document using coordinate data.  
- **Correct:** User edits incorrect values.  
- **Verify:** Calls `/verify` → shows match/mismatch with color coding.  

---

## Phase 4: Advanced OCR Features

**Goal:** Add "Good-to-Have" features.

- **Handwritten text:** Switch to `trocr-base-handwritten` when requested.  
- **Multi-lingual:** Add support for one non-Latin language (Arabic/Hindi) by plugging compatible TrOCR/transformer models.  
- **Partial data mapping:** Allow null fields when no confident match is found.  

---

## Phase 5: Production-Grade Enhancements

**Goal:** Bonus features for professional-grade solution.

- **Capture quality score:** Blur + lighting analysis before OCR.  
- **Multi-page PDF support:** Loop extraction per page, aggregate results.  
- **MOSIP integration:** Push extracted + verified data into MOSIP flows.  

---

## Phase 6: Finalization & Deliverables

- Push full codebase to GitHub/GitLab.  
- API documentation (endpoints, request/response).  
- Workflow docs: Architecture, Data Flow, Installation Guide.  
- Test cases + sample documents.  
- Demo video + PPT presentation.  

## Phase X: Accuracy Enhancement with Donut

Once the base solution (OCR Extraction + Verification APIs) is stable and functional, integrate Donut (Document Understanding Transformer) as an optional backend to improve field-level extraction accuracy.

**Objectives** 

- Increase field–value mapping accuracy (esp. in multi-layout or noisy forms).

- Reduce cascading errors from separate detection, recognition, and parsing stages.

- Provide structured JSON output directly from scanned documents.

**Steps**

- Model Integration

- Add Donut as an alternate OCR engine.

- Accept scanned documents → output structured JSON (fields + values).

- Pipeline Adaptation

- Convert Donut’s JSON output into the same schema used by the Verification API.

- Implement a switch/flag (mode=baseline vs mode=donut).

**Evaluation**

- Benchmark accuracy improvement on the same validation set.

- Measure Field Match Rate (%) and Verification Confidence Score differences.

- Deployment Strategy

- Keep TrOCR + LayoutLMv3 as the default (lighter, faster) mode.

- Offer Donut mode for high-reliability scenarios where accuracy is critical.

**Expected Outcomes**

- Accuracy Gain: +4–7% field-level correctness.

- Tradeoff: Slightly slower inference and higher compute needs.

- Result: End-to-end reliability boost for sensitive workflows (ID verification, certificates, legal documents
---

⚖️ **Tradeoff captured:**  
- Detection: **CRAFT/MMOCR > EAST** → slower but much higher accuracy.  
- Extraction: **LayoutLMv3** → adds semantic reliability, less reliance on brittle rules.  
- Recognition: **TrOCR** → matches recommended stack and handles printed + handwritten text.  
