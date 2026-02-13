# 🐘 Elephant ID - Streamlit Web App QuickStart

Quick guide to run the Streamlit web interface for elephant re-identification.

---

## Prerequisites

- Python 3.8+ installed
- Model and gallery files present:
  - `makhna_model.pth`
  - `gallery_embeddings.pt`

---

## Installation

### 1. Install Dependencies

```bash
pip install streamlit torch torchvision pillow numpy
```

### 2. Verify Files

Ensure these files exist in the project root:

```
Elephant_ReIdentification/
├── app.py
├── makhna_model.pth
├── gallery_embeddings.pt
└── src/
    └── models/
        └── dual_branch_extractor.py
```

---

## Running the App

### Local Development

From the project root directory:

```bash
streamlit run app.py
```

The app will open automatically in your browser at:
- **Local**: `http://localhost:8501`
- **Network**: `http://192.168.x.x:8501` (accessible from other devices on your network)

---

## Using the App

1. **Upload Image**: Click "Upload Photo" and select an elephant image (JPG/PNG)
2. **View Results**: 
   - ✅ **Match Found**: Shows elephant ID and confidence score
   - ❓ **Unknown**: No confident match (below threshold)
3. **Enhanced Mode**: Enable "Enhanced Scan Mode" for better accuracy on difficult images

---

## Features

- **Database Status**: Sidebar shows registered elephants count
- **Top-5 Candidates**: View alternative matches if uncertain
- **Confidence Scores**: Percentage-based similarity metrics
- **New Registration**: Add new elephants or correct misidentifications

---

## Deployment to Streamlit Cloud

### 1. Commit Model Files

Model files are now allowed in `.gitignore`:

```bash
git add makhna_model.pth gallery_embeddings.pt app.py
git commit -m "Add model files for deployment"
git push
```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set:
   - **Branch**: `main`
   - **Main file**: `app.py`
6. Click **Deploy**

Your app will be live at: `https://your-app-name.streamlit.app`

---

## Troubleshooting

### Model Not Found Error

**Symptom**: `❌ AI Model missing at ...`

**Fix**: Ensure you're running from the project root directory where `makhna_model.pth` exists.

```bash
cd Elephant_ReIdentification
streamlit run app.py
```

### Gallery Empty (0 Elephants)

**Symptom**: Sidebar shows "0 Elephants Registered"

**Fix**: Check that `gallery_embeddings.pt` exists and is not corrupted:

```bash
python -c "import torch; data = torch.load('gallery_embeddings.pt'); print(f'Loaded {len(data[\"labels\"])} images')"
```

### Port Already in Use

**Symptom**: `Port 8501 is not available`

**Fix**: Kill existing Streamlit process or use a different port:

```bash
streamlit run app.py --server.port 8502
```

---

## Configuration

Edit `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#2E7D32"        # Green theme
backgroundColor = "#FFFFFF"      # White background
secondaryBackgroundColor = "#F1F8E9"  # Light green

[server]
headless = true
port = 8501
```

---

## Performance Tips

- **Enhanced Mode**: Only use when images are blurry/difficult (slower but more accurate)
- **Local Network**: Access from mobile devices using the network URL
- **GPU**: Model runs on CPU by default (sufficient for web inference)

---

## System Information

- **Model**: Makhna 19-ID Baseline
- **Architecture**: Dual-branch feature extractor with BAM attention
- **Input Size**: 256×128 pixels
- **Embedding**: 128-dimensional normalized features
- **Threshold**: 0.4 cosine similarity for positive match

---

## Next Steps

- **Desktop App**: For fully offline use, see `ElephantID_Offline/` folder
- **API Deployment**: For batch processing, consider FastAPI wrapper
- **Mobile**: Use Streamlit Cloud URL on mobile browsers

---

**Wildlife Institute of India**  
Makhna Elephant Re-ID System v1.0
