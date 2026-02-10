# Kaggle Deployment Guide - Elephant Re-ID Training

## 🚀 Quick Start

This guide will help you train the Elephant Re-Identification model on Kaggle with **GPU acceleration** (up to 30 hours/week free GPU time).

---

## 📋 Prerequisites

1. **Kaggle Account** - Sign up at [kaggle.com](https://www.kaggle.com)
2. **Dataset Uploaded** - Upload your processed elephant images as a Kaggle dataset
3. **GPU Quota** - Ensure you have GPU hours available (free tier: 30 hours/week)

---

## 📦 Step 1: Create Kaggle Dataset

### Option A: Upload via Kaggle Web Interface

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"New Dataset"**
3. Upload your `data/processed_megadetector` folder
4. Name it: `elephant-reid-processed`
5. Make it **Public** or **Private** (your choice)

### Option B: Upload via Kaggle API (Recommended for large datasets)

```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials (get from kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Create dataset metadata
cd data/processed_megadetector
kaggle datasets init -p .

# Edit dataset-metadata.json with your details
# Then upload:
kaggle datasets create -p . -r zip
```

**Expected Dataset Structure:**
```
elephant-reid-processed/
├── Makhna/
│   ├── Makhna_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Makhna_2/
│   └── ...
└── Herd/
    ├── Herd_1/
    └── ...
```

---

## 📝 Step 2: Create Kaggle Notebook

### Recommended: Use Enhanced Notebook V2 (with checkpoints & early stopping)

**File: `kaggle/elephant_reid_training_v2.ipynb`**

This enhanced version includes:
- ✅ **Checkpoint Management** - Save every 5 epochs, resume interrupted training
- ✅ **Early Stopping** - Automatically stop when validation plateaus (patience=15)
- ✅ **Mixed Precision (AMP)** - 2-3x faster training on GPU
- ✅ **Gradient Clipping** - Improved training stability
- ✅ **Learning Rate Warmup** - Better convergence
- ✅ **Enhanced Augmentation** - RandomAffine, GaussianBlur, RandomErasing
- ✅ **Attention Map Visualization** - Monitor what the model learns
- ✅ **Better Error Handling** - Path validation and helpful error messages

### Alternative: Original Notebook

**File: `kaggle/elephant_reid_training.ipynb`**

Simple, straightforward training without advanced features.

### How to Upload

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Copy the contents from `kaggle/elephant_reid_training.ipynb` (created below)
4. Paste into the Kaggle notebook

### Method 2: Upload the Notebook Directly

1. Download `kaggle/elephant_reid_training.ipynb`
2. Go to [kaggle.com/code](https://www.kaggle.com/code)
3. Click **"Import Notebook"**
4. Upload the `.ipynb` file

---

## ⚙️ Step 3: Configure Kaggle Notebook

### Enable GPU Acceleration

1. In your Kaggle notebook, click **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU P100** or **GPU T4**
3. Click **Save**

### Add Dataset

1. In **Settings** → **Add Data**
2. Search for your dataset: `elephant-reid-processed`
3. Click **Add**
4. The dataset will be available at `/kaggle/input/elephant-reid-processed/`

### Set Internet to ON (if needed)

1. In **Settings** → **Internet**
2. Toggle **ON** (needed for installing packages)

---

## 🏃 Step 4: Run Training

### In the Kaggle Notebook:

1. **Cell 1:** Install dependencies
   ```python
   !pip install -q torch torchvision tqdm opencv-python-headless
   ```

2. **Cell 2:** Upload model code (copy from local files)
   - The notebook template includes all necessary code

3. **Cell 3:** Configure paths
   ```python
   DATA_ROOT = "/kaggle/input/elephant-reid-processed"
   OUTPUT_DIR = "/kaggle/working/outputs"
   ```

4. **Cell 4:** Start training
   ```python
   trainer.train(train_loader, val_loader)
   ```

5. Click **Run All** or **Shift+Enter** through cells

---

## 📊 Step 5: Monitor Training

### In Kaggle Notebook:

- **Progress bars** show epoch progress
- **Loss curves** are printed after each epoch
- **Checkpoints** are saved to `/kaggle/working/outputs/models/`

### Expected Training Time:

**With Enhanced Notebook V2 (Mixed Precision):**
- **GPU P100:** ~1.5-2 hours for 100 epochs (2-3x faster)
- **GPU T4:** ~2-3 hours for 100 epochs
- Early stopping may reduce actual time if model converges early

**Original Notebook:**
- **GPU P100:** ~2-3 hours for 100 epochs
- **GPU T4:** ~3-4 hours for 100 epochs
- **CPU:** ~20-30 hours (not recommended)

---

## 💾 Step 6: Download Trained Model

### Option A: Download from Notebook Output

1. After training completes, go to **Output** tab (right sidebar)
2. Download `outputs/models/best_model.pth`
3. Also download training logs and visualizations

### Option B: Save as Kaggle Dataset (Recommended)

```python
# At the end of your notebook
!kaggle datasets version -p /kaggle/working/outputs -m "Trained model - Epoch 100"
```

This creates a new version of your output dataset that you can download later.

---

## 🔍 Step 7: Verify Results

### Check Training Logs

```python
import json
with open('/kaggle/working/outputs/results/training_log.json', 'r') as f:
    logs = json.load(f)
    
print(f"Best validation loss: {min(logs['val_losses']):.4f}")
print(f"Final training loss: {logs['train_losses'][-1]:.4f}")
```

### Visualize Attention Maps

```python
# The notebook includes visualization code
# Attention maps will be saved to /kaggle/working/outputs/visualizations/
```

---

## 🎯 Optimization Tips

### 1. Use Mixed Precision Training (Faster)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    embeddings = model(images)
    loss = criterion(embeddings, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Increase Batch Size on GPU

```python
BATCH_SIZE = 32  # or even 64 on P100
```

### 3. Use DataLoader Optimizations

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,  # Kaggle has 4 CPUs
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True
)
```

### 4. Enable TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/kaggle/working/outputs/tensorboard')
# Log metrics during training
writer.add_scalar('Loss/train', train_loss, epoch)
```

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" Error

**Solution:**
- Reduce `BATCH_SIZE` (try 16 or 8)
- Reduce `IMAGE_SIZE` (try 192x192 instead of 224x224)
- Use gradient accumulation

### Issue: "Dataset not found"

**Solution:**
- Check dataset path: `/kaggle/input/your-dataset-name/`
- Verify dataset is added in Settings → Add Data
- Check dataset structure matches expected format

### Issue: "Kernel died" or "Session timeout"

**Solution:**
- Kaggle sessions timeout after 9 hours of inactivity
- Save checkpoints frequently (every 5 epochs)
- Use `SAVE_FREQ = 5` in config

### Issue: Slow training even with GPU

**Solution:**
- Verify GPU is enabled: `torch.cuda.is_available()` should return `True`
- Check GPU usage: `!nvidia-smi`
- Ensure data is on GPU: `images = images.to('cuda')`

---

## 📈 Expected Results

### After 100 Epochs:

- **Training Loss:** ~0.1-0.2
- **Validation Loss:** ~0.2-0.3
- **Embedding Quality:** Well-separated clusters for different elephants
- **Attention Maps:** Clear focus on biologically relevant regions

### Model Size:

- **Parameters:** ~9M
- **File Size:** ~35 MB (FP32)
- **Inference Time:** ~10-15ms per image (GPU)

---

## 🔄 Iterative Training & Checkpoint Restoration

### Resume from Checkpoint (Enhanced Notebook V2)

The enhanced notebook automatically detects and offers to resume from `latest_checkpoint.pth`.

To manually load a checkpoint:
```python
# The notebook will prompt you during setup
# Or manually load:
resume_checkpoint = CHECKPOINT_DIR / 'checkpoint_epoch_50.pth'
if resume_checkpoint.exists():
    start_epoch, train_losses, val_losses, best_val_loss = load_checkpoint(
        model, optimizer, scheduler, resume_checkpoint
    )
```

### Resume from Checkpoint (Original Notebook)

```python
# Load checkpoint
checkpoint = torch.load('/kaggle/input/previous-run/best_model.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch'] + 1

# Continue training
for epoch in range(start_epoch, NUM_EPOCHS):
    # ... training loop ...
```

---

## 📚 Additional Resources

- **Kaggle GPU Docs:** [kaggle.com/docs/notebooks](https://www.kaggle.com/docs/notebooks)
- **PyTorch on Kaggle:** [kaggle.com/docs/efficient-gpu-usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- **Model Documentation:** `../docs/implementation/Phase_C_README.md`

---

## ✅ Checklist

Before starting training on Kaggle:

- [ ] Dataset uploaded to Kaggle
- [ ] GPU accelerator enabled
- [ ] Dataset added to notebook
- [ ] All model files copied to notebook
- [ ] Paths configured correctly
- [ ] Internet enabled (for package installation)
- [ ] Checkpoint directory created

---

## 🎉 Success Criteria

Your training is successful when:

1. ✅ Training loss decreases consistently
2. ✅ Validation loss decreases (may plateau)
3. ✅ No "Out of Memory" errors
4. ✅ Checkpoints are saved regularly
5. ✅ Attention maps show biological focus
6. ✅ Model file is downloaded successfully

---

## 💡 Pro Tips

1. **Use Kaggle's 30 hours/week wisely** - Train in batches if needed
2. **Save intermediate checkpoints** - Don't lose progress if session times out
3. **Version your datasets** - Keep track of data changes
4. **Use Kaggle Datasets for outputs** - Easier to share and download
5. **Monitor GPU usage** - Use `!nvidia-smi` to check utilization

---

**Ready to train? Follow the steps above and you'll have a trained model in 2-3 hours!** 🚀
