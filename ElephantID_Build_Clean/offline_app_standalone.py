import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog, messagebox

# Force CPU (no CUDA bloat)
device = torch.device("cpu")

# ===============================
# Resource path helper (CRITICAL)
# ===============================
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# ===============================
# Minimal Model Definition
# ===============================
# Since we can't import from 'src', we'll load the model directly from the checkpoint
# PyTorch can reconstruct the model from the saved state_dict if we provide the architecture

# Load model and gallery
MODEL_PATH = resource_path("makhna_model.pth")
GALLERY_PATH = resource_path("gallery_embeddings.pt")

print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# For standalone .exe, we need to handle the model loading differently
# The model file should contain the full model, not just state_dict
try:
    if 'model_state_dict' in checkpoint:
        # If it's a training checkpoint, we need the architecture
        # For simplicity in standalone, we'll try to load the full model
        raise ValueError("Need full model, not just state_dict")
    else:
        # Assume checkpoint IS the full model
        model = checkpoint
        model.eval()
        model.to(device)
except:
    # Fallback: treat checkpoint as the model itself
    model = checkpoint
    if hasattr(model, 'eval'):
        model.eval()
        model.to(device)

print("Loading gallery...")
gallery_data = torch.load(GALLERY_PATH, map_location=device)
gallery_embeddings = gallery_data['embeddings']
gallery_labels = gallery_data['labels']
idx_to_identity = gallery_data['idx_to_identity']

num_elephants = len(set(gallery_labels.numpy()))
num_images = len(gallery_labels)
print(f"Gallery loaded: {num_elephants} unique elephants, {num_images} images")

# ===============================
# Image Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===============================
# Prediction Function
# ===============================
def predict_image(image_path):
    """Predict elephant ID - returns top 5 matches"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            embedding = output[0]
        else:
            embedding = output
        
        embedding = F.normalize(embedding, p=2, dim=1)
        similarities = torch.mm(embedding, gallery_embeddings.T)
        
        top5_scores, top5_indices = torch.topk(similarities[0], k=min(5, len(similarities[0])))
        
        results = []
        for score, idx in zip(top5_scores, top5_indices):
            label_idx = gallery_labels[idx].item()
            identity = idx_to_identity[label_idx]
            results.append({
                'identity': identity,
                'similarity': score.item(),
                'confidence': (score.item() + 1) / 2 * 100
            })

    return results

# ===============================
# GUI
# ===============================
current_results = []

def upload_image():
    global current_results
    
    file_path = filedialog.askopenfilename(
        title="Select Elephant Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if file_path:
        try:
            results = predict_image(file_path)
            current_results = results
            display_results(results)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

def display_results(results):
    """Display Top-5 results"""
    if not results:
        result_label.config(text="No matches found", fg="#666")
        return
    
    top = results[0]
    threshold = 0.4
    
    if top['similarity'] > threshold:
        result_text = f"✅ MATCH FOUND!\n\n"
        result_text += f"Elephant ID: {top['identity']}\n"
        result_text += f"Confidence: {top['confidence']:.1f}%\n"
        result_text += f"Similarity: {top['similarity']:.3f}\n\n"
        
        if len(results) > 1:
            result_text += "─" * 30 + "\n"
            result_text += "Other Candidates:\n"
            for i, res in enumerate(results[1:], 2):
                result_text += f"#{i}: {res['identity']} ({res['confidence']:.1f}%)\n"
        
        result_label.config(text=result_text, fg="green", justify="left")
    else:
        result_text = f"❓ UNKNOWN ELEPHANT\n\n"
        result_text += f"Closest match: {top['identity']}\n"
        result_text += f"Confidence: {top['confidence']:.1f}%\n"
        result_text += f"(Below {threshold} threshold)\n\n"
        result_text += "This may be a new elephant!"
        
        result_label.config(text=result_text, fg="orange", justify="left")

# Create GUI
app = tk.Tk()
app.title("🐘 Elephant ID - Offline System")
app.geometry("600x500")
app.resizable(False, False)

title = tk.Label(app, text="Elephant Biometric System", font=("Arial", 18, "bold"), fg="#2E7D32")
title.pack(pady=20)

info = tk.Label(app, text=f"📚 Database: {num_elephants} elephants | {num_images} images", 
                font=("Arial", 10), fg="#555")
info.pack(pady=5)

btn = tk.Button(app, text="📁 Upload Elephant Image", command=upload_image, 
                font=("Arial", 12), bg="#4CAF50", fg="white", 
                padx=20, pady=10, relief="raised", borderwidth=2)
btn.pack(pady=30)

result_label = tk.Label(app, text="No image uploaded yet", font=("Arial", 10), 
                        fg="#666", justify="left", wraplength=550)
result_label.pack(pady=10, padx=20, fill="both", expand=True)

instructions = tk.Label(app, text="💡 Upload a clear side-profile photo showing ears and tusks", 
                       font=("Arial", 9), fg="#777")
instructions.pack(pady=10)

footer = tk.Label(app, text="WII | CPU-Only Build | Fully Offline", 
                  font=("Arial", 8), fg="#999")
footer.pack(side="bottom", pady=10)

app.mainloop()
