import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# ===============================
# Resource path helper (CRITICAL)
# ===============================
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ===============================
# Load Model + Gallery
# ===============================
MODEL_PATH = resource_path("makhna_model.pth")
GALLERY_PATH = resource_path("gallery_embeddings.pt")

device = torch.device("cpu")

# Import our model architecture
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from src.models.dual_branch_extractor import DualBranchFeatureExtractor
except ImportError:
    # For packaged version, we need to add the model definition inline
    # We'll handle this after testing
    pass

def load_model():
    """Load the trained elephant re-identification model"""
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Extract model parameters
    embedding_dim = checkpoint.get('embedding_dim', 128)
    num_classes = checkpoint.get('num_classes', 19)
    
    # Initialize model architecture
    model = DualBranchFeatureExtractor(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        use_bam=True
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model.to(device)
    return model

def load_gallery():
    """Load gallery embeddings and labels"""
    gallery_data = torch.load(GALLERY_PATH, map_location=device)
    embeddings = gallery_data['embeddings']  # shape: [N, embedding_dim]
    labels = gallery_data['labels']  # shape: [N]
    idx_to_identity = gallery_data['idx_to_identity']
    
    return embeddings, labels, idx_to_identity

print("Loading model...")
model = load_model()
print("Loading gallery...")
gallery_embeddings, gallery_labels, idx_to_identity = load_gallery()
print(f"Gallery loaded: {len(set(gallery_labels.numpy()))} unique elephants, {len(gallery_labels)} images")

# ===============================
# Image Transform (match training)
# ===============================
transform = transforms.Compose([
    transforms.Resize((256, 128)),  # Match training size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# Prediction Function
# ===============================
def predict_image(image_path):
    """Predict elephant ID from image"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Model returns (embedding, logits) or just embedding
        output = model(image_tensor)
        if isinstance(output, tuple):
            embedding = output[0]
        else:
            embedding = output
        
        # Normalize embedding
        embedding = F.normalize(embedding, p=2, dim=1)

        # Compare with gallery
        similarities = torch.mm(embedding, gallery_embeddings.T)
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[0][best_idx].item()
        
        # Get identity label
        label_idx = gallery_labels[best_idx].item()
        predicted_identity = idx_to_identity[label_idx]

    return predicted_identity, best_score

# ===============================
# GUI
# ===============================
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select Elephant Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if file_path:
        try:
            identity, score = predict_image(file_path)
            
            # Convert similarity to confidence percentage
            confidence = (score + 1) / 2 * 100  # Map from [-1, 1] to [0, 100]
            
            # Determine if it's a confident match
            if score > 0.4:  # Threshold
                result_text = f"✅ Match Found!\n\nElephant ID: {identity}\nConfidence: {confidence:.1f}%\nSimilarity: {score:.3f}"
                result_label.config(text=result_text, fg="green")
            else:
                result_text = f"❓ Unknown Elephant\n\nClosest match: {identity}\nConfidence: {confidence:.1f}%\n(Below threshold)"
                result_label.config(text=result_text, fg="orange")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

# Create main window
app = tk.Tk()
app.title("🐘 Elephant ID - Offline Biometric System")
app.geometry("500x350")
app.resizable(False, False)

# Title
title = tk.Label(app, text="Elephant Biometric System", font=("Arial", 18, "bold"), fg="#2E7D32")
title.pack(pady=20)

# Info label
info = tk.Label(app, text=f"Database: {len(set(gallery_labels.numpy()))} registered elephants", 
                font=("Arial", 10), fg="#555")
info.pack(pady=5)

# Upload button
btn = tk.Button(app, text="📁 Upload Elephant Image", command=upload_image, 
                font=("Arial", 12), bg="#4CAF50", fg="white", 
                padx=20, pady=10, relief="raised", borderwidth=2)
btn.pack(pady=30)

# Result label
result_label = tk.Label(app, text="No image uploaded yet", font=("Arial", 11), 
                        fg="#666", justify="center")
result_label.pack(pady=10, padx=20)

# Footer
footer = tk.Label(app, text="Wildlife Institute of India | Makhna Model v1.0", 
                  font=("Arial", 8), fg="#999")
footer.pack(side="bottom", pady=10)

app.mainloop()
