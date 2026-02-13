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
    """Predict elephant ID from image - returns top 5 matches"""
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

        # Compare with gallery - get all similarities
        similarities = torch.mm(embedding, gallery_embeddings.T)
        
        # Get top 5 matches
        top5_scores, top5_indices = torch.topk(similarities[0], k=min(5, len(similarities[0])))
        
        # Build results list
        results = []
        for score, idx in zip(top5_scores, top5_indices):
            label_idx = gallery_labels[idx].item()
            identity = idx_to_identity[label_idx]
            results.append({
                'identity': identity,
                'similarity': score.item(),
                'confidence': (score.item() + 1) / 2 * 100  # Map [-1,1] to [0,100]
            })

    return results

# ===============================
# GUI
# ===============================
current_results = []
current_image_path = None

def upload_image():
    global current_results, current_image_path
    
    file_path = filedialog.askopenfilename(
        title="Select Elephant Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if file_path:
        try:
            current_image_path = file_path
            results = predict_image(file_path)
            current_results = results
            
            # Display results
            display_results(results)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

def display_results(results):
    """Display Top-5 results in the GUI"""
    if not results:
        result_label.config(text="No matches found", fg="#666")
        return
    
    # Top match
    top = results[0]
    threshold = 0.4
    
    if top['similarity'] > threshold:
        # Confident match
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
        # Unknown elephant
        result_text = f"❓ UNKNOWN ELEPHANT\n\n"
        result_text += f"Closest match: {top['identity']}\n"
        result_text += f"Confidence: {top['confidence']:.1f}%\n"
        result_text += f"(Below {threshold} threshold)\n\n"
        result_text += "This may be a new elephant!"
        
        result_label.config(text=result_text, fg="orange", justify="left")

# Create main window
app = tk.Tk()
app.title("🐘 Elephant ID - Offline Biometric System")
app.geometry("600x500")
app.resizable(False, False)

# Title
title = tk.Label(app, text="Elephant Biometric System", font=("Arial", 18, "bold"), fg="#2E7D32")
title.pack(pady=20)

# Database info
num_elephants = len(set(gallery_labels.numpy()))
num_images = len(gallery_labels)
info = tk.Label(app, text=f"📚 Database: {num_elephants} elephants | {num_images} images", 
                font=("Arial", 10), fg="#555")
info.pack(pady=5)

# Upload button
btn = tk.Button(app, text="📁 Upload Elephant Image", command=upload_image, 
                font=("Arial", 12), bg="#4CAF50", fg="white", 
                padx=20, pady=10, relief="raised", borderwidth=2)
btn.pack(pady=30)

# Result label (larger area for Top-5)
result_label = tk.Label(app, text="No image uploaded yet", font=("Arial", 10), 
                        fg="#666", justify="left", wraplength=550)
result_label.pack(pady=10, padx=20, fill="both", expand=True)

# Instructions
instructions = tk.Label(app, text="💡 Upload a clear side-profile photo showing ears and tusks", 
                       font=("Arial", 9), fg="#777")
instructions.pack(pady=10)

# Footer
footer = tk.Label(app, text="Wildlife Institute of India | Makhna Model v1.0 | Fully Offline", 
                  font=("Arial", 8), fg="#999")
footer.pack(side="bottom", pady=10)

app.mainloop()
