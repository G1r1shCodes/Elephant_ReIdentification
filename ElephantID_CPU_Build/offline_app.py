import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
# Model Architecture (Embedded)
# ===============================
class BAM(nn.Module):
    """Biological Attention Map"""
    def __init__(self, in_channels, reduction_ratio=16, dilated=True):
        super(BAM, self).__init__()
        self.in_channels = in_channels
        
        # Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # Spatial Attention
        if dilated:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
                nn.BatchNorm2d(in_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, 3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(in_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, 3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(in_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, 1, 1, bias=False),
                nn.BatchNorm2d(1)
            )
        else:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
                nn.BatchNorm2d(in_channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction_ratio, 1, 1, bias=False),
                nn.BatchNorm2d(1)
            )

    def forward(self, x):
        att_c = self.channel_att(x)
        att_s = self.spatial_att(x)
        att = F.sigmoid(att_c + att_s)
        return x * att, att

class DualBranchFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=None, use_bam=False):
        super().__init__()
        self.use_bam = use_bam
        self.num_classes = num_classes
        
        # ResNet50 backbone
        base_model = models.resnet50(pretrained=False)
        
        self.layer0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Texture Branch
        self.texture_reducer = nn.Conv2d(512, 1024, kernel_size=1)
        if self.use_bam:
            self.texture_bam = BAM(1024)
        
        # Semantic Branch
        if self.use_bam:
            self.semantic_bam = BAM(2048)
        
        # Embedding head
        self.fc = nn.Linear(2048 + 1024, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        
        # Classification
        if self.num_classes:
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
            
    def texture_branch(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat = self.texture_reducer(x)
        
        if self.use_bam:
            feat_att, _ = self.texture_bam(feat)
            return feat_att
        return feat

    def semantic_branch(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.use_bam:
            feat_att, _ = self.semantic_bam(x)
            return feat_att
        return x

    def forward(self, x):
        tex_feat_spatial = self.texture_branch(x)
        tex_feat = self.global_pool(tex_feat_spatial).flatten(1)
        
        sem_feat_spatial = self.semantic_branch(x)
        sem_feat = self.global_pool(sem_feat_spatial).flatten(1)
        
        combined = torch.cat([tex_feat, sem_feat], dim=1)
        embedding_raw = self.fc(combined)
        embedding_raw = self.bn(embedding_raw)
        
        embedding = F.normalize(embedding_raw, p=2, dim=1)
        
        if self.training and self.num_classes:
            logits = self.classifier(embedding_raw)
            return embedding, logits
            
        return embedding

# ===============================
# Load Model
# ===============================
MODEL_PATH = resource_path("makhna_model.pth")
GALLERY_PATH = resource_path("gallery_embeddings.pt")

print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Extract metadata
embedding_dim = checkpoint.get('embedding_dim', 128)
num_classes = checkpoint.get('num_classes', 19)

# Reconstruct model architecture
model = DualBranchFeatureExtractor(
    embedding_dim=embedding_dim,
    num_classes=num_classes,
    use_bam=True  # Model was trained with BAM
)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

print("Loading gallery...")
gallery_data = torch.load(GALLERY_PATH, map_location=device, weights_only=False)
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
        embedding = model(image_tensor)
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        
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
# Enhanced GUI
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
            # Show processing message
            result_label.config(text="🔄 Processing image...", fg="#2196F3")
            app.update()
            
            results = predict_image(file_path)
            current_results = results
            display_results(results)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

def display_results(results):
    """Display Top-5 results with enhanced formatting"""
    if not results:
        result_label.config(text="No matches found", fg="#666")
        return
    
    top = results[0]
    threshold = 0.4
    
    if top['similarity'] > threshold:
        result_text = "╔══════════════════════════════════════╗\n"
        result_text += "║        ✅ MATCH FOUND!                ║\n"
        result_text += "╚══════════════════════════════════════╝\n\n"
        result_text += f"🐘 Elephant ID: {top['identity']}\n"
        result_text += f"📊 Confidence: {top['confidence']:.1f}%\n"
        result_text += f"🎯 Similarity: {top['similarity']:.3f}\n\n"
        
        if len(results) > 1:
            result_text += "─" * 38 + "\n"
            result_text += "📋 Other Candidates:\n\n"
            for i, res in enumerate(results[1:], 2):
                result_text += f"  {i}. {res['identity']}\n"
                result_text += f"     └─ Confidence: {res['confidence']:.1f}%\n"
        
        result_label.config(text=result_text, fg="#2E7D32")
    else:
        result_text = "╔══════════════════════════════════════╗\n"
        result_text += "║     ❓ UNKNOWN ELEPHANT              ║\n"
        result_text += "╚══════════════════════════════════════╝\n\n"
        result_text += f"📍 Closest match: {top['identity']}\n"
        result_text += f"📊 Confidence: {top['confidence']:.1f}%\n"
        result_text += f"⚠️  Below {threshold} threshold\n\n"
        result_text += "💡 This may be a new elephant!\n"
        result_text += "   Consider adding to database."
        
        result_label.config(text=result_text, fg="#FF6F00")

# ===============================
# Create Modern GUI
# ===============================
app = tk.Tk()
app.title("🐘 Elephant Biometric ID System - WII")
app.geometry("700x650")
app.resizable(False, False)
app.configure(bg="#F5F5F5")

# Header Frame
header_frame = tk.Frame(app, bg="#2E7D32", height=100)
header_frame.pack(fill="x")
header_frame.pack_propagate(False)

title = tk.Label(
    header_frame, 
    text="🐘 Elephant Biometric System", 
    font=("Segoe UI", 22, "bold"), 
    fg="white",
    bg="#2E7D32"
)
title.pack(pady=25)

# Database Info Frame
info_frame = tk.Frame(app, bg="#E8F5E9", height=60)
info_frame.pack(fill="x", padx=20, pady=(20, 10))
info_frame.pack_propagate(False)

info_title = tk.Label(
    info_frame, 
    text="📚 Database Statistics", 
    font=("Segoe UI", 11, "bold"), 
    fg="#1B5E20",
    bg="#E8F5E9"
)
info_title.pack(pady=(8, 2))

info_stats = tk.Label(
    info_frame, 
    text=f"{num_elephants} Registered Elephants  •  {num_images} Reference Images", 
    font=("Segoe UI", 10), 
    fg="#2E7D32",
    bg="#E8F5E9"
)
info_stats.pack()

# Upload Button
btn_frame = tk.Frame(app, bg="#F5F5F5")
btn_frame.pack(pady=20)

upload_btn = tk.Button(
    btn_frame, 
    text="📁  UPLOAD ELEPHANT IMAGE", 
    command=upload_image, 
    font=("Segoe UI", 13, "bold"), 
    bg="#4CAF50", 
    fg="white",
    activebackground="#45A049",
    activeforeground="white",
    padx=30, 
    pady=15, 
    relief="flat",
    cursor="hand2",
    borderwidth=0
)
upload_btn.pack()

# Results Frame
results_frame = tk.Frame(app, bg="white", relief="solid", borderwidth=1)
results_frame.pack(fill="both", expand=True, padx=20, pady=(10, 20))

result_label = tk.Label(
    results_frame, 
    text="No image uploaded yet\n\n💡 Click the button above to begin", 
    font=("Consolas", 10), 
    fg="#757575",
    bg="white",
    justify="left",
    wraplength=630,
    padx=20,
    pady=20
)
result_label.pack(fill="both", expand=True)

# Instructions Frame
instructions_frame = tk.Frame(app, bg="#FFF3E0", height=50)
instructions_frame.pack(fill="x", side="bottom")
instructions_frame.pack_propagate(False)

instructions = tk.Label(
    instructions_frame, 
    text="💡 Best results: Upload clear side-profile photos showing ears and tusks", 
    font=("Segoe UI", 9), 
    fg="#E65100",
    bg="#FFF3E0"
)
instructions.pack(pady=15)

# Footer
footer = tk.Label(
    app, 
    text="Wildlife Institute of India  •  CPU-Only Build  •  Fully Offline", 
    font=("Segoe UI", 8), 
    fg="#9E9E9E",
    bg="#F5F5F5"
)
footer.pack(side="bottom", pady=10)

app.mainloop()
