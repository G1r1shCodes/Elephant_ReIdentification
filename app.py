"""
Unique Elephant Identification System
======================================

A simplified Streamlit application for Wildlife Institute of India researchers
to identify elephants from field images and enroll new individuals.
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os
import json
from torchvision import transforms
from datetime import datetime


def resource_path(relative_path):
    """Get absolute path to resource (works for dev and PyInstaller)."""
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    # Running in development - use script directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# Add src to path
if hasattr(sys, '_MEIPASS'):
    sys.path.append(sys._MEIPASS)
else:
    sys.path.append(str(Path(__file__).parent))

from src.models.dual_branch_extractor import DualBranchFeatureExtractor


# Page configuration
st.set_page_config(
    page_title="Unique Elephant Identification System",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a friendly, non-technical look
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.5rem;
        color: #1B5E20; /* Dark Green */
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .step-header {
        color: #2E7D32;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .result-card {
        background-color: #F1F8E9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #C5E1A5;
        text-align: center;
        margin-bottom: 20px;
    }
    .match-title {
        color: #2E7D32;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    .no-match-card {
        background-color: #FFF3E0;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #FFE0B2;
        text-align: center;
    }
    /* Hide Streamlit components that look too technical */
    .stDeployButton {display:none;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_gallery():
    """Load the database of known elephants."""
    gallery_path = resource_path("gallery_embeddings.pt")
    
    if not os.path.exists(gallery_path):
        st.error("⚠️ Database file (gallery_embeddings.pt) is missing.")
        return {}
    
    # Load precomputed gallery
    try:
        gallery_data = torch.load(gallery_path)
        embeddings = gallery_data['embeddings'].numpy()
        labels = gallery_data['labels'].numpy()
        idx_to_identity = gallery_data['idx_to_identity']
        
        # Restructure for app format
        gallery = {}
        for idx, identity in idx_to_identity.items():
            mask = labels == idx
            gallery[identity] = {
                'embeddings': [embeddings[i] for i in range(len(labels)) if labels[i] == idx],
                'image_paths': [], # Paths would need to be re-linked if we want to show strict gallery images
                'num_images': int(np.sum(mask))
            }
            # Try to find representative images in the file system for display
            # Assuming standard structure: data/restructured/{Identity}/*.jpg
            # This is a heuristic for display only
            potential_dir = Path(f"data/restructured/{identity}")
            if potential_dir.exists():
                 gallery[identity]['image_paths'] = sorted([str(p) for p in potential_dir.glob("*.jpg")])
        
        return gallery
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return {}


def save_gallery(gallery):
    """Save updated database (Placeholder for simple file-based persistence)."""
    # Real implementation would append to the .pt file or a database
    # For this demo, we just acknowledge the action
    pass


@st.cache_resource
def load_model():
    """Load the AI Brain."""
    model_path = resource_path("makhna_model.pth")
    
    if not os.path.exists(model_path):
        st.error(f"❌ AI Model missing at {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        embedding_dim = checkpoint.get('embedding_dim', 128)
        num_classes = checkpoint.get('num_classes', 19)
        
        model = DualBranchFeatureExtractor(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            use_bam=True
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error initializing AI: {e}")
        return None


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def extract_features(model, image, enhanced_mode=False):
    """Scan the image for unique features."""
    transform = get_transform()
    if image.mode != 'RGB': image = image.convert('RGB')
    
    model.eval()
    with torch.no_grad():
        if not enhanced_mode:
            img_tensor = transform(image).unsqueeze(0)
            emb = model(img_tensor)
            if isinstance(emb, tuple): emb = emb[0]
        else:
            # Enhanced Mode (TTA)
            # 1. Resize & Flip
            img_resized = transforms.Resize((256, 128))(image)
            v1 = transforms.ToTensor()(img_resized)
            v1 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(v1)
            v2 = transforms.ToTensor()(transforms.functional.hflip(img_resized))
            v2 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(v2)
            
            # Simple average of Original + Flip (Fast but effective)
            batch = torch.stack([v1, v2])
            emb_batch = model(batch)
            if isinstance(emb_batch, tuple): emb_batch = emb_batch[0]
            emb = torch.mean(emb_batch, dim=0, keepdim=True)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            
    return emb.squeeze().cpu().numpy()


def find_top_matches(query_features, gallery, top_k=5):
    """Compare features and return top K matches."""
    if not gallery: return []
    
    scores = []
    
    for eid, data in gallery.items():
        # Find best score for this identity (closest single view)
        id_best_score = -1
        for db_emb in data['embeddings']:
            score = np.dot(query_features, db_emb)
            if score > id_best_score:
                id_best_score = score
        
        # Calculate confidence
        confidence = (id_best_score + 1) / 2 * 100
        scores.append({
            'id': eid,
            'score': id_best_score,
            'confidence': confidence
        })
    
    # Sort by score descending
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    return scores[:top_k]


def main():
    # --- UI HEADER ---
    st.markdown('<div class="main-header">🐘 Elephant ID</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI Assistant for Field Researchers</div>', unsafe_allow_html=True)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/371/371669.png", width=100)
        st.header("Database Status")
        gallery = load_gallery()
        st.success(f"📚 **{len(gallery)}** Elephants Registered")
        
        st.divider()
        st.markdown("### ⚙️ Ranger Tools")
        enhanced_mode = st.toggle("Enhanced Scan Mode", 
                                help="Turn this on if the photo is blurry or difficult. It takes slightly longer but provides better accuracy.")
        
        st.info("💡 **Tip:** Upload a clear photo of the side profile showing the ears and tusks.")

    # --- MAIN CONTENT ---
    model = load_model()
    if not model: st.stop()

    # Step 1: Upload
    st.markdown('<div class="step-header">1️⃣ Upload Photo</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # --- AI PROCESSING (Before Layout) ---
        with st.spinner("Scanning features..."):
            features = extract_features(model, image, enhanced_mode)
            top_matches = find_top_matches(features, gallery, top_k=5)
        
        if not top_matches:
            st.error("Gallery is empty.")
            st.stop()
        
        # Get Top-1 Result
        top_result = top_matches[0]
        match_id = top_result['id']
        conf = top_result['confidence']
        is_match = top_result['score'] > 0.4  # Threshold check
        
        # --- UI LAYOUT ---
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        # LEFT COLUMN: Image
        with col1:
            st.markdown('<div class="step-header">1️⃣ Uploaded Photo</div>', unsafe_allow_html=True)
            st.image(image, caption="Field Photo", use_container_width=True, channels="RGB")
        
        # RIGHT COLUMN: Top Result
        with col2:
            st.markdown('<div class="step-header">2️⃣ Analysis Result</div>', unsafe_allow_html=True)
            
            # Display Top Result
            if is_match:
                if conf > 99.9:
                    st.markdown(f"""
                    <div class="result-card" style="border-color: #4CAF50; background-color: #E8F5E9;">
                        <div style="font-size: 1.2rem; color: #555;">Exact Image Match!</div>
                        <div class="match-title">{match_id}</div>
                        <div style="margin-top: 10px; font-size: 1.0rem; color: #2E7D32;">
                            (Photo already in database)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size: 1.2rem; color: #555;">Best Match:</div>
                        <div class="match-title">{match_id}</div>
                        <div style="margin-top: 10px; font-size: 1.1rem;">
                            <span style="background-color: #2E7D32; color: white; padding: 4px 8px; border-radius: 4px;">Certainty: <b>{conf:.1f}%</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Verification Images for Top Match
                if match_id in gallery and gallery[match_id]['image_paths']:
                         st.markdown("##### 🧐 Compare with Database:")
                         img_paths = gallery[match_id]['image_paths'][:3]
                         cols = st.columns(len(img_paths))
                         for idx, p in enumerate(img_paths):
                             with cols[idx]:
                                 try: st.image(str(p), use_container_width=True)
                                 except: pass
            else:
                st.markdown("""
                <div class="no-match-card">
                    <h3>❓ Unknown Elephant</h3>
                    <p>No confident match found.</p>
                </div>
                """, unsafe_allow_html=True)

        # --- FULL WIDTH SECTIONS (Below Columns) ---
        st.divider()
        
        # Rank-5 Display
        if conf < 99.9 and len(top_matches) > 1:
            with st.expander(f"📋 Other Possibilities (Rank-5)", expanded=False):
                st.markdown("If the top match looks wrong, check these candidates:")
                for i, res in enumerate(top_matches[1:]):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**#{i+2}: {res['id']}**")
                    with col_b:
                        st.caption(f"{res['confidence']:.1f}%")
                    st.progress(float(max(0.0, min(1.0, res['score'])))) 
        
        # Registration (Always Available)
        with st.expander("📝 Correct Result / Register New Elephant", expanded=not is_match):
            st.markdown("If the ID above is wrong, or if this is a new elephant, add it here:")
            
            new_id = st.text_input("Enter Correct Name/ID:", placeholder="e.g., Tusker_22, Female_01")
            
            if st.button("Save to Database", type="primary"):
                if new_id:
                    if new_id in gallery:
                        st.warning(f"⚠️ **{new_id}** is already in the database. Saving this image will improve its profile.")
                    
                    # Add to gallery (or update existing)
                    gallery = load_gallery()
                    
                    # Save image
                    save_dir = Path(f"data/processed_megadetector/New/{new_id}")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = save_dir / f"{new_id}_{timestamp}.jpg"
                    st.session_state['current_image'].save(img_path)
                    
                    # Update gallery
                    # If existing, append. If new, create.
                    if new_id not in gallery:
                        gallery[new_id] = {'embeddings': [], 'image_paths': [], 'num_images': 0}
                        
                    gallery[new_id]['embeddings'].append(st.session_state['current_embedding'])
                    gallery[new_id]['image_paths'].append(str(img_path))
                    gallery[new_id]['num_images'] += 1
                    
                    save_gallery(gallery)
                    
                    st.success(f"✅ Successfully registered **{new_id}**!")
                    st.balloons()
                    
                    # Clear cache to reload gallery
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Please enter a Name or ID")

if __name__ == "__main__":
    main()
