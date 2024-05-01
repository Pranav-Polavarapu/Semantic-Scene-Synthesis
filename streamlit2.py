import streamlit as st
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
import re  # For pattern matching
import sys
import argparse
sys.path.append('/Users/pranavpolavarapu/DAGAN/DAGAN_v1')
from models.networks import define_G

# Function to display images
def display_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(titles[i])
    st.pyplot(fig)

# Define the dataset class
class CityscapesSubset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = [img for img in os.listdir(images_dir) if img.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label_path = os.path.join(self.labels_dir, self.images[index].replace('leftImg8bit', 'gtFine_labelIds'))
        label = Image.open(label_path).convert('RGB')
        if self.transform:
            label = self.transform(label)
        return label, self.images[index]

# Set base directories
base_path = '/Users/pranavpolavarapu/Downloads/autonomous/cityscapes/data/'
subset_images_dir = '/Users/pranavpolavarapu/Downloads/autonomous/cityscapes/subset2/images/'
subset_labels_dir = '/Users/pranavpolavarapu/Downloads/autonomous/cityscapes/subset2/labels/'

# Model and transformation setup
opts = argparse.Namespace()
opts.ngf = 64
opts.z_dim = 256
opts.semantic_nc = 3
opts.num_upsampling_layers = 'normal'
opts.crop_size = 512
opts.aspect_ratio = 1.0
opts.use_vae = False
opts.norm_G = 'spectralspadesyncbatch3x3'
opts.netG = 'spade'
opts.init_type = 'normal'
opts.init_variance = 0.02
opts.gpu_ids = []

generator = define_G(opts)
generator.eval()

checkpoint_path = 'FinalModel.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['model_state_dict'])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Streamlit app setup
st.set_page_config(page_title="Semantic Scene Synthesis", layout="wide")
page = st.sidebar.selectbox("Go to Page", ["Home", "Generator"])

if page == "Home":
    st.title("Semantic Scene Synthesis")
    st.subheader("CSE 4/573 PROJECT")
    st.markdown("""
    **Team: Autonomous**
    - Pranav Polavarapu
    - Paige Shoemaker
    """)
    st.markdown("""
    ## Instructions
    Follow these steps to transform semantic labels into breathtaking scenes:
    - **Navigate** to the 'Generator' page via the sidebar.
    - **Select** an input label image from the displayed semantic map options.
    - **Input Semantic Map options keep getting refreshed at every reload**
    - **Generate** the scene by pressing the 'Generate Image' button.
    - **View** the results, and regenerate as needed.
    """)

elif page == "Generator":
    batch_size = st.sidebar.selectbox("Select batch size for display:", [5, 10, 15, 20], index=0)
    label_dataset = CityscapesSubset(subset_labels_dir, subset_labels_dir, transform)
    label_dataloader = DataLoader(label_dataset, batch_size=batch_size, shuffle=True)

    st.title("Autonomous Scene Synthesis")
    st.markdown("Select an input label image from the grid below to generate a scene:")

    label_images, file_names = next(iter(label_dataloader))
    cols_per_row = 3
    rows = [st.columns(cols_per_row) for _ in range((len(label_images) + cols_per_row - 1) // cols_per_row)]
    selected_idx = None
    selected_label_image = None

    idx = 0
    for row in rows:
        for col in row:
            if idx < len(label_images):
                with col:
                    label_img_np = label_images[idx].permute(1, 2, 0).numpy()
                    st.image(label_img_np, caption=file_names[idx], width=150)
                    if st.checkbox("Select this image", key=idx):
                        selected_idx = idx
                        selected_label_image = label_images[idx]
            idx += 1

    if selected_idx is not None:
        with st.spinner('Generating image...'):
            image_id = re.match(r'([a-zA-Z]+_[0-9]+_[0-9]+)', file_names[selected_idx]).group(1)
            selected_image_path = os.path.join(subset_images_dir, f'{image_id}_leftImg8bit.png')
            selected_image = Image.open(selected_image_path).convert('RGB')
            selected_image = transform(selected_image).unsqueeze(0)
            output = generator(selected_image)
            generated_img = output.squeeze().detach().numpy()
            generated_img = np.transpose(generated_img, (1, 2, 0))
            norm_image = (generated_img - generated_img.min()) / (generated_img.max() - generated_img.min())
            gamma_corrected = np.power(norm_image, 0.5)
            label_img_np = selected_label_image.permute(1, 2, 0).numpy()
            display_images([label_img_np, norm_image, gamma_corrected],
                           ['Label Image', 'Raw Generated Image', 'Gamma Corrected Image'])
            st.success('Image generation completed!')
