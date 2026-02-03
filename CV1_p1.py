import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class ColorImageProcessor:
    """Class in charge of processing color images"""
    
    def __init__(self, image_path):
        """
        Initialize with image path
        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self.image_name = Path(image_path).stem
        # Load ảnh màu (BGR format của OpenCV)
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise ValueError(f"Cannot load image: {image_path}")
        # Convert to RGB for correct color display
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        print(f"Image loaded: {image_path}")
        print(f"  Size: {self.img_rgb.shape}")
        
    def convert_to_grayscale(self):
        """
        Convert color image to grayscale
        
        Returns:
            Gray image using two methods:
        """
        # Method 1: Using OpenCV built-in function
        gray_cv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Method 2: Manual conversion using luminosity method
        R = self.img_rgb[:,:,0]
        G = self.img_rgb[:,:,1]
        B = self.img_rgb[:,:,2]
        gray_manual = 0.299 * R + 0.587 * G + 0.114 * B
        gray_manual = gray_manual.astype(np.uint8)
        
        print(f"Converted to grayscale")
        return gray_cv, gray_manual
    
    def convert_grayscale_to_color(self, gray_image):
        """
        Convert grayscale image to color (pseudo-color)
        
        Args:
            gray_image: Grayscale image (1 channel)
        Returns:
            Color image (3 channels) by replicating gray channel
        """
        # Copy grayscale channel to all 3 RGB channels
        color_from_gray = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        print(f"Converted grayscale to pseudo-color (3 channels)")
        return color_from_gray
    
    def split_channels(self):
        """
        Split the color channels R, G, B
    
        
        Returns:
            Tuple (R, G, B) - individual color channels
        """
        R = self.img_rgb[:,:,0]
        G = self.img_rgb[:,:,1]
        B = self.img_rgb[:,:,2]
        
        print(f"Successfully split into 3 color channels")
        print(f"  - Red channel: {R.shape}")
        print(f"  - Green channel: {G.shape}")
        print(f"  - Blue channel: {B.shape}")
        
        return R, G, B
    
    def reconstruct_from_channels(self, R, G, B):
        """
        Reconstruct color image from 3 channels
        
        Args:
            R, G, B: Individual color channels
        Returns:
            Reconstructed color image
        """
        reconstructed = np.stack([R, G, B], axis=2)
        print(f"Successfully reconstructed color image from 3 channels")
        return reconstructed
    
    def create_color_variations(self, R, G, B):
        """
        Create color variations by swapping/replacing channels
        
        Args:
            R, G, B: Original color channels
        Returns:
            Dictionary containing variations
        """
        variations = {}
        
        # 1. RGB → BGR (swap Red and Blue)
        variations['BGR'] = np.stack([B, G, R], axis=2)
        
        # 2. RGB → GBR (rotate channels)
        variations['GBR'] = np.stack([G, B, R], axis=2)
        
        # 3. RGB → RBG
        variations['RBG'] = np.stack([R, B, G], axis=2)
        
        # 4. Only keep Red channel
        variations['Only_Red'] = np.stack([R, np.zeros_like(G), np.zeros_like(B)], axis=2)
        
        # 5. Only keep Green channel
        variations['Only_Green'] = np.stack([np.zeros_like(R), G, np.zeros_like(B)], axis=2)
        
        # 6. Only keep Blue channel
        variations['Only_Blue'] = np.stack([np.zeros_like(R), np.zeros_like(G), B], axis=2)
        
        # 7. Remove Red channel (Cyan)
        variations['No_Red_Cyan'] = np.stack([np.zeros_like(R), G, B], axis=2)
        
        # 8. Remove Green channel (Magenta)
        variations['No_Green_Magenta'] = np.stack([R, np.zeros_like(G), B], axis=2)
        
        # 9. Remove Blue channel (Yellow)
        variations['No_Blue_Yellow'] = np.stack([R, G, np.zeros_like(B)], axis=2)
        
        print(f"Successfully created {len(variations)} color variations")
        return variations
    
    def visualize_all(self):
        """Display all processing results"""
        
        # 1. Convert color to grayscale
        gray_cv, gray_manual = self.convert_to_grayscale()
        
        # 2. Convert grayscale to "color"
        color_from_gray = self.convert_grayscale_to_color(gray_cv)
        
        # 3. Split channels
        R, G, B = self.split_channels()
        
        # 4. Reconstruct image
        reconstructed = self.reconstruct_from_channels(R, G, B)
        
        # 5. Create variations
        variations = self.create_color_variations(R, G, B)
        
        # PLOT RESULTS
        self._plot_results(gray_cv, gray_manual, color_from_gray, 
                          R, G, B, reconstructed, variations)
    
    def _plot_results(self, gray_cv, gray_manual, color_from_gray,
                     R, G, B, reconstructed, variations):
        """Display all results"""
        
        # Figure 1: Original image and grayscale conversion
        fig1 = plt.figure(figsize=(16, 10))
        fig1.suptitle(f'IMAGE PROCESSING: {self.image_name}', fontsize=16, fontweight='bold')
        
        # Row 1: Original image and conversions
        plt.subplot(3, 4, 1)
        plt.imshow(self.img_rgb)
        plt.title('Original Image (RGB)', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(3, 4, 2)
        plt.imshow(gray_cv, cmap='gray')
        plt.title('Grayscale Image (OpenCV)', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(3, 4, 3)
        plt.imshow(gray_manual, cmap='gray')
        plt.title('Grayscale Image (Manual)\n0.299R+0.587G+0.114B', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(3, 4, 4)
        plt.imshow(color_from_gray)
        plt.title('Gray → "Color"\n(Pseudo-color)', fontweight='bold')
        plt.axis('off')
        
        # Row 2: Individual color channels
        plt.subplot(3, 4, 5)
        plt.imshow(R, cmap='gray')
        plt.title('Red Channel', fontweight='bold', color='red')
        plt.axis('off')
        
        plt.subplot(3, 4, 6)
        plt.imshow(G, cmap='gray')
        plt.title('Green Channel', fontweight='bold', color='green')
        plt.axis('off')
        
        plt.subplot(3, 4, 7)
        plt.imshow(B, cmap='gray')
        plt.title('Blue Channel', fontweight='bold', color='blue')
        plt.axis('off')
        
        plt.subplot(3, 4, 8)
        plt.imshow(reconstructed)
        plt.title('Reconstructed from R+G+B\n(Same as original)', fontweight='bold')
        plt.axis('off')
        
        # Row 3: Some color variations
        plt.subplot(3, 4, 9)
        plt.imshow(variations['Only_Red'])
        plt.title('Only Red Channel', fontweight='bold', color='red')
        plt.axis('off')
        
        plt.subplot(3, 4, 10)
        plt.imshow(variations['Only_Green'])
        plt.title('Only Green Channel', fontweight='bold', color='green')
        plt.axis('off')
        
        plt.subplot(3, 4, 11)
        plt.imshow(variations['Only_Blue'])
        plt.title('Only Blue Channel', fontweight='bold', color='blue')
        plt.axis('off')
        
        plt.subplot(3, 4, 12)
        plt.imshow(variations['BGR'])
        plt.title('Swap R↔B\n(BGR)', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Figure 2: Other color variations
        fig2 = plt.figure(figsize=(16, 8))
        fig2.suptitle(f'OTHER COLOR VARIATIONS - {self.image_name}', fontsize=16, fontweight='bold')
        
        variation_list = ['GBR', 'RBG', 'No_Red_Cyan', 'No_Green_Magenta', 'No_Blue_Yellow']
        titles = ['GBR (circular shift)', 'RBG', 'No Red (Cyan)', 
                 'No Green (Magenta)', 'No Blue (Yellow)']
        
        for idx, (var_name, title) in enumerate(zip(variation_list, titles)):
            plt.subplot(2, 3, idx + 1)
            plt.imshow(variations[var_name])
            plt.title(title, fontweight='bold')
            plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(self.img_rgb)
        plt.title('Original Image (Reference)', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        return fig1, fig2


def process_all_images(image_folder='./images'):
    """
    Process all images in a folder
    
    Args:
        image_folder: Path to the folder containing images
    """
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    if os.path.exists(image_folder):
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        print(f"⚠ No images found in folder: {image_folder}")
        return
    
    print(f"{'='*70}")
    print(f"COLOR IMAGE PROCESSING PROGRAM")
    print(f"Found {len(image_files)} images")
    print(f"{'='*70}\n")
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        print(f"{'─'*70}")
        
        try:
            processor = ColorImageProcessor(image_path)
            processor.visualize_all()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"\n{'='*70}")
    print(f"COMPLETED PROCESSING ALL IMAGES")
    print(f"{'='*70}")
    
    plt.show()




if __name__ == "__main__":
    
    process_all_images('./images')