import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14

# Define the path to the dataset
data_dir = Path('/Users/onurakyuz/Desktop/Brain Tumor MRI/data')
output_dir = Path('/Users/onurakyuz/Desktop/Brain Tumor MRI/analysis_results')
os.makedirs(output_dir, exist_ok=True)

# Function to count images in each class
def count_images():
    counts = {'Training': {}, 'Testing': {}}
    total_counts = {'Training': 0, 'Testing': 0}
    
    for split in ['Training', 'Testing']:
        split_dir = data_dir / split
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                num_images = len(list(class_dir.glob('*.jpg')))
                counts[split][class_name] = num_images
                total_counts[split] += num_images
    
    return counts, total_counts

# Function to plot class distribution
def plot_class_distribution(counts):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for i, split in enumerate(['Training', 'Testing']):
        ax = axes[i]
        class_names = list(counts[split].keys())
        class_counts = list(counts[split].values())
        
        # Sort by class name for consistency
        sorted_indices = sorted(range(len(class_names)), key=lambda k: class_names[k])
        class_names = [class_names[i] for i in sorted_indices]
        class_counts = [class_counts[i] for i in sorted_indices]
        
        # Create a bar plot
        bars = sns.barplot(x=class_names, y=class_counts, ax=ax, palette='viridis')
        ax.set_title(f'{split} Set Class Distribution', fontsize=16)
        ax.set_xlabel('Class', fontsize=14)
        ax.set_ylabel('Number of Images', fontsize=14)
        
        # Add count labels on top of bars
        for j, count in enumerate(class_counts):
            ax.text(j, count + 20, str(count), ha='center', fontsize=12)
            
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to display sample images from each class
def display_sample_images(num_samples=3):
    fig, axes = plt.subplots(4, num_samples, figsize=(15, 12))
    
    # Get class names from Training set
    class_names = sorted([d.name for d in (data_dir / 'Training').iterdir() if d.is_dir()])
    
    for i, class_name in enumerate(class_names):
        class_dir = data_dir / 'Training' / class_name
        image_paths = list(class_dir.glob('*.jpg'))
        
        # Randomly select images
        selected_images = random.sample(image_paths, min(num_samples, len(image_paths)))
        
        for j, img_path in enumerate(selected_images):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            
            axes[i, j].imshow(img)
            axes[i, j].set_title(f'{class_name}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Sample Images from Each Class', fontsize=16, y=1.02)
    plt.savefig(output_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to analyze image properties
def analyze_image_properties(max_per_class=100):
    print("Analyzing image properties...")
    
    # Lists to store image properties
    widths = []
    heights = []
    aspect_ratios = []
    sizes = []  # in KB
    classes = []
    splits = []
    
    for split in ['Training', 'Testing']:
        split_dir = data_dir / split
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                img_paths = list(class_dir.glob('*.jpg'))
                
                # Sample a subset of images
                selected_paths = random.sample(img_paths, min(max_per_class, len(img_paths)))
                
                for img_path in tqdm(selected_paths, desc=f"{split} - {class_name}"):
                    # Get image size in KB
                    size_kb = os.path.getsize(img_path) / 1024
                    
                    # Open image and get dimensions
                    with Image.open(img_path) as img:
                        width, height = img.size
                        aspect_ratio = width / height
                    
                    # Append to lists
                    widths.append(width)
                    heights.append(height)
                    aspect_ratios.append(aspect_ratio)
                    sizes.append(size_kb)
                    classes.append(class_name)
                    splits.append(split)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Split': splits,
        'Class': classes,
        'Width': widths,
        'Height': heights,
        'Aspect_Ratio': aspect_ratios,
        'Size_KB': sizes
    })
    
    # Save the DataFrame to CSV
    df.to_csv(output_dir / 'image_properties.csv', index=False)
    
    return df

# Function to visualize image properties
def visualize_image_properties(df):
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Distribution of image widths
    sns.histplot(data=df, x='Width', hue='Class', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Image Widths', fontsize=16)
    axes[0, 0].set_xlabel('Width (pixels)', fontsize=14)
    axes[0, 0].set_ylabel('Count', fontsize=14)
    
    # Plot 2: Distribution of image heights
    sns.histplot(data=df, x='Height', hue='Class', kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Image Heights', fontsize=16)
    axes[0, 1].set_xlabel('Height (pixels)', fontsize=14)
    axes[0, 1].set_ylabel('Count', fontsize=14)
    
    # Plot 3: Distribution of aspect ratios
    sns.histplot(data=df, x='Aspect_Ratio', hue='Class', kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Aspect Ratios', fontsize=16)
    axes[1, 0].set_xlabel('Aspect Ratio (Width/Height)', fontsize=14)
    axes[1, 0].set_ylabel('Count', fontsize=14)
    
    # Plot 4: Distribution of file sizes
    sns.histplot(data=df, x='Size_KB', hue='Class', kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of File Sizes', fontsize=16)
    axes[1, 1].set_xlabel('Size (KB)', fontsize=14)
    axes[1, 1].set_ylabel('Count', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'image_properties_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Box plots for each property by class
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Box plot of image widths by class
    sns.boxplot(data=df, x='Class', y='Width', ax=axes[0, 0])
    axes[0, 0].set_title('Image Widths by Class', fontsize=16)
    axes[0, 0].set_xlabel('Class', fontsize=14)
    axes[0, 0].set_ylabel('Width (pixels)', fontsize=14)
    
    # Plot 2: Box plot of image heights by class
    sns.boxplot(data=df, x='Class', y='Height', ax=axes[0, 1])
    axes[0, 1].set_title('Image Heights by Class', fontsize=16)
    axes[0, 1].set_xlabel('Class', fontsize=14)
    axes[0, 1].set_ylabel('Height (pixels)', fontsize=14)
    
    # Plot 3: Box plot of aspect ratios by class
    sns.boxplot(data=df, x='Class', y='Aspect_Ratio', ax=axes[1, 0])
    axes[1, 0].set_title('Aspect Ratios by Class', fontsize=16)
    axes[1, 0].set_xlabel('Class', fontsize=14)
    axes[1, 0].set_ylabel('Aspect Ratio (Width/Height)', fontsize=14)
    
    # Plot 4: Box plot of file sizes by class
    sns.boxplot(data=df, x='Class', y='Size_KB', ax=axes[1, 1])
    axes[1, 1].set_title('File Sizes by Class', fontsize=16)
    axes[1, 1].set_xlabel('Class', fontsize=14)
    axes[1, 1].set_ylabel('Size (KB)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'image_properties_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to analyze pixel intensities
def analyze_pixel_intensities(num_samples=25):
    print("Analyzing pixel intensities...")
    
    # Get class names from Training set
    class_names = sorted([d.name for d in (data_dir / 'Training').iterdir() if d.is_dir()])
    
    # Dictionary to store mean pixel intensities for each class
    class_intensities = {}
    class_intensities_std = {}
    
    for class_name in class_names:
        # List to store mean intensities for this class
        intensities = []
        std_devs = []
        
        # Get image paths from both Training and Testing sets
        image_paths = []
        for split in ['Training', 'Testing']:
            class_dir = data_dir / split / class_name
            if class_dir.exists():
                image_paths.extend(list(class_dir.glob('*.jpg')))
        
        # Randomly select images
        selected_images = random.sample(image_paths, min(num_samples, len(image_paths)))
        
        for img_path in tqdm(selected_images, desc=f"Pixel analysis - {class_name}"):
            # Read the image in grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            # Calculate mean pixel intensity and standard deviation
            mean_intensity = np.mean(img)
            std_dev = np.std(img)
            
            intensities.append(mean_intensity)
            std_devs.append(std_dev)
        
        class_intensities[class_name] = intensities
        class_intensities_std[class_name] = std_devs
    
    # Create DataFrames
    intensity_df = pd.DataFrame(class_intensities)
    std_dev_df = pd.DataFrame(class_intensities_std)
    
    # Save to CSV
    intensity_df.to_csv(output_dir / 'pixel_intensities.csv', index=False)
    std_dev_df.to_csv(output_dir / 'pixel_std_devs.csv', index=False)
    
    # Plot the distribution of mean pixel intensities
    plt.figure(figsize=(14, 8))
    
    # Create violin plots
    sns.violinplot(data=intensity_df, palette='viridis')
    
    plt.title('Distribution of Mean Pixel Intensities by Class', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Mean Pixel Intensity', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_dir / 'pixel_intensities_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot the distribution of pixel standard deviations
    plt.figure(figsize=(14, 8))
    
    # Create violin plots
    sns.violinplot(data=std_dev_df, palette='viridis')
    
    plt.title('Distribution of Pixel Standard Deviations by Class', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Pixel Standard Deviation', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_dir / 'pixel_std_devs_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return intensity_df, std_dev_df

# Function to show image preprocessing steps
def show_preprocessing_steps():
    # Get class names from Training set
    class_names = sorted([d.name for d in (data_dir / 'Training').iterdir() if d.is_dir()])
    
    # Randomly select one class
    class_name = random.choice(class_names)
    class_dir = data_dir / 'Training' / class_name
    
    # Randomly select one image
    img_path = random.choice(list(class_dir.glob('*.jpg')))
    
    # Read the image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # Resize the image
    img_resized = cv2.resize(img, (224, 224))
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Apply data augmentation
    # Rotation
    rows, cols = img_resized.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    img_rotated = cv2.warpAffine(img_resized, M, (cols, rows))
    
    # Horizontal flip
    img_flipped = cv2.flip(img_resized, 1)
    
    # Display the preprocessing steps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'Original Image\n{class_name}\nSize: {img.shape[1]}x{img.shape[0]}', fontsize=14)
    axes[0, 0].axis('off')
    
    # Resized image
    axes[0, 1].imshow(img_resized)
    axes[0, 1].set_title(f'Resized Image\nSize: 224x224', fontsize=14)
    axes[0, 1].axis('off')
    
    # Normalized image
    axes[0, 2].imshow(img_normalized)
    axes[0, 2].set_title('Normalized Image\nPixel Values: [0, 1]', fontsize=14)
    axes[0, 2].axis('off')
    
    # Rotated image
    axes[1, 0].imshow(img_rotated)
    axes[1, 0].set_title('Rotated Image\n(15 degrees)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Flipped image
    axes[1, 1].imshow(img_flipped)
    axes[1, 1].set_title('Horizontally Flipped Image', fontsize=14)
    axes[1, 1].axis('off')
    
    # Grayscale image
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    axes[1, 2].imshow(img_gray, cmap='gray')
    axes[1, 2].set_title('Grayscale Image', fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Image Preprocessing Steps', fontsize=16, y=1.02)
    plt.savefig(output_dir / 'preprocessing_steps.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to analyze color histograms
def analyze_color_histograms():
    print("Analyzing color histograms...")
    
    # Get class names from Training set
    class_names = sorted([d.name for d in (data_dir / 'Training').iterdir() if d.is_dir()])
    
    # Create a figure
    fig, axes = plt.subplots(len(class_names), 3, figsize=(18, 5*len(class_names)))
    
    for i, class_name in enumerate(class_names):
        class_dir = data_dir / 'Training' / class_name
        
        # Randomly select one image
        img_path = random.choice(list(class_dir.glob('*.jpg')))
        
        # Read the image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'{class_name} Sample', fontsize=14)
        axes[i, 0].axis('off')
        
        # Calculate histograms
        color = ('r', 'g', 'b')
        for j, col in enumerate(color):
            hist = cv2.calcHist([img], [j], None, [256], [0, 256])
            axes[i, 1].plot(hist, color=col)
            axes[i, 1].set_xlim([0, 256])
        
        axes[i, 1].set_title('RGB Histogram', fontsize=14)
        axes[i, 1].grid(alpha=0.3)
        
        # Grayscale histogram
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        axes[i, 2].plot(hist_gray, color='k')
        axes[i, 2].set_xlim([0, 256])
        axes[i, 2].set_title('Grayscale Histogram', fontsize=14)
        axes[i, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'color_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to create a summary report
def create_summary_report(counts, total_counts, image_props_df):
    # Calculate basic statistics
    width_stats = image_props_df['Width'].describe()
    height_stats = image_props_df['Height'].describe()
    aspect_ratio_stats = image_props_df['Aspect_Ratio'].describe()
    size_stats = image_props_df['Size_KB'].describe()
    
    # Check if all images have the same dimensions
    unique_dimensions = image_props_df[['Width', 'Height']].drop_duplicates()
    num_unique_dimensions = len(unique_dimensions)
    
    # Create a summary report
    report = f"""# Brain Tumor MRI Dataset Analysis Summary

## Dataset Overview
- Total number of images: {total_counts['Training'] + total_counts['Testing']}
  - Training set: {total_counts['Training']} images
  - Testing set: {total_counts['Testing']} images

## Class Distribution
- Training set:
{('  - ' + pd.Series(counts['Training']).to_string(header=False)).replace('\\n', '\\n  - ')}

- Testing set:
{('  - ' + pd.Series(counts['Testing']).to_string(header=False)).replace('\\n', '\\n  - ')}

## Image Properties
- Number of unique image dimensions: {num_unique_dimensions}
- Width statistics (pixels):
  - Min: {width_stats['min']:.2f}
  - Max: {width_stats['max']:.2f}
  - Mean: {width_stats['mean']:.2f}
  - Std: {width_stats['std']:.2f}

- Height statistics (pixels):
  - Min: {height_stats['min']:.2f}
  - Max: {height_stats['max']:.2f}
  - Mean: {height_stats['mean']:.2f}
  - Std: {height_stats['std']:.2f}

- Aspect ratio statistics (width/height):
  - Min: {aspect_ratio_stats['min']:.2f}
  - Max: {aspect_ratio_stats['max']:.2f}
  - Mean: {aspect_ratio_stats['mean']:.2f}
  - Std: {aspect_ratio_stats['std']:.2f}

- File size statistics (KB):
  - Min: {size_stats['min']:.2f}
  - Max: {size_stats['max']:.2f}
  - Mean: {size_stats['mean']:.2f}
  - Std: {size_stats['std']:.2f}

## Preprocessing Recommendations
Based on the analysis, the following preprocessing steps are recommended:
1. Resize all images to a standard size (e.g., 224x224) for model input
2. Normalize pixel values to [0, 1]
3. Apply data augmentation techniques to increase the diversity of the training data:
   - Random rotations
   - Horizontal flips
   - Zoom in/out
   - Width/height shifts

## Visualizations
The following visualizations have been generated and saved in the `analysis_results` directory:
1. Class distribution
2. Sample images from each class
3. Image properties distributions
4. Image properties box plots
5. Pixel intensity distributions
6. Preprocessing steps demonstration
7. Color histograms

## Next Steps
1. Develop a CNN model for brain tumor classification
2. Implement data augmentation to improve model generalization
3. Evaluate model performance on the testing set
4. Create an application that can analyze new MR images and provide diagnostic information
"""
    
    # Save the report
    with open(output_dir / 'analysis_summary.md', 'w') as f:
        f.write(report)

# Main function to run all analyses
def main():
    print("Starting brain tumor MRI dataset analysis...")
    
    # Count images in each class
    counts, total_counts = count_images()
    print(f"Total images: {total_counts['Training'] + total_counts['Testing']}")
    print(f"Training set: {total_counts['Training']} images")
    print(f"Testing set: {total_counts['Testing']} images")
    
    # Plot class distribution
    print("Plotting class distribution...")
    plot_class_distribution(counts)
    
    # Display sample images
    print("Generating sample images...")
    display_sample_images(num_samples=4)
    
    # Analyze image properties
    image_props_df = analyze_image_properties(max_per_class=50)
    
    # Visualize image properties
    print("Visualizing image properties...")
    visualize_image_properties(image_props_df)
    
    # Analyze pixel intensities
    print("Analyzing pixel intensities...")
    intensity_df, std_dev_df = analyze_pixel_intensities(num_samples=25)
    
    # Show preprocessing steps
    print("Demonstrating preprocessing steps...")
    show_preprocessing_steps()
    
    # Analyze color histograms
    analyze_color_histograms()
    
    # Create summary report
    print("Creating summary report...")
    create_summary_report(counts, total_counts, image_props_df)
    
    print("Analysis complete! Results saved to:", output_dir)

if __name__ == "__main__":
    main()
