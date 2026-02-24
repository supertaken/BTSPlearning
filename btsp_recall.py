import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def read_excel_to_array(file_path):
    """Read a 2D array from an Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(file_path, header=None)
        # Convert to NumPy array
        array = df.to_numpy()
        print(f"Successfully read 2D array from {file_path}")
        print(f"Array shape: {array.shape}")
        return array
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

def process_images_with_array(base_array, input_dir1, input_dir2):
    all_image_files = []

    # Process the first folder
    if os.path.exists(input_dir1):
        image_files1 = [f for f in os.listdir(input_dir1) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if image_files1:
            print(f"Found {len(image_files1)} images in {input_dir1}")
            all_image_files.extend([(input_dir1, f) for f in image_files1])
    else:
        print(f"Warning: Directory {input_dir1} does not exist")

    # Process the second folder
    if os.path.exists(input_dir2):
        image_files2 = [f for f in os.listdir(input_dir2) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if image_files2:
            print(f"Found {len(image_files2)} images in {input_dir2}")
            all_image_files.extend([(input_dir2, f) for f in image_files2])
    else:
        print(f"Warning: Directory {input_dir2} does not exist")

    if not all_image_files:
        print("No image files found in either input directory")
        return None

    print(f"Found a total of {len(all_image_files)} images to process")

    # Initialize result array
    result_array = np.zeros((len(all_image_files), base_array.shape[1]), dtype=int)

    # Process each image
    for idx, (input_dir, img_file) in enumerate(all_image_files):
        img_path = os.path.join(input_dir, img_file)

        try:
            # Open image and ensure it's binary (1-bit)
            with Image.open(img_path) as img:
                # Convert to binary mode (1-bit depth)
                if img.mode != '1':
                    img = img.convert('1')

                # Ensure image dimensions match the base array
                if img.size != (base_array.shape[1], base_array.shape[0]):
                    img = img.resize((base_array.shape[1], base_array.shape[0]))

                # Convert image to NumPy array
                img_array = np.array(img, dtype=int)

                # Treat black pixels as 1 (in binary images, 0 usually represents black)
                img_array = 1 - img_array

                print(f"Processing {os.path.basename(input_dir)}/{img_file}")

                # Create the result row for this image
                result_row = np.zeros(base_array.shape[1], dtype=int)

                # Iterate through each column
                for col in range(base_array.shape[1]):
                    # Check if there exists a position in this column where both are 1
                    for row in range(base_array.shape[0]):
                        if base_array[row, col] == 1 and img_array[row, col] == 1:
                            result_row[col] = 1
                            break  # Stop searching this column as soon as one match is found

                # Add the result row to the result array
                result_array[idx] = result_row

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

    return result_array

def save_array_to_excel(array, output_path):
    """Save a 2D array to an Excel file"""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert NumPy array to Pandas DataFrame
        df = pd.DataFrame(array)

        # Save as Excel file
        df.to_excel(output_path, index=False, header=False)
        print(f"Result array successfully saved as Excel file: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving Excel file: {str(e)}")
        return False

def visualize_result(array, output_path):
    """Visualize the result array"""
    if array is None or array.size == 0:
        print("Cannot visualize an empty array")
        return

    # Calculate figure size (10 pixels per row, 1 pixel per column)
    height = array.shape[0] * 10
    width = array.shape[1] * 1

    # Create figure
    plt.figure(figsize=(width/100, height/100), dpi=100)

    # Custom colormap: white and blue
    from matplotlib.colors import ListedColormap
    colors = ['white', 'blue']  # 0=white, 1=blue
    cmap = ListedColormap(colors)

    # Display array - use nearest-neighbor interpolation to avoid blurring
    plt.imshow(array, cmap=cmap, interpolation='nearest', aspect='auto')
    plt.title('Result Array Visualization')

    # Hide axes
    plt.axis('off')

    # Save visualization result
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Visualization result saved: {output_path}")

def main():
    # Set file paths
    excel_path = "./btsp_result_array.xlsx"
    input_dir1 = "./cars1"  # First input folder
    input_dir2 = "./cars2"  # Second input folder
    output_excel = "./btsp_similar.xlsx"  # Output Excel file
    output_png = "./btsp_similar.png"  # Output PNG file

    # Read the base array
    base_array = read_excel_to_array(excel_path)
    if base_array is None:
        return

    print("\n" + "="*50)
    print("Starting to process images from two folders...")
    print("="*50)

    # Process images from two folders
    result_array = process_images_with_array(base_array, input_dir1, input_dir2)

    if result_array is None or result_array.size == 0:
        print("No results generated, program exiting")
        return

    # Save the result array as an Excel file
    save_success = save_array_to_excel(result_array, output_excel)

    # Visualize result
    visualize_result(result_array, output_png)

    # Output statistics
    print("\n" + "="*50)
    print("Processing result statistics:")
    print(f"Result array shape: {result_array.shape}")
    print(f"Number of 1s in the array: {np.sum(result_array)}")
    print(f"Number of 0s in the array: {np.sum(result_array == 0)}")
    print(f"Proportion of 1s: {np.sum(result_array) / result_array.size * 100:.2f}%")

    if save_success:
        print("\nProgram execution completed, all results saved")
    else:
        print("\nProgram execution completed, but result saving failed")

if __name__ == "__main__":
    main()