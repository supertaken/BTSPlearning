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

def process_images_with_array(base_array, input_dir):
    """Process images using the base array"""
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print(f"No image files found in the {input_dir} directory")
        return None

    print(f"Found {len(image_files)} images for processing")

    # Initialize result array
    result_array = np.zeros((len(image_files), base_array.shape[1]), dtype=int)

    # Process each image
    for idx, img_file in enumerate(image_files):
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
                # Invert the array so that black pixels become 1 and white pixels become 0
                img_array = 1 - img_array

                print(f"Processing {img_file}: Image array shape {img_array.shape}")

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
                print(f"Finished processing {img_file}")

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

    return result_array

def save_array_to_excel(array, output_path):
    """Save a 2D array to an Excel file"""
    try:
        # Convert NumPy array to Pandas DataFrame
        df = pd.DataFrame(array)

        # Save as Excel file
        df.to_excel(output_path, index=False, header=False)
        print(f"Result array successfully saved as Excel file: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving Excel file: {str(e)}")
        return False

def visualize_result(array):
    """Visualize the result array"""
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
    plt.savefig("./stdp_mask_object.png", bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # Set file paths
    excel_path = "./stdp_result_array.xlsx"
    input_dir = "./mask_cars/mask_car3"
    output_path = "./stdp_mask_object.xlsx"

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created {input_dir} directory, please place images in this directory")
        return

    # Read the base array
    base_array = read_excel_to_array(excel_path)
    if base_array is None:
        return

    # Process images and get the result array
    result_array = process_images_with_array(base_array, input_dir)
    if result_array is None:
        return

    # Save the result array as an Excel file
    save_success = save_array_to_excel(result_array, output_path)

    # Visualize result - no blurring effect
    visualize_result(result_array)

    if save_success:
        print("Program execution completed, results saved")
    else:
        print("Program execution completed, but result saving failed")

if __name__ == "__main__":
    main()