import os
import numpy as np
from PIL import Image
import random
import pandas as pd

def process_images(folder1_path="./folder1", folder2_path="./folder2",
                   folder1_columns=9, folder2_columns=90):
    
    # Create a 900x900 array of zeros
    array = np.zeros((900, 900), dtype=int)
    # Create a counter array to track selection times for each pixel
    selection_counter = np.zeros((900, 900), dtype=int)

    # Define folder list, including paths and corresponding number of columns to select
    folders = [
        {"path": folder1_path, "columns": folder1_columns, "name": "folder1"},
        {"path": folder2_path, "columns": folder2_columns, "name": "folder2"}
    ]

    # Process each folder
    for folder_info in folders:
        folder_path = folder_info["path"]
        columns_to_select = folder_info["columns"]
        folder_name = folder_info["name"]

        if not os.path.exists(folder_path):
            continue

        # Get all image files (support common formats)
        image_files = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

        if not image_files:
            continue

        # Process each image in this folder
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)

            try:
                # Open image and ensure it's binary (1-bit)
                with Image.open(img_path) as img:
                    # Convert to binary mode (1-bit depth)
                    if img.mode != '1':
                        img = img.convert('1')

                    # Ensure image size is 900x900
                    if img.size != (900, 900):
                        img = img.resize((900, 900))

                    # Convert image to NumPy array
                    img_array = np.array(img, dtype=int)

                    # Randomly select a specified number of columns
                    columns_to_process = random.sample(range(900), columns_to_select)

                    # Process the selected columns
                    for row in columns_to_process:
                        for col in range(900):
                            # Check if it's a black pixel (0 represents black in binary images)
                            if img_array[row, col] == 0:
                                # Only process if the array value is still 0
                                if array[row, col] == 0:
                                    # Increment the selection counter for this pixel
                                    selection_counter[row, col] += 1
                                    
                                    # Check if the pixel has been selected 10 times
                                    if selection_counter[row, col] >= 10:
                                        array[row, col] = 1
                                # If array[row, col] is already 1, do nothing (keep it as 1)

            except Exception as e:
                pass

    return array

def save_array_to_excel(array, filename="./stdp_result_array.xlsx"):
    """Save a 2D array to an Excel file"""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Convert NumPy array to Pandas DataFrame
        df = pd.DataFrame(array)

        # Save as Excel file
        df.to_excel(filename, index=False, header=False)
        return True
    except Exception as e:
        return False

def main():
    # Set folder paths and number of columns to select
    FOLDER1_PATH = "./cars1"  # Path to the first folder
    FOLDER2_PATH = "./cars2"  # Path to the second folder

    FOLDER1_COLUMNS = 30  # Number of columns to select from the first folder
    FOLDER2_COLUMNS = 30  # Number of columns to select from the second folder

    # Check if folders exist
    if not os.path.exists(FOLDER1_PATH) and not os.path.exists(FOLDER2_PATH):
        return

    # Process images and get the result array
    result_array = process_images(
        folder1_path=FOLDER1_PATH,
        folder2_path=FOLDER2_PATH,
        folder1_columns=FOLDER1_COLUMNS,
        folder2_columns=FOLDER2_COLUMNS
    )

    # Save array as Excel file
    save_array_to_excel(result_array)

    # Save result array as an image
    result_img = Image.fromarray((result_array * 255).astype(np.uint8))
    result_img.save("./stdp_result_array.png")

if __name__ == "__main__":
    main()