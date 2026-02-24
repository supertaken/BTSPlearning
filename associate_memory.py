import pandas as pd
import numpy as np

# Read two Excel files
df1 = pd.read_excel('./stdp_mask_object.xlsx', header=None)  # Assuming no header
df2 = pd.read_excel('./stdp_similar.xlsx', header=None)

# Convert to NumPy arrays
array1 = df1.values
array2 = df2.values

# Verify that the number of columns match
if array1.shape[1] != array2.shape[1]:
    raise ValueError("The two arrays must have the same number of columns for calculation")

# Method 1: Use loop comparison (intuitive but slower)
def compare_arrays_loop(arr1, arr2):
    """Compare two arrays using loops, increment count when numbers in the same column are identical"""
    m, n = arr1.shape
    k, _ = arr2.shape
    
    result = np.zeros((m, k), dtype=int)
    
    for i in range(m):
        for j in range(k):
            count = 0
            for col in range(n):
                if arr1[i, col] == arr2[j, col]:
                    count += 1
            result[i, j] = count
    
    return result

# Method 2: Use vectorized operations (faster)
def compare_arrays_vectorized(arr1, arr2):
    """Compare two arrays using vectorized operations, increment count when numbers in the same column are identical"""
    # Expand dimensions for broadcasting comparison
    # arr1: (m, n) -> (m, 1, n)
    # arr2: (k, n) -> (1, k, n)
    arr1_expanded = arr1[:, np.newaxis, :]
    arr2_expanded = arr2[np.newaxis, :, :]
    
    # Check if each position is equal, then sum along the column direction
    equality_matrix = (arr1_expanded == arr2_expanded)
    result = np.sum(equality_matrix, axis=2)
    
    return result

# Use the vectorized method for comparison
dot_products = compare_arrays_vectorized(array1, array2)

# Create a result DataFrame with annotations
result_df = pd.DataFrame(dot_products)

# Add row annotations (row indices of array1)
result_df.index = [f"Array1-Row{i}" for i in range(array1.shape[0])]

# Add column annotations (row indices of array2)
result_df.columns = [f"Array2-Row{j}" for j in range(array2.shape[0])]

# Save as Excel file
with pd.ExcelWriter('./associate_mask_or.xlsx') as writer:
    result_df.to_excel(writer, sheet_name='Comparison Results', startrow=0)

# Calculate statistical information
max_value = np.max(dot_products)
min_value = np.min(dot_products)
avg_value = np.mean(dot_products)
std_value = np.std(dot_products)
total_columns = array1.shape[1]

print("Array comparison calculation completed")
print(f"Array1 shape: {array1.shape}")
print(f"Array2 shape: {array2.shape}")
print(f"Comparison result shape: {dot_products.shape}")
print(f"\nStatistical Information:")
print(f"  Number of columns: {total_columns}")
print(f"  Maximum identical columns: {max_value} ({max_value/total_columns*100:.1f}%)")
print(f"  Minimum identical columns: {min_value} ({min_value/total_columns*100:.1f}%)")
print(f"  Average identical columns: {avg_value:.1f} ({avg_value/total_columns*100:.1f}%)")
print(f"  Standard deviation: {std_value:.1f}")
print(f"\nComparison results saved to './associate_mask_or.xlsx'")

# Optional: Find the pair with the most identical columns
max_indices = np.unravel_index(np.argmax(dot_products), dot_products.shape)
max_value_pair = dot_products[max_indices]
print(f"\nBest match:")
print(f"  Array1-Row{max_indices[0]} and Array2-Row{max_indices[1]}")
print(f"  Number of identical columns: {max_value_pair} ({max_value_pair/total_columns*100:.1f}%)")