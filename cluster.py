import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from adjustText import adjust_text
import datetime

# 1. Read Excel files from folder and process data
def load_data_from_folder(folder_path):
    file_names = []  # Store file names (cluster names)
    all_vectors = []  # Store all vectors
    labels = []  # Store file name labels for each vector
    
    # Traverse all Excel files in the folder
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx') or file.endswith('.xls'):
            file_path = os.path.join(folder_path, file)
            try:
                # Read Excel file
                df = pd.read_excel(file_path, header=None)
                
                # Ensure data is numeric
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Process each row of data
                for i in range(min(26, len(df))):  # Take at most 4 rows
                    row = df.iloc[i].values
                    
                    # Ensure data has 900 dimensions
                    if len(row) > 900:
                        row = row[:900]  # Truncate parts exceeding 900 dimensions
                    elif len(row) < 900:
                        # Pad with zeros if less than 900 dimensions
                        row = np.pad(row, (0, 900 - len(row)), 'constant')
                    
                    # Convert to binary vector (0 or 1)
                    binary_vector = np.where(row > 0, 1, 0)
                    
                    all_vectors.append(binary_vector)
                    labels.append(os.path.splitext(file)[0])  # File name without extension
                
                file_names.append(os.path.splitext(file)[0])
                
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
    
    return np.array(all_vectors), np.array(labels), file_names

# 2. Main program
def main():
    # Set folder path (modify to your Excel file path)
    folder_path = "./cluster_btsp"  # Replace with your folder path
    
    # Load data
    data, labels, file_names = load_data_from_folder(folder_path)
    
    if len(data) == 0:
        print("No Excel files found or data is empty!")
        return
    
    print(f"Successfully loaded {len(data)} vectors from {len(file_names)} files")
    print(f"Vector dimension: {data.shape[1]}")
    
    # 3. Data preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 4. Dimensionality reduction
    def reduce_dimension(method='tsne', data=scaled_data):
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(data)
        elif method == 'tsne':
            # Ensure perplexity does not exceed data size
            perplexity = min(30, len(data) // 3)
            if perplexity < 1:
                perplexity = 1
                
            reducer = TSNE(n_components=2, 
                           perplexity=perplexity, 
                           learning_rate=200, 
                           random_state=42,
                           init='pca')
            reduced_data = reducer.fit_transform(data)
        elif method == 'mds':
            reducer = MDS(n_components=2, 
                          max_iter=300, 
                          eps=1e-9, 
                          random_state=42,
                          dissimilarity="euclidean")
            reduced_data = reducer.fit_transform(data)
        else:  # Default to t-SNE
            # Ensure perplexity does not exceed data size
            perplexity = min(30, len(data) // 3)
            if perplexity < 1:
                perplexity = 1
                
            reducer = TSNE(n_components=2, 
                           perplexity=perplexity, 
                           learning_rate=200, 
                           random_state=42,
                           init='pca')
            reduced_data = reducer.fit_transform(data)
        
        # Normalize to [0,1] range
        scaler_2d = MinMaxScaler()
        return scaler_2d.fit_transform(reduced_data)
    
    # Use t-SNE for dimensionality reduction
    reduced_data = reduce_dimension('tsne', scaled_data)
    
    # 5. Clustering analysis - Try multiple methods and select the best
    num_clusters = min(10, len(file_names))  # Number of clusters should not exceed number of files
    
    # Method 1: KMeans
    kmeans = KMeans(n_clusters=num_clusters, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300, 
                    random_state=42)
    kmeans_clusters = kmeans.fit_predict(scaled_data)
    kmeans_score = silhouette_score(scaled_data, kmeans_clusters)
    
    # Method 2: Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters, 
                                           linkage='ward')
    hierarchical_clusters = hierarchical.fit_predict(scaled_data)
    hierarchical_score = silhouette_score(scaled_data, hierarchical_clusters)
    
    # Method 3: DBSCAN (density-based clustering)
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    dbscan_clusters = dbscan.fit_predict(scaled_data)
    
    # If DBSCAN finds a reasonable number of clusters
    if len(np.unique(dbscan_clusters)) > 1:
        dbscan_score = silhouette_score(scaled_data, dbscan_clusters)
    else:
        dbscan_score = -1  # Invalid score
        # Ensure dbscan_clusters is valid
        dbscan_clusters = np.zeros(len(scaled_data), dtype=int)  # Set to default cluster
    
    # Select the best clustering method
    clustering_methods = {
        'KMeans': (kmeans_clusters, kmeans_score),
        'Hierarchical': (hierarchical_clusters, hierarchical_score),
        'DBSCAN': (dbscan_clusters, dbscan_score) if dbscan_score > -1 else None
    }
    
    # Find the best clustering method
    best_method = 'KMeans'
    best_score = kmeans_score
    best_clusters = kmeans_clusters
    
    # Fix: Properly handle None values
    for method, result in clustering_methods.items():
        if result is None:
            continue  # Skip None values
            
        clusters, score = result
        if score > best_score:
            best_method = method
            best_score = score
            best_clusters = clusters
    
    print(f"\nSelected clustering method: {best_method}, Silhouette score: {best_score:.4f}")
    
    # 6. Visualize results
    plt.figure(figsize=(16, 12))
    
    # Create color mapping
    unique_clusters = np.unique(best_clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    
    # Draw scatter plot
    for cluster in unique_clusters:
        indices = np.where(best_clusters == cluster)[0]
        plt.scatter(reduced_data[indices, 0], 
                    reduced_data[indices, 1], 
                    c=[cluster_colors[cluster]] * len(indices),
                    s=60, 
                    alpha=0.8,
                    edgecolors='w',
                    linewidth=0.8,
                    label=f'Cluster {cluster}')
    
    plt.title(f'Clustering of {data.shape[1]}-D Binary Vectors ({best_method})', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(alpha=0.2)
    
    # 7. Add file name labels (avoid overlap)
    texts = []
    unique_labels = {}
    
    # Find a representative point for each file (closest point)
    for file in file_names:
        # Find all points for this file
        indices = np.where(labels == file)[0]
        if len(indices) > 0:
            # Calculate the center of these points
            center = np.mean(reduced_data[indices], axis=0)
            # Record label position
            unique_labels[file] = center
    
    # Add labels to the chart
    for file, pos in unique_labels.items():
        texts.append(plt.text(pos[0], pos[1], file, 
                             fontsize=10, 
                             ha='center', 
                             va='center',
                             bbox=dict(boxstyle="round,pad=0.3", 
                                       fc="white", 
                                       ec="gray", 
                                       alpha=0.8)))
    
    # Adjust label positions to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    # Add legend
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # 8. Save high-quality image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f'clustering_result_{timestamp}.png'
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    print(f"Clustering visualization saved as: {image_filename}")
    plt.show()
    
    # 9. Create result DataFrame and save to Excel
    # Prepare result data
    result_df = pd.DataFrame({
        'File Name': labels,
        'Cluster Label': best_clusters,
        'Reduced_Dimension_Coordinate_X': reduced_data[:, 0],
        'Reduced_Dimension_Coordinate_Y': reduced_data[:, 1]
    })
    
    # Add original vector data (optional)
    for i in range(data.shape[1]):
        if i < 10:  # Only save first 10 dimensions to reduce file size
            result_df[f'Dimension_{i+1}'] = data[:, i]
    
    # 10. Output clustering result statistics
    cluster_stats = []
    print("\nClustering result statistics:")
    for cluster_id in unique_clusters:
        cluster_indices = np.where(best_clusters == cluster_id)[0]
        cluster_files = labels[cluster_indices]
        unique_files, counts = np.unique(cluster_files, return_counts=True)
        
        # Collect statistical information
        cluster_stats.append({
            'Cluster ID': cluster_id,
            'Number of Vectors': len(cluster_indices),
            'Contained Files': ', '.join(unique_files),
            'Number of Files': len(unique_files)
        })
        
        print(f"\nCluster {cluster_id} (contains {len(cluster_indices)} vectors):")
        for file, count in zip(unique_files, counts):
            print(f"  - {file}: {count} vectors")
    
    # Create cluster statistics DataFrame
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    # 11. Output clustering quality evaluation
    quality_info = []
    if len(unique_clusters) > 1:
        silhouette = silhouette_score(scaled_data, best_clusters)
        calinski = calinski_harabasz_score(scaled_data, best_clusters)
        
        quality_info.append({
            'Evaluation Metric': 'Silhouette Score',
            'Value': silhouette,
            'Description': 'Range[-1,1], higher is better'
        })
        
        quality_info.append({
            'Evaluation Metric': 'Calinski-Harabasz Index',
            'Value': calinski,
            'Description': 'Higher is better'
        })
        
        print(f"\nClustering quality evaluation:")
        print(f"Silhouette Score: {silhouette:.4f} (Range[-1,1], higher is better)")
        print(f"Calinski-Harabasz Index: {calinski:.2f} (Higher is better)")
    
    # Create quality evaluation DataFrame
    quality_df = pd.DataFrame(quality_info)
    
    # 12. Save all results to Excel file
    excel_filename = f'clustering_results_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_filename) as writer:
        # Save clustering results
        result_df.to_excel(writer, sheet_name='Clustering Results', index=False)
        
        # Save cluster statistics
        cluster_stats_df.to_excel(writer, sheet_name='Cluster Statistics', index=False)
        
        # Save quality evaluation
        if not quality_df.empty:
            quality_df.to_excel(writer, sheet_name='Quality Evaluation', index=False)
        
        # Save clustering method information
        method_info = pd.DataFrame({
            'Parameter': ['Clustering Method', 'Number of Clusters', 'Dimensionality Reduction Method', 'Silhouette Score'],
            'Value': [best_method, num_clusters, 't-SNE', best_score]
        })
        method_info.to_excel(writer, sheet_name='Method Information', index=False)
    
    print(f"\nAll clustering results saved to Excel file: {excel_filename}")

if __name__ == "__main__":
    main()