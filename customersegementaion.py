import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time

# Load the first dataset
data1 = pd.read_csv('F:\\projects\\Comparative_Study_of_Clustering_maths4\\datasets\\customer_det.csv', encoding='latin1')

# Assuming data1 has columns 'CustomerID', 'InvoiceDate', 'InvoiceNo', 'UnitPrice'
data1['InvoiceDate'] = pd.to_datetime(data1['InvoiceDate'], format='%d/%m/%Y %H:%M')

# Calculate Recency, Frequency, and Monetary values for data1
snapshot_date1 = data1['InvoiceDate'].max() + pd.DateOffset(days=1)
rfm_data1 = data1.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date1 - x.max()).days,
    'InvoiceNo': 'count',
    'UnitPrice': 'sum'
}).reset_index()
rfm_data1.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
# Print RFM analysis
print("\nRFM Analysis:")
print(rfm_data1.head())

# Load the second dataset
data2 = pd.read_csv('F:\\projects\\Comparative_Study_of_Clustering_maths4\\datasets\\shopping_trends.csv', encoding='latin1')

# Assuming data2 has relevant columns like 'Customer ID', 'Purchase Amount (USD)', 'Previous Purchases'
data2.columns = ['CustomerID', 'Age', 'Gender', 'ItemPurchased', 'Category', 'PurchaseAmount', 
                 'Location', 'Size', 'Color', 'Season', 'ReviewRating', 'SubscriptionStatus', 
                 'PaymentMethod', 'ShippingType', 'DiscountApplied', 'PromoCodeUsed', 
                 'PreviousPurchases', 'PreferredPaymentMethod', 'FrequencyOfPurchases']

# Calculate RFM values for data2 (using PurchaseAmount for Monetary)
rfm_data2 = data2.groupby('CustomerID').agg({
    'PreviousPurchases': 'sum',  # Using PreviousPurchases as Frequency
    'PurchaseAmount': 'sum'      # Using PurchaseAmount as Monetary
}).reset_index()
rfm_data2.columns = ['CustomerID','Frequency', 'Monetary']
# Assuming Recency can be simulated or taken as a known value (using Purchase Date would be ideal if available)
# For simplicity, we will use a fixed Recency value here
rfm_data2['Recency'] = np.random.randint(1, 365, size=rfm_data2.shape[0])

# Print RFM analysis
print("\nRFM Analysis:")
print(rfm_data2.head())


# Standardize the data for both datasets
scaler = StandardScaler()
scaled_data1 = scaler.fit_transform(rfm_data1[['Recency', 'Frequency', 'Monetary']])
scaled_data2 = scaler.fit_transform(rfm_data2[['Recency', 'Frequency', 'Monetary']])

# Function to perform clustering and evaluation
def perform_clustering(data, dataset_name):
    clustering_results = []

    # K-means
    start_time = time.time()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(data)
    kmeans_time = time.time() - start_time
    kmeans_silhouette = silhouette_score(data, kmeans.labels_)
    clustering_results.append(('K-means', kmeans_silhouette, kmeans_time))

    # Hierarchical
    start_time = time.time()
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical.fit(data)
    hierarchical_time = time.time() - start_time
    hierarchical_silhouette = silhouette_score(data, hierarchical.labels_)
    clustering_results.append(('Hierarchical', hierarchical_silhouette, hierarchical_time))

    # DBSCAN
    start_time = time.time()
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(data)
    dbscan_time = time.time() - start_time
    if len(set(dbscan.labels_)) > 1:
        dbscan_silhouette = silhouette_score(data[dbscan.labels_ != -1], dbscan.labels_[dbscan.labels_ != -1])
    else:
        dbscan_silhouette = np.nan  # Silhouette score is not defined for a single cluster
    clustering_results.append(('DBSCAN', dbscan_silhouette, dbscan_time))

    # Spectral
    start_time = time.time()
    spectral = SpectralClustering(n_clusters=3, random_state=42)
    spectral.fit(data)
    spectral_time = time.time() - start_time
    spectral_silhouette = silhouette_score(data, spectral.labels_)
    clustering_results.append(('Spectral', spectral_silhouette, spectral_time))

    # Print results in table format
    print(f"\n{dataset_name} - Evaluation Metrics:")
    print("{:<15} {:<20} {:<20}".format('Method', 'Silhouette Score', 'Time Taken (seconds)'))
    for method, silhouette, time_taken in clustering_results:
        print("{:<15} {:<20} {:<20}".format(method, f"{silhouette:.4f}" if not np.isnan(silhouette) else "N/A", f"{time_taken:.4f}"))

    


    # Plot the results
    plt.figure(figsize=(24, 6))

    # K-means
    plt.subplot(1, 4, 1)
    for i in range(3):
        plt.scatter(data[:, 0][kmeans.labels_ == i], data[:, 2][kmeans.labels_ == i], label=f'Cluster {i+1}')
    plt.title(f'K-means Clustering (Time: {kmeans_time:.2f}s)')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.legend()

    # Hierarchical
    plt.subplot(1, 4, 2)
    for i in range(3):
        plt.scatter(data[:, 0][hierarchical.labels_ == i], data[:, 2][hierarchical.labels_ == i], label=f'Cluster {i+1}')
    plt.title(f'Hierarchical Clustering (Time: {hierarchical_time:.2f}s)')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.legend()

    # DBSCAN
    plt.subplot(1, 4, 3)
    unique_labels_dbscan = set(dbscan.labels_)
    for i, cluster_label in enumerate(unique_labels_dbscan):
        if cluster_label == -1:
            plt.scatter(data[:, 0][dbscan.labels_ == cluster_label], data[:, 2][dbscan.labels_ == cluster_label], label='Noise')
        else:
            plt.scatter(data[:, 0][dbscan.labels_ == cluster_label], data[:, 2][dbscan.labels_ == cluster_label], label=f'Cluster {i+1}')
    plt.title(f'DBSCAN Clustering (Time: {dbscan_time:.2f}s)')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.legend()
    

    # Spectral
    plt.subplot(1, 4, 4)
    unique_labels_spectral = set(spectral.labels_)
    for i, cluster_label in enumerate(unique_labels_spectral):
        plt.scatter(data[:, 0][spectral.labels_ == cluster_label], data[:, 2][spectral.labels_ == cluster_label], label=f'Cluster {i+1}')
    plt.title(f'Spectral Clustering (Time: {spectral_time:.2f}s)')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.legend()
    

    plt.suptitle(f'Clustering Results ({dataset_name})', fontsize=16)  # Title for the entire figure
    plt.tight_layout()
    plt.show()


    return clustering_results

# Perform clustering on both datasets
results1 = perform_clustering(scaled_data1, "Dataset 1")
results2 = perform_clustering(scaled_data2, "Dataset 2")


def find_best_classifier(results, metric_index, metric_name):
    best_method = None
    best_value = None
    for method, value, time_taken in results:
        if best_value is None or (metric_index == 1 and (best_value is None or value > best_value)) or (metric_index == 2 and (best_value is None or time_taken < best_value)):
            best_method = method
            best_value = value if metric_index == 1 else time_taken
    return best_method, best_value


# Find the best classifier for each dataset based on silhouette score
best_silhouette1, silhouette_value1 = find_best_classifier(results1, 1, 'Silhouette Score')
best_silhouette2, silhouette_value2 = find_best_classifier(results2, 1, 'Silhouette Score')

# Find the best classifier for each dataset based on time taken
best_time1, time_value1 = find_best_classifier(results1, 2, 'Time Taken')
best_time2, time_value2 = find_best_classifier(results2, 2, 'Time Taken')

# Print best classifiers
print(f"\nBest classifiers based on silhouette score:")
print(f"Dataset 1: {best_silhouette1} with Silhouette Score {silhouette_value1:.4f}")
print(f"Dataset 2: {best_silhouette2} with Silhouette Score {silhouette_value2:.4f}")

print(f"\nBest classifiers based on time complexity:")
print(f"Dataset 1: {best_time1} with Time Taken {time_value1:.4f} seconds")
print(f"Dataset 2: {best_time2} with Time Taken {time_value2:.4f} seconds")

# Summarize overall best classifier
overall_best_silhouette = best_silhouette1 if silhouette_value1 > silhouette_value2 else best_silhouette2
overall_best_time = best_time1 if time_value1 < time_value2 else best_time2

print(f"\nOverall best classifier based on silhouette score: {overall_best_silhouette}")
print(f"Overall best classifier based on time complexity: {overall_best_time}")

