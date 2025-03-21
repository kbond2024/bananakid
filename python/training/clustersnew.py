import numpy as np
from cuml import KMeans
import cudf
import argparse
import gc
import time
import os
from sklearn.metrics import silhouette_score


def cluster(data, n_clusters, seed=42):
    print(f"Using full KMeans with {n_clusters} clusters on {data.shape[0]} data points...")
    df = cudf.DataFrame(data)

    # Use multiple initializations to find better clusters
    best_km = None
    best_score = float('-inf')

    for init_method in ["k-means++", "random"]:
        print(f"Trying initialization method: {init_method}")
        km = KMeans(n_clusters=n_clusters,
                    init=init_method,
                    n_init=10,  # Try multiple initializations
                    max_iter=300,
                    random_state=seed)
        km.fit(df)

        # Calculate silhouette score on a subset to save memory
        if df.shape[0] > 10000:
            sample_indices = np.random.choice(df.shape[0], 10000, replace=False)
            sample_df = df.iloc[sample_indices]
            sample_labels = km.labels_[sample_indices]
            current_score = -km.inertia_  # Negative because lower inertia is better


        else:
            current_score = silhouette_score(df, km.labels_)

        print(f"Initialization {init_method} score: {current_score:.4f}")

        if current_score > best_score:
            best_score = current_score
            best_km = km

    print(f"Best silhouette score: {best_score:.4f}")
    print(f"Inertia: {best_km.inertia_}")
    return best_km.labels_.to_pandas().values


def cluster_batched(i, n_clusters, seed=42):
    os.makedirs("cluster_checkpoints", exist_ok=True)
    checkpoint_file = f"cluster_checkpoints/clusters_r{i}_c{n_clusters}_partial.npy"

    # Check if we can resume from a checkpoint
    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        return np.load(checkpoint_file)

    # Process first batch to get sample size
    first_batch = np.load(f"features_{i}_b0.npy")
    print(f"First batch shape: {first_batch.shape}")

    # Calculate total data points
    print("Calculating total data points...")
    total_data_points = 0
    for b in range(10):
        batch = np.load(f"features_{i}_b{b}.npy")
        total_data_points += batch.shape[0]
        del batch
        gc.collect()

    # Initialize array for storing all labels
    print(f"Total data points: {total_data_points}")
    all_labels = np.zeros(total_data_points, dtype=np.int32)

    # Create a stratified sample for better representation
    print("Creating filtered dataset for KMeans fitting...")
    sample_size = min(800000, total_data_points // 10)  # Use larger sample for better clustering
    selected_data = []
    points_collected = 0

    # Use stratified sampling from all batches
    for b in range(10):
        if points_collected >= sample_size:
            break

        batch = np.load(f"features_{i}_b{b}.npy")
        # Use systematic sampling with different offsets for each batch
        stride = max(1, batch.shape[0] // (sample_size // 10))
        offset = np.random.randint(0, stride)
        filtered_batch = batch[offset::stride]

        to_take = min(sample_size - points_collected, filtered_batch.shape[0])
        selected_data.append(filtered_batch[:to_take])
        points_collected += to_take

        del batch, filtered_batch
        gc.collect()

    # Combine selected data and fit KMeans
    filtered_data = np.concatenate(selected_data)
    print(f"Fitting KMeans on {filtered_data.shape[0]} points...")

    # Normalize features for better clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    filtered_data_scaled = scaler.fit_transform(filtered_data)

    df = cudf.DataFrame(filtered_data_scaled)

    # Try multiple initializations and algorithms
    km = KMeans(n_clusters=n_clusters,
                init="k-means++",
                n_init=10,  # More initializations for better results
                max_iter=300,
                random_state=seed)

    print("Starting KMeans fitting...")
    start_time = time.time()
    km.fit(df)
    end_time = time.time()
    print(f"KMeans fitting completed in {end_time - start_time:.2f} seconds")

    del filtered_data, df, selected_data
    gc.collect()

    # Process each batch individually for prediction
    offset = 0
    for b in range(10):
        print(f"Processing batch {b} for prediction...")
        batch = np.load(f"features_{i}_b{b}.npy")
        batch_size = batch.shape[0]

        # Scale the batch using the same scaler
        batch_scaled = scaler.transform(batch)

        # Process in chunks of 100,000 if batch is large
        chunk_size = 100000
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk = batch_scaled[chunk_start:chunk_end]

            batch_df = cudf.DataFrame(chunk)
            batch_labels = km.predict(batch_df).to_pandas().values

            # Store in the pre-allocated array
            all_labels[offset + chunk_start:offset + chunk_end] = batch_labels

            del chunk, batch_df, batch_labels
            gc.collect()

        offset += batch_size
        del batch, batch_scaled
        gc.collect()

        # Save checkpoint after each batch for resilience
        np.save(checkpoint_file, all_labels)

    # Calculate score on a sample for evaluation
    sample_batch = np.load(f"features_{i}_b0.npy")
    sample_scaled = scaler.transform(sample_batch[:10000])
    sample_score = km.score(cudf.DataFrame(sample_scaled))
    print(f"Sample score (first 10k points of batch 0): {sample_score}")

    # Validate clusters have good distribution
    unique_clusters, counts = np.unique(all_labels, return_counts=True)
    print(f"Number of unique clusters: {len(unique_clusters)}")
    print(f"Min cluster size: {counts.min()}, Max cluster size: {counts.max()}")
    print(f"Std dev of cluster sizes: {counts.std()}")

    return all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=int)
    parser.add_argument("--clusters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"{args.round=}")
    print(f"{args.clusters=}")
    print(f"{args.seed=}")

    start_time = time.time()

    try:
        if args.round == 3:
            labels = cluster_batched(args.round, args.clusters, args.seed)
        else:
            data = np.load(f"features_{args.round}.npy")
            labels = cluster(data, args.clusters, args.seed)

        print(f"Creating output file...")
        fn = f"clusters_r{args.round}_c{args.clusters}.npy"
        np.save(fn, labels)

        # Also create a backup
        backup_fn = f"clusters_r{args.round}_c{args.clusters}_backup.npy"
        np.save(backup_fn, labels)

        end_time = time.time()
        print(f"Clusters written to {fn}")
        print(f"Total processing time: {(end_time - start_time) / 60:.2f} minutes")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()