## K-means Clustering
- Divide dataset to $K$ clusters, each have a cluster center $\mathbf{c}$
- Assign each datapoint to a cluster by Euclidean distance $||\mathbf{x}^{(i)} - \mathbf{c}||_2$
- Pick $K$ (random) initial centers the followings and the center will finally converge
  1. Assign all data points to the closest center based on Euclidean distance
  2. Update the center of all clusters with cluster's mean vector
- K-means attempts to minimize the within-cluster variation over all clusters
- Training process is exactly **coordinate descent**
  - Assignment step **fixes all centers** and update all datapoints' cluster
  - Update step **fixes all data points** and update all centers
- Result heavily depends on inital cluster centers

## Solution for cluster center initialization
- Random restart of the algo
  - Take the best result
- Selecting random data points as initial centers
- Use $K\log(K)$ clusters, then merge to $K$ clusters

## Choosing K
- Elbow method
  - Compute the **sum of square error** (SSE) for each cluster and sum them up
  - Compare the result of summed SSE for each K
  - When summed SSE starts to drop slowly, choose the K that starts flatting summed SSE

## Problem for K-means clustering
- Assuming clusters are within circle shape
  - Due to using the Euclidean distance
  - Perform bad for elliptical clusters