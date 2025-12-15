from ldpc.bp_decoder import BpDecoder

def pick_max_avg_residual_cluster(residuals, clusters):
    max_residual = -float('inf')
    selected_cluster = None
    selected_index = -1

    for i, cluster in enumerate(clusters):
        cluster_residual = sum(residuals[j] for j in cluster)
        cluster_avg_residual = cluster_residual / len(cluster) if len(cluster) > 0 else 0
        if cluster_avg_residual > max_residual:
            max_residual = cluster_avg_residual
            selected_cluster = cluster
            selected_index = i

    return selected_index, selected_cluster

def pick_max_max_residual_cluster(residuals, clusters):
    max_residual = -float('inf')
    selected_cluster = None

    for i, cluster in enumerate(clusters):
        cluster_max_residual = max(residuals[j] for j in cluster) if len(cluster) > 0 else -float('inf')
        if cluster_max_residual > max_residual:
            max_residual = cluster_max_residual
            selected_cluster = cluster
            selected_index = i

    return selected_index, selected_cluster