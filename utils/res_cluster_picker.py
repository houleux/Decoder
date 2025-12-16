import random

def pick_max_avg_residual_cluster(residuals, clusters):
    max_residual = -float('inf')
    best_candidates = []

    for i, cluster in enumerate(clusters):
        cluster_residual = sum(residuals[j] for j in cluster)
        cluster_avg_residual = cluster_residual / len(cluster) if len(cluster) > 0 else 0
        
        if cluster_avg_residual > max_residual:
            max_residual = cluster_avg_residual
            best_candidates = [(i, cluster)]
        elif cluster_avg_residual == max_residual:
            best_candidates.append((i, cluster))

    if not best_candidates:
        return -1, None

    return random.choice(best_candidates)

def pick_max_max_residual_cluster(residuals, clusters):
    max_residual = -float('inf')
    best_candidates = []

    for i, cluster in enumerate(clusters):
        cluster_max_residual = max(residuals[j] for j in cluster) if len(cluster) > 0 else -float('inf')
        
        if cluster_max_residual > max_residual:
            max_residual = cluster_max_residual
            best_candidates = [(i, cluster)]
        elif cluster_max_residual == max_residual:
            best_candidates.append((i, cluster))

    if not best_candidates:
        return -1, None

    return random.choice(best_candidates)