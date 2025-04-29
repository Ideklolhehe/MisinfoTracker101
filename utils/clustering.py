"""
Pluggable clustering architecture for the CIVILIAN system.
Supports multiple clustering algorithms through a registry pattern.
"""

import logging
import time
from typing import Dict, List, Callable, Any, Tuple, Optional, Union
import inspect
import numpy as np

from utils.metrics import time_operation, increment_counter, set_gauge
from utils.metrics import clustering_counter, cluster_count_gauge, silhouette_gauge
from utils.environment import CLUSTERING_EPSILON, CLUSTERING_MIN_SAMPLES

# Configure module logger
logger = logging.getLogger(__name__)

# Registry for clustering algorithms
CLUSTERING_REGISTRY = {}


def register_clustering_algorithm(name: str):
    """
    Decorator to register a clustering algorithm.
    
    Args:
        name: Name of the algorithm
        
    Returns:
        Decorator function
    """
    def decorator(func):
        if name in CLUSTERING_REGISTRY:
            logger.warning(f"Overwriting existing clustering algorithm '{name}'")
            
        CLUSTERING_REGISTRY[name] = func
        logger.info(f"Registered clustering algorithm '{name}'")
        return func
        
    return decorator


def get_clustering_algorithm(name: str) -> Optional[Callable]:
    """
    Get a clustering algorithm by name.
    
    Args:
        name: Name of the algorithm
        
    Returns:
        Clustering algorithm function or None if not found
    """
    if name not in CLUSTERING_REGISTRY:
        logger.warning(f"Clustering algorithm '{name}' not found")
        return None
        
    return CLUSTERING_REGISTRY[name]


def list_clustering_algorithms() -> List[str]:
    """
    List all registered clustering algorithms.
    
    Returns:
        List of algorithm names
    """
    return list(CLUSTERING_REGISTRY.keys())


def get_algorithm_parameters(name: str) -> Dict[str, Any]:
    """
    Get parameters for a clustering algorithm.
    
    Args:
        name: Name of the algorithm
        
    Returns:
        Dictionary of parameter names and default values
    """
    algorithm = get_clustering_algorithm(name)
    if not algorithm:
        return {}
        
    sig = inspect.signature(algorithm)
    params = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'X' or param_name == 'embeddings':
            # Skip the input data parameter
            continue
            
        if param.default is not inspect.Parameter.empty:
            params[param_name] = param.default
        else:
            params[param_name] = None
            
    return params


def perform_clustering(
    embeddings: np.ndarray,
    algorithm: str,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Perform clustering using the specified algorithm.
    
    Args:
        embeddings: Embeddings to cluster (n_samples x n_features)
        algorithm: Name of the clustering algorithm
        **kwargs: Additional parameters for the algorithm
        
    Returns:
        Tuple of (cluster labels, silhouette score if available)
    """
    if len(embeddings) == 0:
        logger.warning("Cannot perform clustering on empty embeddings")
        return np.array([]), 0.0
        
    # Get algorithm
    clustering_func = get_clustering_algorithm(algorithm)
    if not clustering_func:
        logger.error(f"Clustering algorithm '{algorithm}' not found")
        return np.array([-1] * len(embeddings)), 0.0
        
    # Perform clustering
    silhouette = 0.0
    with time_operation(f"clustering_{algorithm}"):
        try:
            labels = clustering_func(embeddings, **kwargs)
            
            # Record metrics
            increment_counter(clustering_counter, {"algorithm": algorithm})
            
            # Calculate number of clusters
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            if -1 in unique_labels:  # Exclude noise points
                n_clusters -= 1
                
            set_gauge(cluster_count_gauge, n_clusters, {"algorithm": algorithm})
            
            # Calculate silhouette score if there are at least 2 clusters
            if n_clusters >= 2 and len(embeddings) > n_clusters:
                from sklearn.metrics import silhouette_score
                
                # Exclude noise points for silhouette calculation
                if -1 in unique_labels:
                    mask = labels != -1
                    if np.sum(mask) > n_clusters:
                        silhouette = silhouette_score(
                            embeddings[mask], 
                            labels[mask],
                            metric='euclidean'
                        )
                else:
                    silhouette = silhouette_score(
                        embeddings,
                        labels,
                        metric='euclidean'
                    )
                    
                set_gauge(silhouette_gauge, silhouette, {"algorithm": algorithm})
                
            return labels, silhouette
        except Exception as e:
            logger.error(f"Error in clustering algorithm '{algorithm}': {e}")
            return np.array([-1] * len(embeddings)), 0.0


# Register standard clustering algorithms

@register_clustering_algorithm("dbscan")
def dbscan_clustering(
    X: np.ndarray,
    eps: float = CLUSTERING_EPSILON,
    min_samples: int = CLUSTERING_MIN_SAMPLES,
    **kwargs
) -> np.ndarray:
    """
    DBSCAN clustering algorithm.
    
    Args:
        X: Input data (n_samples x n_features)
        eps: Maximum distance between samples
        min_samples: Minimum samples in neighborhood
        **kwargs: Additional parameters for sklearn DBSCAN
        
    Returns:
        Cluster labels
    """
    from sklearn.cluster import DBSCAN
    
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    return model.fit_predict(X)


@register_clustering_algorithm("denstream")
def denstream_clustering(
    X: np.ndarray,
    epsilon: float = CLUSTERING_EPSILON,
    mu: int = CLUSTERING_MIN_SAMPLES,
    beta: float = 0.5,
    **kwargs
) -> np.ndarray:
    """
    DenStream clustering algorithm from River.
    This is a wrapper that handles offline batch processing.
    
    Args:
        X: Input data (n_samples x n_features)
        epsilon: Maximum distance between samples
        mu: Minimum samples in core microcluster
        beta: Microcluster decay factor
        **kwargs: Additional parameters for River DenStream
        
    Returns:
        Cluster labels
    """
    try:
        from river.cluster import DBSCAN, DenStream
        
        # Create model
        model = DenStream(
            decaying_factor=beta,
            epsilon=epsilon,
            mu=mu,
            **kwargs
        )
        
        # Train model on each sample
        for i, sample in enumerate(X):
            model.learn_one(sample)
            
        # Extract clusters
        dbscan = DBSCAN(eps=epsilon, min_samples=mu)
        labels = np.ones(len(X), dtype=int) * -1  # Initialize as noise
        
        if len(model.p_micro_clusters) + len(model.o_micro_clusters) == 0:
            return labels
            
        # Get microcluster centers
        centers = []
        for mc in model.p_micro_clusters + model.o_micro_clusters:
            centers.append(mc.center)
            
        if not centers:
            return labels
            
        # Cluster the microcluster centers
        centers = np.array(centers)
        center_labels = dbscan.fit_predict(centers)
        
        # Assign each sample to nearest microcluster
        for i, sample in enumerate(X):
            min_dist = float('inf')
            best_cluster = -1
            
            for j, center in enumerate(centers):
                dist = np.linalg.norm(sample - center)
                if dist < min_dist and dist <= epsilon:
                    min_dist = dist
                    best_cluster = center_labels[j]
                    
            labels[i] = best_cluster
            
        return labels
    except ImportError:
        logger.error("River package not available for DenStream clustering")
        return np.array([-1] * len(X))
    except Exception as e:
        logger.error(f"Error in DenStream clustering: {e}")
        return np.array([-1] * len(X))


@register_clustering_algorithm("clustream")
def clustream_clustering(
    X: np.ndarray,
    time_decay: float = 0.1,
    max_micro_clusters: int = 100,
    epsilon: float = CLUSTERING_EPSILON,
    **kwargs
) -> np.ndarray:
    """
    CluStream clustering algorithm from River.
    This is a wrapper that handles offline batch processing.
    
    Args:
        X: Input data (n_samples x n_features)
        time_decay: Temporal decay rate
        max_micro_clusters: Maximum number of microclusters
        epsilon: Final clustering distance
        **kwargs: Additional parameters for River CluStream
        
    Returns:
        Cluster labels
    """
    try:
        from river.cluster import KMeans, CluStream
        import time
        
        # Create model
        model = CluStream(
            time_window=1 / time_decay if time_decay > 0 else 1000,
            max_micro_clusters=max_micro_clusters,
            **kwargs
        )
        
        # Train model on each sample
        current_time = time.time()
        for i, sample in enumerate(X):
            # Simulate time progression
            timestamp = current_time + i * 0.001
            model.learn_one(sample, timestamp=timestamp)
            
        # Extract clusters
        kmeans = KMeans(n_clusters=10)  # Initial cluster count
        labels = np.ones(len(X), dtype=int) * -1  # Initialize as noise
        
        if len(model.micro_clusters) == 0:
            return labels
            
        # Get microcluster centers
        centers = []
        for mc in model.micro_clusters:
            centers.append(mc.center)
            
        if not centers or len(centers) < 2:
            return labels
            
        # Determine optimal cluster count
        from sklearn.cluster import KMeans as SKKMeans
        from sklearn.metrics import silhouette_score
        
        centers = np.array(centers)
        best_score = -1
        best_kmeans = None
        best_k = 2
        
        # Try different cluster counts
        max_k = min(10, len(centers) - 1)
        for k in range(2, max_k + 1):
            kmeans = SKKMeans(n_clusters=k, n_init=3)
            center_labels = kmeans.fit_predict(centers)
            
            if len(set(center_labels)) >= 2:
                score = silhouette_score(centers, center_labels)
                if score > best_score:
                    best_score = score
                    best_kmeans = kmeans
                    best_k = k
                    
        # Use best K-means model or default to k=2
        if best_kmeans is None:
            best_kmeans = SKKMeans(n_clusters=2, n_init=3)
            best_kmeans.fit(centers)
            
        center_labels = best_kmeans.predict(centers)
        
        # Assign each sample to nearest microcluster
        for i, sample in enumerate(X):
            min_dist = float('inf')
            best_cluster = -1
            
            for j, center in enumerate(centers):
                dist = np.linalg.norm(sample - center)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = center_labels[j]
                    
            if min_dist <= epsilon:
                labels[i] = best_cluster
                
        return labels
    except ImportError:
        logger.error("River package not available for CluStream clustering")
        return np.array([-1] * len(X))
    except Exception as e:
        logger.error(f"Error in CluStream clustering: {e}")
        return np.array([-1] * len(X))


@register_clustering_algorithm("kmeans")
def kmeans_clustering(
    X: np.ndarray,
    n_clusters: int = 5,
    **kwargs
) -> np.ndarray:
    """
    K-means clustering algorithm.
    
    Args:
        X: Input data (n_samples x n_features)
        n_clusters: Number of clusters
        **kwargs: Additional parameters for sklearn KMeans
        
    Returns:
        Cluster labels
    """
    from sklearn.cluster import KMeans
    
    model = KMeans(n_clusters=n_clusters, n_init=10, **kwargs)
    return model.fit_predict(X)


@register_clustering_algorithm("birch")
def birch_clustering(
    X: np.ndarray,
    threshold: float = 0.5,
    branching_factor: int = 50,
    n_clusters: int = None,
    **kwargs
) -> np.ndarray:
    """
    BIRCH clustering algorithm.
    
    Args:
        X: Input data (n_samples x n_features)
        threshold: Radius of the subcluster
        branching_factor: Maximum number of subclusters in each node
        n_clusters: Number of clusters for global clustering
        **kwargs: Additional parameters for sklearn Birch
        
    Returns:
        Cluster labels
    """
    from sklearn.cluster import Birch
    
    model = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
        **kwargs
    )
    return model.fit_predict(X)


@register_clustering_algorithm("agglomerative")
def agglomerative_clustering(
    X: np.ndarray,
    n_clusters: int = 5,
    linkage: str = "ward",
    **kwargs
) -> np.ndarray:
    """
    Agglomerative clustering algorithm.
    
    Args:
        X: Input data (n_samples x n_features)
        n_clusters: Number of clusters
        linkage: Linkage criterion
        **kwargs: Additional parameters for sklearn AgglomerativeClustering
        
    Returns:
        Cluster labels
    """
    from sklearn.cluster import AgglomerativeClustering
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        **kwargs
    )
    return model.fit_predict(X)