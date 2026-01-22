"""
Membership Functions Module
Provides various membership function types and adaptive partitioning for fuzzy systems.
Includes techniques to handle low-accuracy datasets through adaptive MF optimization.
"""

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional


class MembershipFunctionManager:
    """
    Manages membership functions for fuzzy variables.
    Supports multiple MF types and adaptive partitioning strategies.
    """
    
    MF_TYPES = ['triangular', 'trapezoidal', 'gaussian', 'sigmoid', 'bell']
    
    def __init__(self, n_partitions: int = 5, mf_type: str = 'triangular'):
        self.n_partitions = n_partitions
        self.mf_type = mf_type
        self.mf_params = {}  # {feature_idx: [(params), ...]}
        self.feature_ranges = {}
        self.linguistic_labels = self._generate_labels(n_partitions)
    
    def _generate_labels(self, n: int) -> List[str]:
        """Generate linguistic labels based on number of partitions."""
        if n == 3:
            return ['Low', 'Medium', 'High']
        elif n == 5:
            return ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
        elif n == 7:
            return ['VeryLow', 'Low', 'MediumLow', 'Medium', 'MediumHigh', 'High', 'VeryHigh']
        else:
            return [f'L{i}' for i in range(n)]
    
    def fit(self, X: np.ndarray, method: str = 'uniform', y: np.ndarray = None):
        """
        Fit membership functions to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            method: Partitioning method ('uniform', 'quantile', 'kmeans', 'adaptive', 'class_aware')
            y: Target labels (required for class_aware method)
        """
        n_features = X.shape[1]
        
        for feat_idx in range(n_features):
            feat_data = X[:, feat_idx]
            self.feature_ranges[feat_idx] = (np.min(feat_data), np.max(feat_data))
            
            if method == 'uniform':
                params = self._uniform_partition(feat_data)
            elif method == 'quantile':
                params = self._quantile_partition(feat_data)
            elif method == 'kmeans':
                params = self._kmeans_partition(feat_data)
            elif method == 'adaptive':
                params = self._adaptive_partition(feat_data)
            elif method == 'class_aware' and y is not None:
                params = self._class_aware_partition(feat_data, y)
            else:
                params = self._uniform_partition(feat_data)
            
            self.mf_params[feat_idx] = params
        
        return self
    
    def _uniform_partition(self, data: np.ndarray) -> List[Tuple]:
        """Create uniformly distributed membership functions."""
        min_val, max_val = np.min(data), np.max(data)
        margin = (max_val - min_val) * 0.1
        min_val -= margin
        max_val += margin
        
        centers = np.linspace(min_val, max_val, self.n_partitions)
        width = (max_val - min_val) / (self.n_partitions - 1)
        
        params = []
        for i, center in enumerate(centers):
            if self.mf_type == 'triangular':
                left = center - width if i > 0 else min_val - width
                right = center + width if i < self.n_partitions - 1 else max_val + width
                params.append((left, center, right))
            elif self.mf_type == 'gaussian':
                sigma = width / 2.5
                params.append((center, sigma))
            elif self.mf_type == 'trapezoidal':
                left_foot = center - width
                left_shoulder = center - width/3
                right_shoulder = center + width/3
                right_foot = center + width
                params.append((left_foot, left_shoulder, right_shoulder, right_foot))
            elif self.mf_type == 'bell':
                params.append((width/2, 2.0, center))  # (a, b, c) for generalized bell
        
        return params
    
    def _quantile_partition(self, data: np.ndarray) -> List[Tuple]:
        """Create membership functions based on data quantiles."""
        quantiles = np.linspace(0, 100, self.n_partitions + 2)[1:-1]
        centers = np.percentile(data, quantiles)
        
        min_val, max_val = np.min(data), np.max(data)
        margin = (max_val - min_val) * 0.1
        
        params = []
        for i, center in enumerate(centers):
            if i == 0:
                left = min_val - margin
            else:
                left = (centers[i-1] + center) / 2
            
            if i == len(centers) - 1:
                right = max_val + margin
            else:
                right = (center + centers[i+1]) / 2
            
            if self.mf_type == 'triangular':
                params.append((left, center, right))
            elif self.mf_type == 'gaussian':
                sigma = (right - left) / 4
                params.append((center, sigma))
        
        return params
    
    def _kmeans_partition(self, data: np.ndarray) -> List[Tuple]:
        """Create membership functions using K-means clustering."""
        data_reshaped = data.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_partitions, random_state=42, n_init=10)
        kmeans.fit(data_reshaped)
        
        centers = np.sort(kmeans.cluster_centers_.flatten())
        min_val, max_val = np.min(data), np.max(data)
        margin = (max_val - min_val) * 0.1
        
        params = []
        for i, center in enumerate(centers):
            if i == 0:
                left = min_val - margin
            else:
                left = (centers[i-1] + center) / 2
            
            if i == len(centers) - 1:
                right = max_val + margin
            else:
                right = (center + centers[i+1]) / 2
            
            if self.mf_type == 'triangular':
                params.append((left, center, right))
            elif self.mf_type == 'gaussian':
                sigma = (right - left) / 4
                params.append((center, sigma))
        
        return params
    
    def _adaptive_partition(self, data: np.ndarray) -> List[Tuple]:
        """
        Adaptive partitioning based on data density.
        Places more MFs in high-density regions for better accuracy.
        """
        try:
            kde = gaussian_kde(data)
            x_range = np.linspace(np.min(data), np.max(data), 1000)
            density = kde(x_range)
            
            # Find peaks in density
            cumsum = np.cumsum(density)
            cumsum = cumsum / cumsum[-1]
            
            # Place centers at equal probability mass points
            centers = []
            for i in range(self.n_partitions):
                target = (i + 0.5) / self.n_partitions
                idx = np.argmin(np.abs(cumsum - target))
                centers.append(x_range[idx])
            
            centers = np.array(centers)
        except:
            # Fallback to uniform if KDE fails
            return self._uniform_partition(data)
        
        min_val, max_val = np.min(data), np.max(data)
        margin = (max_val - min_val) * 0.1
        
        params = []
        for i, center in enumerate(centers):
            if i == 0:
                left = min_val - margin
            else:
                left = (centers[i-1] + center) / 2
            
            if i == len(centers) - 1:
                right = max_val + margin
            else:
                right = (center + centers[i+1]) / 2
            
            if self.mf_type == 'triangular':
                params.append((left, center, right))
            elif self.mf_type == 'gaussian':
                sigma = (right - left) / 4
                params.append((center, sigma))
        
        return params
    
    def _class_aware_partition(self, data: np.ndarray, y: np.ndarray) -> List[Tuple]:
        """
        Class-aware partitioning that considers class boundaries.
        Improves accuracy by placing MFs at class decision boundaries.
        """
        unique_classes = np.unique(y)
        class_means = []
        class_stds = []
        
        for cls in unique_classes:
            cls_data = data[y == cls]
            class_means.append(np.mean(cls_data))
            class_stds.append(np.std(cls_data))
        
        # Combine class statistics with uniform partitioning
        min_val, max_val = np.min(data), np.max(data)
        margin = (max_val - min_val) * 0.1
        
        # Create centers that include class means
        uniform_centers = np.linspace(min_val, max_val, self.n_partitions)
        all_centers = np.concatenate([uniform_centers, class_means])
        
        # Select n_partitions centers using k-means on the combined set
        if len(all_centers) > self.n_partitions:
            kmeans = KMeans(n_clusters=self.n_partitions, random_state=42, n_init=10)
            kmeans.fit(all_centers.reshape(-1, 1))
            centers = np.sort(kmeans.cluster_centers_.flatten())
        else:
            centers = np.sort(all_centers)[:self.n_partitions]
        
        params = []
        for i, center in enumerate(centers):
            if i == 0:
                left = min_val - margin
            else:
                left = (centers[i-1] + center) / 2
            
            if i == len(centers) - 1:
                right = max_val + margin
            else:
                right = (center + centers[i+1]) / 2
            
            if self.mf_type == 'triangular':
                params.append((left, center, right))
            elif self.mf_type == 'gaussian':
                sigma = (right - left) / 4
                params.append((center, sigma))
        
        return params
    
    def fuzzify(self, x: float, feature_idx: int) -> np.ndarray:
        """
        Compute membership degrees for a value across all fuzzy sets.
        """
        params = self.mf_params[feature_idx]
        memberships = np.zeros(self.n_partitions)
        
        for i, p in enumerate(params):
            memberships[i] = self._compute_membership(x, p)
        
        return memberships
    
    def fuzzify_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Fuzzify entire dataset.
        
        Returns:
            Fuzzified matrix (n_samples, n_features, n_partitions)
        """
        n_samples, n_features = X.shape
        fuzzified = np.zeros((n_samples, n_features, self.n_partitions))
        
        for i in range(n_samples):
            for j in range(n_features):
                fuzzified[i, j, :] = self.fuzzify(X[i, j], j)
        
        return fuzzified
    
    def _compute_membership(self, x: float, params: Tuple) -> float:
        """Compute membership degree based on MF type."""
        if self.mf_type == 'triangular':
            return self._triangular_mf(x, *params)
        elif self.mf_type == 'gaussian':
            return self._gaussian_mf(x, *params)
        elif self.mf_type == 'trapezoidal':
            return self._trapezoidal_mf(x, *params)
        elif self.mf_type == 'bell':
            return self._bell_mf(x, *params)
        else:
            return self._triangular_mf(x, *params)
    
    @staticmethod
    def _triangular_mf(x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function."""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:  # b < x < c
            return (c - x) / (c - b) if c != b else 1.0
    
    @staticmethod
    def _gaussian_mf(x: float, mean: float, sigma: float) -> float:
        """Gaussian membership function."""
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    
    @staticmethod
    def _trapezoidal_mf(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function."""
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c) if d != c else 1.0
    
    @staticmethod
    def _bell_mf(x: float, a: float, b: float, c: float) -> float:
        """Generalized bell membership function."""
        return 1.0 / (1.0 + np.abs((x - c) / a) ** (2 * b))
    
    def get_dominant_fuzzy_set(self, x: float, feature_idx: int) -> int:
        """Get the index of the fuzzy set with highest membership."""
        memberships = self.fuzzify(x, feature_idx)
        return np.argmax(memberships)
    
    def update_params(self, feature_idx: int, partition_idx: int, new_params: Tuple):
        """Update parameters for a specific membership function."""
        self.mf_params[feature_idx][partition_idx] = new_params
    
    def get_params_flat(self) -> np.ndarray:
        """Get all MF parameters as a flat array (for GA optimization)."""
        params_list = []
        for feat_idx in sorted(self.mf_params.keys()):
            for p in self.mf_params[feat_idx]:
                params_list.extend(p)
        return np.array(params_list)
    
    def set_params_flat(self, flat_params: np.ndarray):
        """Set MF parameters from a flat array (for GA optimization)."""
        idx = 0
        params_per_mf = len(self.mf_params[0][0])
        
        for feat_idx in sorted(self.mf_params.keys()):
            new_params = []
            for _ in range(self.n_partitions):
                p = tuple(flat_params[idx:idx + params_per_mf])
                new_params.append(p)
                idx += params_per_mf
            self.mf_params[feat_idx] = new_params
    
    def get_linguistic_term(self, partition_idx: int) -> str:
        """Get linguistic label for a partition index."""
        return self.linguistic_labels[partition_idx]
    
    def copy(self):
        """Create a deep copy of this manager."""
        new_manager = MembershipFunctionManager(self.n_partitions, self.mf_type)
        new_manager.feature_ranges = self.feature_ranges.copy()
        new_manager.mf_params = {k: list(v) for k, v in self.mf_params.items()}
        return new_manager
