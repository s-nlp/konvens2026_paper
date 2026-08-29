import numpy as np
from scipy.special import betainc
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Tuple, Optional

class ReASCMetrics:
    @staticmethod
    def compute_confidence(log_probs: List[float], window_size: int = 128) -> float:
        """
        Computes 'Bottom 10% Group Confidence' as defined in the paper.
        """
        if log_probs is None or len(log_probs) == 0:
            return -np.inf
        
        # If sequence is shorter than window, take average of whole sequence
        if len(log_probs) <= window_size:
            return np.mean(log_probs)
        
        # Sliding window averages
        # Efficient convolution for moving average
        kernel = np.ones(window_size) / window_size
        # Valid padding ensures we don't pad with zeros, only using real data
        window_scores = np.convolve(log_probs, kernel, mode='valid')
        
        # Sort and take bottom 10%
        n_bottom = max(1, int(len(window_scores) * 0.10))
        sorted_scores = np.sort(window_scores)
        bottom_10_percent = sorted_scores[:n_bottom]
        
        return float(np.mean(bottom_10_percent))

    @staticmethod
    def standardize_confidence(score: float, stats: Tuple[float, float]) -> float:
        """Z-score normalization using calibration stats (mu, sigma)."""
        mu, sigma = stats
        if sigma == 0:
            return 0.0
        return (score - mu) / sigma

class ReASCStopper:
    def __init__(self, target_accuracy: float = 0.95, lambda_param: float = 0.7):
        self.threshold = target_accuracy
        self.lambda_param = lambda_param

    def get_beta_confidence(self, counts_most: float, counts_second: float) -> float:
        """
        Calculates P(p1 > p2 | V) using Regularized Incomplete Beta Function.
        Eq 6 in paper: 1 - I_{0.5}(alpha, beta)
        """
        alpha = counts_most + 1
        beta = counts_second + 1
        
        # betainc is the regularized incomplete beta function
        # We want 1 - I_{0.5}(alpha, beta)
        # Note: scipy's betainc matches the paper's notation
        prob = 1.0 - betainc(alpha, beta, 0.5)
        return prob

    def compute_weight(self, raw_confidence: float, stats: Tuple[float, float]) -> float:
        """
        Computes confidence-weighted update.
        Eq 5: v(y) <- v(y) + max(1, exp(lambda * z(y)))
        """
        z_score = ReASCMetrics.standardize_confidence(raw_confidence, stats)
        weight = max(1.0, np.exp(self.lambda_param * z_score))
        return weight

class CalibrationManager:
    def __init__(self):
        self.gmm = GaussianMixture(n_components=2, random_state=42)
        
    def online_calibration(self, batch_confidences: List[float], p_target: float = 0.9) -> float:
        """
        Algorithm 2: Online Gating Threshold Calibration using GMM.
        This fits a GMM to the unlabeled confidences of the current batch/test set.
        """
        X = np.array(batch_confidences).reshape(-1, 1)
        self.gmm.fit(X)
        
        # Identify which component is 'correct' (higher mean confidence)
        means = self.gmm.means_.flatten()
        correct_idx = np.argmax(means)
        
        # 1. Surrogate correct mean
        mu_approx = means[correct_idx]
        
        # 2. Posterior-based search
        # We search for the smallest t where P(z=correct | confidence=t) >= p_target
        sorted_conf = np.sort(batch_confidences)
        tau_post = mu_approx # fallback
        
        for t in sorted_conf:
            # Predict posterior prob of being in the 'correct' cluster
            # predict_proba returns [n_samples, n_components]
            probs = self.gmm.predict_proba([[t]])[0]
            if probs[correct_idx] >= p_target:
                tau_post = t
                break
                
        # Final threshold logic:
        # Instead of strict max(mu_approx, tau_post), we favor tau_post 
        # to allow for the P-target optimization to actually work.
        return tau_post
