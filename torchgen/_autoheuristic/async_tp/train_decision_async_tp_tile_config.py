# mypy: ignore-errors
"""
Training script for async TP tile config decision tree heuristic.

Inherits from AHTrainDecisionTree to leverage the standard AutoHeuristic
training pipeline: grid search, unsafe leaf detection, confidence thresholds,
and artifact code generation.

Choices are tile config name strings like "T128x256_C2x1".
Features reuse the standard set (m, k, n, arith_intensity, m_times_k, m_times_n, k_times_n).

Usage:
    cd torchgen/_autoheuristic/async_tp
    python train_decision_async_tp_tile_config.py async_tp_tile_config_h100_data.txt \
        --heuristic-name AsyncTPTileConfigH100
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).absolute().parents[1]))

from train_decision import AHTrainDecisionTree


class AHTrainDecisionTreeAsyncTPTileConfig(AHTrainDecisionTree):
    def __init__(self):
        super().__init__()

    def add_new_features(self, results):
        """
        Features m, k, n, arith_intensity, m_times_k, m_times_n, k_times_n
        are already computed in the data by convert_tile_config_data.py.
        No additional feature engineering needed.
        """
        return (results, [])

    def get_default_config(self, row):
        """Default config when AutoHeuristic returns 'unsure'.

        Replicates the current hardcoded heuristic in _sm90_tile_heuristic():
          - T128x128_C2x1 if M*N < 2048*2048*2
          - T128x256_C2x1 otherwise
        This ensures regression-free fallback behavior.
        """
        m_times_n = row["m_times_n"]
        if m_times_n < 2048 * 2048 * 2:
            return "T128x128_C2x1"
        return "T128x256_C2x1"

    def get_allowed_wrong_prediction_pct(self):
        """
        Use 1.0 (no hard rejection) following the mixed_mm and mm trainers.
        With 18 tile config classes, strict accuracy thresholds are impractical.
        Safety is enforced by is_unsafe_leaf() instead: predictions >15% slower
        than best return None, triggering fallback to the hardcoded heuristic.
        """
        return 1.0

    def get_test_and_val_size(self):
        """
        Same split as the fuse/no_fuse and native/pipeline heuristics.
        """
        return (0.15, 0.20)

    def is_unsafe_leaf(self, row, predicted_config, choice2time):
        """
        Mark a leaf as unsafe if the predicted tile config is significantly
        slower than the best config for that shape.

        If the leaf is unsafe, the heuristic returns None (unsure), and the
        caller falls back to the default heuristic logic.
        """
        if predicted_config not in choice2time:
            return False

        predicted_time = choice2time[predicted_config]
        best_time = min(choice2time.values())

        # If predicted config is >15% slower than best, mark unsafe
        if predicted_time > 1.15 * best_time:
            return True

        return False

    def get_grid_search_values(self):
        """
        Grid search over hyperparameters. Wider depth range since 5-8 tile
        config classes need more splits to separate than binary choices.
        """
        return {
            "max_depth": [3, 4, 5, 6, 7, 8],
            "min_samples_leaf": [1, 2, 3, 5, 0.05],
            "criterion": ["gini", "entropy"],
        }


if __name__ == "__main__":
    train = AHTrainDecisionTreeAsyncTPTileConfig()
    train.generate_heuristic()
