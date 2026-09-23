import pandas as pd
from typing import Dict, Any

from model.detect import score_reading

class DecisionEngine:
    """
    Canonical Detection Engine that unifies logic across live inference, replay, and batch evaluation.
    This class serves as the single source of truth for the entire detection pipeline.
    """
    
    @staticmethod
    def decide(reading: dict, 
               station_history: pd.DataFrame, 
               peer_snapshot: dict, 
               model_artifact: dict, 
               state: Any = None,
               precomputed_features: pd.Series = None,
               precomputed_neighbors: dict = None,
               precomputed_history_featured: pd.DataFrame = None) -> dict:
        """
        Main canonical entry point.
        
        Args:
            reading: The current raw reading to evaluate.
            station_history: The causal history buffer for the station.
            peer_snapshot: Dictionary mapping neighbor IDs to their history dataframes.
            model_artifact: The loaded isolation forest model/thresholds.
            state: Optional state/explainer instance.
            
        Returns:
            The standard verdict dict.
        """
        # We delegate to score_reading which now contains the fully unified and 
        # architecturally correct logic (parameter-specific spatial checks, quorum limits, etc).
        verdict = score_reading(reading, station_history, model_artifact, peer_snapshot, state, 
                                precomputed_features=precomputed_features, 
                                precomputed_neighbors=precomputed_neighbors,
                                precomputed_history_featured=precomputed_history_featured)
        return verdict
