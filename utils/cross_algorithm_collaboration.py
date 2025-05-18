"""
Cross-Algorithm Collaboration module for the CIVILIAN system.

This module coordinates the three enhanced clustering algorithms:
1. EnhancedDenStream - For real-time clustering
2. EnhancedCluStream - For temporal pattern analysis
3. EnhancedSECLEDS - For detecting subtle and novel patterns

It implements ensemble techniques to combine strengths of all algorithms
and provide a unified view of misinformation narrative clusters.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from datetime import datetime
import threading
import time
import json

from utils.enhanced_denstream import EnhancedDenStream
from utils.enhanced_clustream import EnhancedCluStream
from utils.enhanced_secleds import EnhancedSECLEDS

# Configure logging
logger = logging.getLogger(__name__)

class NarrativeClusterResult:
    """
    Result of narrative clustering across multiple algorithms.
    """
    def __init__(self, narrative_id: str):
        """
        Initialize a narrative cluster result.
        
        Args:
            narrative_id: ID of the narrative
        """
        self.narrative_id = narrative_id
        self.clusters = {}  # Dict of {algorithm: (cluster_id, confidence)}
        self.ensemble_cluster = -1  # Final cluster decision
        self.ensemble_confidence = 0.0  # Confidence in ensemble decision
        self.timestamp = datetime.utcnow()
        self.is_novel = False
        self.is_outlier = False
        self.escalation_status = "normal"  # One of: normal, review, escalated
        self.propagation_score = 0.0  # Estimated propagation potential
        self.threat_level = 0  # 0-5 scale
        
    def add_algorithm_result(self, algorithm: str, cluster_id: int, 
                           confidence: float = 0.5) -> None:
        """
        Add result from a specific algorithm.
        
        Args:
            algorithm: Name of the algorithm
            cluster_id: Assigned cluster ID
            confidence: Confidence in the assignment
        """
        self.clusters[algorithm] = (cluster_id, confidence)
        
        # Check if outlier across all algorithms
        if all(c[0] == -1 for c in self.clusters.values() if c is not None):
            self.is_outlier = True
            
    def compute_ensemble_result(self, weights: Dict[str, float] = None) -> None:
        """
        Compute the final ensemble result based on all algorithms.
        
        Args:
            weights: Optional dict mapping algorithm names to weights
        """
        if not self.clusters:
            return
            
        # Default weights if none provided
        if weights is None:
            weights = {
                "denstream": 0.4,  # Good for real-time clustering
                "clustream": 0.3,  # Good for temporal patterns
                "secleds": 0.3,    # Good for novel patterns
            }
            
        # Normalize weights
        total = sum(weights.get(algo, 1.0) for algo in self.clusters.keys())
        if total > 0:
            norm_weights = {algo: weights.get(algo, 1.0) / total for algo in self.clusters.keys()}
        else:
            norm_weights = {algo: 1.0 / len(self.clusters) for algo in self.clusters.keys()}
            
        # Get weighted votes
        cluster_votes = {}
        confidence_sum = {}
        
        for algo, (cluster_id, confidence) in self.clusters.items():
            if cluster_id == -1:  # Skip noise assignments
                continue
                
            # Weight by algorithm confidence and importance
            weighted_vote = confidence * norm_weights.get(algo, 1.0 / len(self.clusters))
            
            if cluster_id not in cluster_votes:
                cluster_votes[cluster_id] = 0.0
                confidence_sum[cluster_id] = 0.0
                
            cluster_votes[cluster_id] += weighted_vote
            confidence_sum[cluster_id] += confidence
            
        # Find the cluster with highest weighted vote
        if cluster_votes:
            # Get best cluster
            best_cluster = max(cluster_votes.items(), key=lambda x: x[1])
            self.ensemble_cluster = best_cluster[0]
            
            # Calculate confidence as normalized vote strength
            total_votes = sum(cluster_votes.values())
            if total_votes > 0:
                self.ensemble_confidence = best_cluster[1] / total_votes
            else:
                self.ensemble_confidence = 0.0
                
            # Factor in average confidence from algorithms that assigned this cluster
            algo_confidence = confidence_sum[self.ensemble_cluster] / sum(1 for _, (c, _) in self.clusters.items() if c == self.ensemble_cluster)
            # Blend vote confidence with algorithm confidence
            self.ensemble_confidence = 0.7 * self.ensemble_confidence + 0.3 * algo_confidence
        else:
            # All algorithms assigned noise
            self.ensemble_cluster = -1
            self.ensemble_confidence = 0.0
            
        # Determine escalation status
        if self.ensemble_confidence < 0.4:
            self.escalation_status = "review"
        elif self.is_novel:
            self.escalation_status = "review"


class ClusterImpactScoring:
    """
    Computes impact scores for clusters based on various factors.
    """
    def __init__(self):
        """Initialize the impact scoring system."""
        self.size_weight = 0.3
        self.growth_weight = 0.3
        self.novelty_weight = 0.2
        self.stability_weight = 0.1
        self.propagation_weight = 0.1
        
    def calculate_impact_score(self, cluster_data: Dict[str, Any]) -> float:
        """
        Calculate impact score for a cluster.
        
        Args:
            cluster_data: Data about the cluster
            
        Returns:
            Impact score (0-1)
        """
        score_components = {}
        
        # Size component
        cluster_size = cluster_data.get("size", 0)
        # Normalize size: sigmoid function to map 0-infinity to 0-1
        size_score = 1.0 / (1.0 + np.exp(-0.1 * (cluster_size - 10)))
        score_components["size"] = size_score
        
        # Growth component
        growth_rate = cluster_data.get("growth_rate", 0.0)
        # Cap at 10.0 (1000% growth) and normalize
        growth_score = min(1.0, growth_rate / 10.0)
        score_components["growth"] = growth_score
        
        # Novelty component
        novelty = cluster_data.get("novelty", 0.0)
        score_components["novelty"] = novelty
        
        # Stability component (inverse of volatility)
        volatility = cluster_data.get("volatility", 0.5)
        stability_score = 1.0 - volatility
        score_components["stability"] = stability_score
        
        # Propagation potential
        propagation = cluster_data.get("propagation", 0.0)
        score_components["propagation"] = propagation
        
        # Calculate weighted score
        impact_score = (
            self.size_weight * size_score +
            self.growth_weight * growth_score +
            self.novelty_weight * novelty +
            self.stability_weight * stability_score +
            self.propagation_weight * propagation
        )
        
        # Record components for explanation
        cluster_data["impact_components"] = score_components
        
        return impact_score


class CrossAlgorithmCollaborator:
    """
    Main class for cross-algorithm collaboration.
    """
    def __init__(self, 
                denstream_params: Dict[str, Any] = None,
                clustream_params: Dict[str, Any] = None, 
                secleds_params: Dict[str, Any] = None):
        """
        Initialize the cross-algorithm collaborator.
        
        Args:
            denstream_params: Parameters for EnhancedDenStream
            clustream_params: Parameters for EnhancedCluStream
            secleds_params: Parameters for EnhancedSECLEDS
        """
        # Initialize algorithms with default or provided parameters
        self.denstream = EnhancedDenStream(**(denstream_params or {}))
        self.clustream = EnhancedCluStream(**(clustream_params or {}))
        self.secleds = EnhancedSECLEDS(**(secleds_params or {}))
        
        # Storage for results
        self.narrative_results = {}  # Dict of {narrative_id: NarrativeClusterResult}
        self.processed_count = 0
        
        # Cluster impact scoring
        self.impact_scorer = ClusterImpactScoring()
        
        # Algorithm weights for ensemble
        self.algorithm_weights = {
            "denstream": 0.4,
            "clustream": 0.3,
            "secleds": 0.3
        }
        
        # Cluster mapping between algorithms
        self.cluster_mapping = {
            "denstream": {},  # {cluster_id: [narrative_ids]}
            "clustream": {},
            "secleds": {}
        }
        
        # Thread lock for concurrency
        self.lock = threading.Lock()
        
        logger.info("CrossAlgorithmCollaborator initialized")
        
    def process_narrative(self, 
                         vector: np.ndarray, 
                         narrative_id: str,
                         timestamp: Optional[datetime] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a narrative through all algorithms and compute ensemble result.
        
        Args:
            vector: Vector representation of the narrative
            narrative_id: ID of the narrative
            timestamp: Optional timestamp
            metadata: Optional additional metadata
            
        Returns:
            Dictionary with processing results
        """
        with self.lock:
            if timestamp is None:
                timestamp = datetime.utcnow()
                
            metadata = metadata or {}
            
            # Initialize result
            result = NarrativeClusterResult(narrative_id)
            result.timestamp = timestamp
            
            # Track any propagation data
            if "propagation" in metadata:
                result.propagation_score = float(metadata["propagation"])
                
            # Process with DenStream (real-time clustering)
            denstream_cluster = self.denstream.learn_one(
                vector, 
                narrative_id=narrative_id,
                spread_rate=result.propagation_score
            )
            result.add_algorithm_result("denstream", denstream_cluster, 0.8)
            
            # Update cluster mapping
            if denstream_cluster not in self.cluster_mapping["denstream"]:
                self.cluster_mapping["denstream"][denstream_cluster] = []
            self.cluster_mapping["denstream"][denstream_cluster].append(narrative_id)
            
            # Process with CluStream (temporal clustering)
            clustream_cluster = self.clustream.learn_one(
                vector,
                timestamp=timestamp,
                narrative_id=narrative_id
            )
            result.add_algorithm_result("clustream", clustream_cluster, 0.7)
            
            # Update cluster mapping
            if clustream_cluster not in self.cluster_mapping["clustream"]:
                self.cluster_mapping["clustream"][clustream_cluster] = []
            self.cluster_mapping["clustream"][clustream_cluster].append(narrative_id)
            
            # Process with SeCLEDS (concept drift and novelty detection)
            secleds_result = self.secleds.partial_fit(
                vector,
                sequence_id=narrative_id,
                timestamp=timestamp
            )
            secleds_cluster = secleds_result["cluster_id"]
            secleds_confidence = secleds_result["confidence"]
            result.add_algorithm_result("secleds", secleds_cluster, secleds_confidence)
            
            # Update cluster mapping
            if secleds_cluster not in self.cluster_mapping["secleds"]:
                self.cluster_mapping["secleds"][secleds_cluster] = []
            self.cluster_mapping["secleds"][secleds_cluster].append(narrative_id)
            
            # Check if SeCLEDS detected a novel pattern
            if secleds_result["is_novel"]:
                result.is_novel = True
                
            # Check if escalation is needed
            if secleds_result["is_escalated"]:
                result.escalation_status = "review"
                
            # Compute ensemble result
            result.compute_ensemble_result(self.algorithm_weights)
            
            # Store result
            self.narrative_results[narrative_id] = result
            self.processed_count += 1
            
            # Share information between algorithms
            self._share_algorithm_insights(narrative_id, result)
            
            # Compute threat level based on ensemble result and propagation
            self._compute_threat_level(result)
            
            # Format return value
            return self._format_result(result)
            
    def _share_algorithm_insights(self, narrative_id: str, result: NarrativeClusterResult) -> None:
        """
        Share insights between algorithms for better collaboration.
        
        Args:
            narrative_id: ID of the narrative
            result: Clustering result
        """
        # Share SeCLEDS novelty detection with other algorithms
        if result.is_novel:
            # For future implementation: create new clusters in other algorithms
            pass
            
        # Share propagation data with SeCLEDS
        if result.propagation_score > 0:
            # For future implementation: adjust confidence thresholds
            pass
            
        # Every 50 narratives, share cluster mappings between algorithms
        if self.processed_count % 50 == 0:
            self._update_cluster_correspondence()
            
    def _update_cluster_correspondence(self) -> None:
        """Update the correspondence between clusters from different algorithms."""
        # For each algorithm pair, calculate overlap in cluster assignments
        pairs = [
            ("denstream", "clustream"),
            ("denstream", "secleds"),
            ("clustream", "secleds")
        ]
        
        for algo1, algo2 in pairs:
            # Calculate Jaccard similarity between clusters
            for cluster1, narratives1 in self.cluster_mapping[algo1].items():
                if cluster1 == -1:
                    continue  # Skip noise cluster
                    
                for cluster2, narratives2 in self.cluster_mapping[algo2].items():
                    if cluster2 == -1:
                        continue  # Skip noise cluster
                        
                    # Calculate overlap
                    set1 = set(narratives1)
                    set2 = set(narratives2)
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    
                    if union > 0:
                        similarity = intersection / union
                        
                        # If significant overlap, share this information
                        if similarity > 0.5:
                            logger.debug(
                                f"Strong correspondence between {algo1} cluster {cluster1} "
                                f"and {algo2} cluster {cluster2}: {similarity:.2f}"
                            )
                            
                            # Share this information with SeCLEDS
                            if algo1 == "secleds" or algo2 == "secleds":
                                target_algo = algo1 if algo2 == "secleds" else algo2
                                source_algo = algo2 if algo2 == "secleds" else algo1
                                target_cluster = cluster1 if target_algo == algo1 else cluster2
                                
                                # For each narrative in the overlap, add cross-algorithm evidence
                                for narrative_id in set1.intersection(set2):
                                    self.secleds.add_cross_algorithm_evidence(
                                        narrative_id=narrative_id,
                                        algorithm=source_algo,
                                        cluster_id=target_cluster,
                                        confidence=similarity
                                    )
                                    
    def _compute_threat_level(self, result: NarrativeClusterResult) -> None:
        """
        Compute threat level for a narrative.
        
        Args:
            result: Clustering result
        """
        # Base threat on propagation potential and confidence
        base_threat = min(5, int(result.propagation_score * 10))
        
        # Increase if novel pattern
        if result.is_novel:
            base_threat += 1
            
        # Decrease if low confidence 
        if result.ensemble_confidence < 0.5:
            base_threat = max(1, base_threat - 1)
            
        # Cap at 0-5 range
        result.threat_level = max(0, min(5, base_threat))
        
    def _format_result(self, result: NarrativeClusterResult) -> Dict[str, Any]:
        """
        Format result for return.
        
        Args:
            result: Clustering result
            
        Returns:
            Dictionary with formatted result
        """
        return {
            "narrative_id": result.narrative_id,
            "ensemble_cluster": result.ensemble_cluster,
            "confidence": result.ensemble_confidence,
            "algorithm_clusters": {
                algo: {"cluster": cluster_id, "confidence": conf}
                for algo, (cluster_id, conf) in result.clusters.items()
            },
            "is_novel": result.is_novel,
            "is_outlier": result.is_outlier,
            "escalation_status": result.escalation_status,
            "timestamp": result.timestamp.isoformat(),
            "propagation_score": result.propagation_score,
            "threat_level": result.threat_level
        }
        
    def get_cluster_overview(self) -> Dict[str, Any]:
        """
        Get an overview of all clusters across algorithms.
        
        Returns:
            Dictionary with cluster overview
        """
        with self.lock:
            # Get current state from each algorithm
            denstream_clusters = self.denstream.get_clusters()
            clustream_data = self.clustream.get_clusters()
            secleds_stats = self.secleds.get_cluster_stats()
            
            # Get ensemble clusters
            ensemble_clusters = {}
            
            for result in self.narrative_results.values():
                cluster_id = result.ensemble_cluster
                if cluster_id == -1:
                    continue  # Skip noise
                    
                if cluster_id not in ensemble_clusters:
                    ensemble_clusters[cluster_id] = {
                        "narratives": [],
                        "algorithms": {
                            "denstream": set(),
                            "clustream": set(),
                            "secleds": set()
                        },
                        "novel_count": 0,
                        "propagation_avg": 0.0,
                        "confidence_avg": 0.0,
                        "threat_avg": 0.0
                    }
                    
                # Add narrative to ensemble cluster
                ensemble_clusters[cluster_id]["narratives"].append(result.narrative_id)
                
                # Track contributing algorithms
                for algo, (algo_cluster, _) in result.clusters.items():
                    if algo_cluster != -1:
                        ensemble_clusters[cluster_id]["algorithms"][algo].add(algo_cluster)
                        
                # Track stats
                if result.is_novel:
                    ensemble_clusters[cluster_id]["novel_count"] += 1
                    
                ensemble_clusters[cluster_id]["propagation_avg"] += result.propagation_score
                ensemble_clusters[cluster_id]["confidence_avg"] += result.ensemble_confidence
                ensemble_clusters[cluster_id]["threat_avg"] += result.threat_level
            
            # Calculate averages
            for cluster_id, data in ensemble_clusters.items():
                n_narratives = len(data["narratives"])
                if n_narratives > 0:
                    data["propagation_avg"] /= n_narratives
                    data["confidence_avg"] /= n_narratives
                    data["threat_avg"] /= n_narratives
                    
                # Calculate impact score
                cluster_data = {
                    "size": n_narratives,
                    "novelty": data["novel_count"] / max(1, n_narratives),
                    "propagation": data["propagation_avg"],
                    # Add more metrics as needed
                }
                
                data["impact_score"] = self.impact_scorer.calculate_impact_score(cluster_data)
                data["impact_components"] = cluster_data.get("impact_components", {})
                
                # Convert algorithm sets to lists for JSON serialization
                for algo in data["algorithms"]:
                    data["algorithms"][algo] = list(data["algorithms"][algo])
            
            # Rank clusters by impact score
            ranked_clusters = sorted(
                ensemble_clusters.items(),
                key=lambda x: x[1]["impact_score"],
                reverse=True
            )
            
            return {
                "ensemble_clusters": {
                    str(cluster_id): data for cluster_id, data in ranked_clusters
                },
                "algorithm_stats": {
                    "denstream": {
                        "total_clusters": len([c for c in denstream_clusters.keys() if c != -1]),
                        "total_narratives": sum(len(narratives) for cluster_id, narratives in denstream_clusters.items() if cluster_id != -1)
                    },
                    "clustream": {
                        "total_clusters": clustream_data.get("total_macro_clusters", 0),
                        "total_narratives": sum(data.get("size", 0) for data in clustream_data.get("clusters", {}).values())
                    },
                    "secleds": {
                        "total_clusters": len(secleds_stats),
                        "total_narratives": sum(data.get("member_count", 0) for data in secleds_stats.values())
                    }
                },
                "total_processed": self.processed_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    def get_temporal_alerts(self) -> List[Dict[str, Any]]:
        """
        Get temporal alerts based on significant changes in clusters.
        
        Returns:
            List of alert dictionaries
        """
        with self.lock:
            alerts = []
            
            # Get unstable DenStream clusters
            denstream_stability = self.denstream.get_stability_report()
            for cluster in denstream_stability.get("unstable_clusters", []):
                alerts.append({
                    "type": "stability_alert",
                    "algorithm": "denstream",
                    "cluster_id": cluster["cluster_id"],
                    "stability_score": cluster["stability_score"],
                    "narrative_count": cluster["narratives"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Cluster {cluster['cluster_id']} is unstable with {cluster['narratives']} narratives"
                })
                
            # Get novel narratives from SeCLEDS
            novel_narratives = self.secleds.get_novel_narratives()
            for novel in novel_narratives[:5]:  # Limit to 5 most recent
                alerts.append({
                    "type": "novelty_alert",
                    "algorithm": "secleds",
                    "narrative_id": novel["narrative_id"],
                    "novelty_score": novel["novelty_score"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Novel narrative pattern detected: {novel['narrative_id']}"
                })
                
            # Get escalated items from SeCLEDS
            escalated = self.secleds.get_escalated_items()
            for item in escalated[:5]:  # Limit to 5 most recent
                alerts.append({
                    "type": "escalation_alert",
                    "algorithm": "secleds",
                    "narrative_id": item["narrative_id"],
                    "timestamp": item["timestamp"],
                    "reason": item["reason"],
                    "message": f"Narrative requires review: {item['reason']}"
                })
                
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return alerts
            
    def get_narrative_relationship_mapping(self) -> Dict[str, Any]:
        """
        Create a graph of relationships between narratives and clusters.
        
        Returns:
            Dictionary with graph data
        """
        with self.lock:
            nodes = []
            edges = []
            
            # Add ensemble clusters as nodes
            ensemble_clusters = {}
            for result in self.narrative_results.values():
                cluster_id = result.ensemble_cluster
                if cluster_id == -1:
                    continue
                    
                if cluster_id not in ensemble_clusters:
                    ensemble_clusters[cluster_id] = {
                        "narratives": [],
                        "impact_score": 0.0,
                        "propagation_avg": 0.0
                    }
                    
                ensemble_clusters[cluster_id]["narratives"].append(result.narrative_id)
                ensemble_clusters[cluster_id]["propagation_avg"] += result.propagation_score
                
            # Calculate cluster stats and add as nodes
            for cluster_id, data in ensemble_clusters.items():
                n_narratives = len(data["narratives"])
                if n_narratives > 0:
                    data["propagation_avg"] /= n_narratives
                    
                # Calculate impact score
                cluster_data = {
                    "size": n_narratives,
                    "propagation": data["propagation_avg"]
                }
                
                data["impact_score"] = self.impact_scorer.calculate_impact_score(cluster_data)
                
                # Add cluster node
                nodes.append({
                    "id": f"cluster_{cluster_id}",
                    "label": f"Cluster {cluster_id}",
                    "type": "cluster",
                    "size": n_narratives,
                    "impact": data["impact_score"],
                    "narratives": data["narratives"][:10]  # Limit to 10 for visualization
                })
                
            # Find relationships between clusters
            for cluster1_id, data1 in ensemble_clusters.items():
                narratives1 = set(data1["narratives"])
                
                for cluster2_id, data2 in ensemble_clusters.items():
                    if cluster1_id >= cluster2_id:
                        continue  # Avoid duplicates and self-connections
                        
                    narratives2 = set(data2["narratives"])
                    
                    # Calculate shared keywords or other relationships
                    # For now, just use Jaccard similarity of narratives
                    intersection = len(narratives1.intersection(narratives2))
                    union = len(narratives1.union(narratives2))
                    
                    if union > 0:
                        similarity = intersection / union
                        
                        # Add edge if similarity is significant
                        if similarity > 0.1:
                            edges.append({
                                "source": f"cluster_{cluster1_id}",
                                "target": f"cluster_{cluster2_id}",
                                "weight": similarity,
                                "shared_narratives": intersection
                            })
                            
            return {
                "nodes": nodes,
                "edges": edges,
                "total_clusters": len(ensemble_clusters),
                "total_connections": len(edges),
                "timestamp": datetime.utcnow().isoformat()
            }
            
    def get_narrative_analysis(self, narrative_id: str) -> Dict[str, Any]:
        """
        Get detailed analysis for a specific narrative.
        
        Args:
            narrative_id: ID of the narrative
            
        Returns:
            Dictionary with analysis results
        """
        with self.lock:
            if narrative_id not in self.narrative_results:
                return {"error": "Narrative not found"}
                
            result = self.narrative_results[narrative_id]
            
            # Get ensemble cluster details
            ensemble_cluster = result.ensemble_cluster
            cluster_narratives = []
            
            if ensemble_cluster != -1:
                # Find other narratives in the same cluster
                for r_id, r in self.narrative_results.items():
                    if r.ensemble_cluster == ensemble_cluster and r_id != narrative_id:
                        cluster_narratives.append(r_id)
                        
            # Format cross-algorithm details
            algorithm_details = {}
            for algo, (cluster_id, confidence) in result.clusters.items():
                # Get algorithm-specific insights
                if algo == "denstream":
                    # Check stability from DenStream
                    stability_report = self.denstream.get_stability_report()
                    algo_details = {
                        "stability": stability_report.get("overall_stability", 0.0),
                        "decay_rate": stability_report.get("decay_rate", 0.0)
                    }
                elif algo == "clustream":
                    # Get temporal patterns from CluStream
                    if cluster_id != -1:
                        evolution = self.clustream.get_cluster_evolution(cluster_id)
                        algo_details = {
                            "evolution": evolution.get("evolution", "unknown"),
                            "temporal_distribution": evolution.get("temporal_distribution", [])
                        }
                    else:
                        algo_details = {"evolution": "unassigned"}
                elif algo == "secleds":
                    # Get novelty and drift from SeCLEDS
                    if cluster_id != -1 and cluster_id in self.secleds.get_cluster_stats():
                        cluster_stats = self.secleds.get_cluster_stats()[cluster_id]
                        algo_details = {
                            "is_drifting": cluster_stats.get("is_drifting", False),
                            "novelty_score": cluster_stats.get("novelty_score", 0.0),
                            "cross_algorithm_evidence": cluster_stats.get("cross_algorithm_evidence", {})
                        }
                    else:
                        algo_details = {"status": "unassigned"}
                else:
                    algo_details = {}
                    
                algorithm_details[algo] = {
                    "cluster_id": cluster_id,
                    "confidence": confidence,
                    "details": algo_details
                }
                
            return {
                "narrative_id": narrative_id,
                "ensemble_cluster": ensemble_cluster,
                "confidence": result.ensemble_confidence,
                "is_novel": result.is_novel,
                "is_outlier": result.is_outlier,
                "escalation_status": result.escalation_status,
                "timestamp": result.timestamp.isoformat(),
                "propagation_score": result.propagation_score,
                "threat_level": result.threat_level,
                "algorithm_details": algorithm_details,
                "cluster_narratives": cluster_narratives[:10],  # Limit to 10
                "recommendation": self._generate_recommendation(result)
            }
            
    def _generate_recommendation(self, result: NarrativeClusterResult) -> Dict[str, Any]:
        """
        Generate recommendations for handling this narrative.
        
        Args:
            result: Clustering result
            
        Returns:
            Dictionary with recommendations
        """
        # Base recommendation on threat level and confidence
        if result.threat_level >= 4:
            priority = "high"
            action = "monitor_and_counter"
        elif result.threat_level >= 2:
            priority = "medium"
            action = "monitor"
        else:
            priority = "low"
            action = "routine"
            
        # Adjust based on confidence
        if result.ensemble_confidence < 0.5:
            confidence_note = "Low confidence in classification; manual review recommended"
            if priority != "high":
                action = "review"
        else:
            confidence_note = "Classification confidence sufficient for automated handling"
            
        # Adjust for novel patterns
        if result.is_novel:
            novelty_note = "Novel pattern detected; requires human analysis"
            action = "review"
            if priority != "high":
                priority = "medium"
        else:
            novelty_note = "Pattern matches existing clusters; standard protocols apply"
            
        return {
            "priority": priority,
            "recommended_action": action,
            "confidence_note": confidence_note,
            "novelty_note": novelty_note,
            "threat_assessment": f"Level {result.threat_level}/5 threat with {result.propagation_score:.2f} propagation potential"
        }