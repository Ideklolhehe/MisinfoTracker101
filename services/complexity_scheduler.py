import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from app import db
from models import DetectedNarrative
from services.complexity_analyzer import ComplexityAnalyzer

logger = logging.getLogger(__name__)

class ComplexityScheduler:
    """
    Scheduler for periodic complexity analysis of narratives.
    
    This service schedules and runs periodic analyses of detected narratives to
    evaluate their complexity. It runs on a separate thread and can be configured
    to analyze narratives based on various criteria (age, status, etc.).
    """
    
    def __init__(self, interval_hours: int = 24):
        """
        Initialize the complexity scheduler.
        
        Args:
            interval_hours: Hours between scheduled runs (default: 24)
        """
        self.interval_hours = interval_hours
        self.analyzer = ComplexityAnalyzer()
        self.running = False
        self.scheduler_thread = None
        self.last_run_time = None
    
    def start(self) -> None:
        """Start the scheduler in a separate thread."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info(f"Complexity scheduler started with {self.interval_hours} hour interval")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_thread:
            # The thread is daemon, so it will terminate when the main program exits
            self.scheduler_thread = None
        logger.info("Complexity scheduler stopped")
    
    def run_single_analysis(self, narrative_id: int) -> Dict[str, Any]:
        """
        Run complexity analysis for a single narrative.
        
        Args:
            narrative_id: ID of the narrative to analyze
            
        Returns:
            Analysis results or error information
        """
        logger.info(f"Running single narrative analysis for ID {narrative_id}")
        return self.analyzer.analyze_narrative(narrative_id)
    
    def _run_scheduler(self) -> None:
        """Internal method to run the scheduler loop."""
        logger.info("Complexity analysis scheduler thread started")
        
        while self.running:
            try:
                # Check if it's time to run the scheduled analysis
                current_time = datetime.now()
                
                if (self.last_run_time is None or 
                        current_time - self.last_run_time > timedelta(hours=self.interval_hours)):
                    
                    logger.info("Starting scheduled complexity analysis")
                    analysis_results = self._run_scheduled_analysis()
                    self.last_run_time = current_time
                    
                    if analysis_results:
                        logger.info(f"Scheduled complexity analysis complete: {analysis_results}")
                    else:
                        logger.warning("Scheduled complexity analysis completed with no results")
                
                # Sleep for 30 minutes before checking again
                time.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in complexity scheduler thread: {e}")
                logger.exception(e)
                # Sleep for 5 minutes before retrying after an error
                time.sleep(300)
    
    def _run_scheduled_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run the scheduled complexity analysis.
        
        Returns:
            Analysis summary or None if error
        """
        try:
            # Identify narratives for analysis based on criteria
            narratives_to_analyze = self._get_narratives_for_analysis()
            
            if not narratives_to_analyze:
                logger.info("No narratives meet criteria for scheduled analysis")
                return None
            
            # Process each narrative
            successful = 0
            failed = 0
            
            for narrative in narratives_to_analyze:
                try:
                    logger.info(f"Analyzing complexity for narrative {narrative.id}")
                    result = self.analyzer.analyze_narrative(narrative.id)
                    
                    if "error" in result:
                        logger.warning(f"Failed to analyze narrative {narrative.id}: {result['error']}")
                        failed += 1
                    else:
                        successful += 1
                    
                    # Add a delay to avoid rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error analyzing narrative {narrative.id}: {e}")
                    failed += 1
            
            return {
                "total_analyzed": successful + failed,
                "successful": successful,
                "failed": failed
            }
            
        except Exception as e:
            logger.error(f"Error in scheduled analysis: {e}")
            logger.exception(e)
            return None
    
    def _get_narratives_for_analysis(self) -> list:
        """
        Get narratives that should be analyzed in the scheduled run.
        
        Returns:
            List of narrative objects for analysis
        """
        try:
            # Define criteria for selection
            # 1. Active narratives
            # 2. High threat level (3+)
            # 3. No complexity analysis or analysis older than 7 days
            # 4. Limit to 5 narratives per run to avoid rate limits
            
            seven_days_ago = datetime.now() - timedelta(days=7)
            
            # Find narratives with no complexity analysis
            unanalyzed_narratives = DetectedNarrative.query.filter(
                DetectedNarrative.status == 'active',
                DetectedNarrative.threat_level >= 3
            ).order_by(
                DetectedNarrative.created_at.desc()
            ).limit(5).all()
            
            # Filter out narratives that already have recent analysis
            narratives_to_analyze = []
            for narrative in unanalyzed_narratives:
                has_recent_analysis = False
                if narrative.meta_data:
                    try:
                        import json
                        metadata = json.loads(narrative.meta_data)
                        if 'complexity_analysis' in metadata:
                            analyzed_at = metadata['complexity_analysis'].get('analyzed_at')
                            if analyzed_at:
                                analysis_time = datetime.fromtimestamp(analyzed_at)
                                if analysis_time > seven_days_ago:
                                    has_recent_analysis = True
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        pass
                
                if not has_recent_analysis:
                    narratives_to_analyze.append(narrative)
                    
                    # Limit to 5 narratives per run
                    if len(narratives_to_analyze) >= 5:
                        break
            
            return narratives_to_analyze
            
        except Exception as e:
            logger.error(f"Error selecting narratives for analysis: {e}")
            return []