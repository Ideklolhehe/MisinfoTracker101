"""
Export service for complexity analysis reports.
Supports exporting to PDF and CSV formats.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
from fpdf import FPDF
from io import StringIO, BytesIO

from models import DetectedNarrative

logger = logging.getLogger(__name__)

class ComplexityReportExporter:
    """Service to export complexity analysis reports in various formats."""
    
    @staticmethod
    def export_to_csv(narrative_id: int) -> Optional[BytesIO]:
        """
        Export complexity analysis for a specific narrative to CSV.
        
        Args:
            narrative_id: ID of the narrative to export
            
        Returns:
            CSV data as BytesIO object, or None if export fails
        """
        try:
            # Get narrative data
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative or not narrative.meta_data:
                logger.warning(f"Cannot export narrative {narrative_id}: not found or no metadata")
                return None
                
            # Parse metadata
            meta_data = json.loads(narrative.meta_data)
            complexity_data = meta_data.get('complexity_analysis', {})
            
            if not complexity_data or 'overall_complexity_score' not in complexity_data:
                logger.warning(f"No complexity data available for narrative {narrative_id}")
                return None
            
            # Create CSV content
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Narrative Complexity Analysis Report'])
            writer.writerow([f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
            writer.writerow([])
            
            # Write narrative info
            writer.writerow(['Narrative Information'])
            writer.writerow(['ID', narrative.id])
            writer.writerow(['Title', narrative.title])
            writer.writerow(['Status', narrative.status])
            writer.writerow(['First Detected', narrative.first_detected.strftime('%Y-%m-%d')])
            writer.writerow(['Last Updated', narrative.last_updated.strftime('%Y-%m-%d')])
            writer.writerow([])
            
            # Write complexity summary
            writer.writerow(['Complexity Summary'])
            writer.writerow(['Overall Complexity Score', complexity_data.get('overall_complexity_score')])
            writer.writerow(['Summary', complexity_data.get('summary', 'N/A')])
            writer.writerow(['Potential Impact', complexity_data.get('potential_impact', 'N/A')])
            writer.writerow(['Analysis Timestamp', 
                            datetime.fromtimestamp(int(complexity_data.get('analyzed_at', 0))).strftime('%Y-%m-%d %H:%M:%S') 
                            if complexity_data.get('analyzed_at') else 'N/A'])
            writer.writerow([])
            
            # Write detailed dimensions
            writer.writerow(['Complexity Dimensions'])
            writer.writerow(['Dimension', 'Score', 'Observations'])
            
            dimensions = [
                ('Linguistic Complexity', 'linguistic_complexity'),
                ('Logical Structure', 'logical_structure'),
                ('Rhetorical Techniques', 'rhetorical_techniques'),
                ('Emotional Manipulation', 'emotional_manipulation')
            ]
            
            for label, key in dimensions:
                dim_data = complexity_data.get(key, {})
                writer.writerow([
                    label, 
                    dim_data.get('score', 'N/A'),
                    dim_data.get('observations', 'N/A')
                ])
            
            # Get output value and convert to BytesIO
            csv_content = output.getvalue()
            csv_bytes = BytesIO(csv_content.encode('utf-8'))
            csv_bytes.seek(0)
            
            return csv_bytes
            
        except Exception as e:
            logger.error(f"Error exporting narrative {narrative_id} to CSV: {e}")
            return None
    
    @staticmethod
    def export_to_pdf(narrative_id: int) -> Optional[BytesIO]:
        """
        Export complexity analysis for a specific narrative to PDF.
        
        Args:
            narrative_id: ID of the narrative to export
            
        Returns:
            PDF data as BytesIO object, or None if export fails
        """
        try:
            # Get narrative data
            narrative = DetectedNarrative.query.get(narrative_id)
            if not narrative or not narrative.meta_data:
                logger.warning(f"Cannot export narrative {narrative_id}: not found or no metadata")
                return None
                
            # Parse metadata
            meta_data = json.loads(narrative.meta_data)
            complexity_data = meta_data.get('complexity_analysis', {})
            
            if not complexity_data or 'overall_complexity_score' not in complexity_data:
                logger.warning(f"No complexity data available for narrative {narrative_id}")
                return None
            
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Set font
            pdf.set_font("Arial", "B", 16)
            
            # Title
            pdf.cell(190, 10, "Narrative Complexity Analysis Report", ln=True, align="C")
            pdf.set_font("Arial", "", 10)
            pdf.cell(190, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(5)
            
            # Narrative information
            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 10, "Narrative Information", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(30, 8, "ID:", 0)
            pdf.cell(160, 8, str(narrative.id), ln=True)
            pdf.cell(30, 8, "Title:", 0)
            
            # Handle long titles gracefully
            title = narrative.title
            if len(title) > 90:
                title = title[:87] + "..."
            pdf.cell(160, 8, title, ln=True)
            
            pdf.cell(30, 8, "Status:", 0)
            pdf.cell(160, 8, narrative.status, ln=True)
            pdf.cell(30, 8, "Detected:", 0)
            pdf.cell(160, 8, narrative.first_detected.strftime('%Y-%m-%d'), ln=True)
            pdf.ln(5)
            
            # Complexity summary
            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 10, "Complexity Summary", ln=True)
            pdf.set_font("Arial", "", 12)
            
            overall_score = complexity_data.get('overall_complexity_score')
            pdf.cell(60, 8, "Overall Complexity Score:", 0)
            
            # Color code based on score
            if overall_score >= 7:
                pdf.set_text_color(192, 0, 0)  # Red for high
            elif overall_score >= 4:
                pdf.set_text_color(255, 128, 0)  # Orange for medium
            else:
                pdf.set_text_color(0, 128, 0)  # Green for low
                
            pdf.cell(130, 8, f"{overall_score}/10", ln=True)
            pdf.set_text_color(0, 0, 0)  # Reset to black
            
            # Summary text with multi-line support
            pdf.set_font("Arial", "B", 12)
            pdf.cell(190, 8, "Summary:", ln=True)
            pdf.set_font("Arial", "", 12)
            
            summary = complexity_data.get('summary', 'N/A')
            pdf.multi_cell(190, 8, summary)
            pdf.ln(5)
            
            # Potential impact
            if complexity_data.get('potential_impact'):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(190, 8, "Potential Impact:", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(190, 8, complexity_data.get('potential_impact'))
                pdf.ln(5)
            
            # Detailed dimensions
            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 10, "Complexity Dimensions", ln=True)
            
            dimensions = [
                ('Linguistic Complexity', 'linguistic_complexity', (0, 0, 255)),  # Blue
                ('Logical Structure', 'logical_structure', (0, 128, 0)),  # Green
                ('Rhetorical Techniques', 'rhetorical_techniques', (255, 128, 0)),  # Orange
                ('Emotional Manipulation', 'emotional_manipulation', (192, 0, 0))  # Red
            ]
            
            for label, key, color in dimensions:
                dim_data = complexity_data.get(key, {})
                score = dim_data.get('score', 'N/A')
                
                # Dimension heading
                pdf.set_font("Arial", "B", 12)
                pdf.set_text_color(color[0], color[1], color[2])
                pdf.cell(140, 8, label, 0)
                pdf.cell(50, 8, f"Score: {score}/10", ln=True)
                pdf.set_text_color(0, 0, 0)  # Reset to black
                
                # Observations
                pdf.set_font("Arial", "", 12)
                observations = dim_data.get('observations', 'No observations available.')
                pdf.multi_cell(190, 8, observations)
                
                # Examples (if available)
                if dim_data.get('examples'):
                    pdf.set_font("Arial", "I", 12)
                    pdf.multi_cell(190, 8, f"Examples: {dim_data.get('examples')}")
                
                pdf.ln(5)
            
            # Export to BytesIO
            pdf_output = BytesIO()
            pdf_output.write(pdf.output(dest='S').encode('latin1'))
            pdf_output.seek(0)
            
            return pdf_output
            
        except Exception as e:
            logger.error(f"Error exporting narrative {narrative_id} to PDF: {e}")
            return None
    
    @staticmethod
    def export_comparison_to_pdf(narrative_ids: List[int]) -> Optional[BytesIO]:
        """
        Export comparison of multiple narratives to PDF.
        
        Args:
            narrative_ids: List of narrative IDs to compare
            
        Returns:
            PDF data as BytesIO object, or None if export fails
        """
        try:
            if not narrative_ids or len(narrative_ids) < 2:
                logger.warning("At least two narratives are required for comparison")
                return None
            
            # Get narrative data
            narratives = []
            for narrative_id in narrative_ids:
                narrative = DetectedNarrative.query.get(narrative_id)
                if not narrative or not narrative.meta_data:
                    continue
                    
                meta_data = json.loads(narrative.meta_data)
                complexity_data = meta_data.get('complexity_analysis', {})
                
                if complexity_data and 'overall_complexity_score' in complexity_data:
                    narratives.append({
                        'id': narrative.id,
                        'title': narrative.title,
                        'status': narrative.status,
                        'first_detected': narrative.first_detected,
                        'complexity_data': complexity_data
                    })
            
            if len(narratives) < 2:
                logger.warning("Not enough narratives with complexity data for comparison")
                return None
            
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(190, 10, "Narrative Complexity Comparison Report", ln=True, align="C")
            pdf.set_font("Arial", "", 10)
            pdf.cell(190, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(5)
            
            # Narrative information table
            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 10, "Narratives Being Compared", ln=True)
            
            # Table header
            pdf.set_font("Arial", "B", 10)
            pdf.cell(20, 8, "ID", 1)
            pdf.cell(120, 8, "Title", 1)
            pdf.cell(50, 8, "Overall Score", 1, ln=True)
            
            # Table rows
            pdf.set_font("Arial", "", 10)
            for narrative in narratives:
                title = narrative['title']
                if len(title) > 60:
                    title = title[:57] + "..."
                
                pdf.cell(20, 8, str(narrative['id']), 1)
                pdf.cell(120, 8, title, 1)
                
                # Color code the overall score
                overall_score = narrative['complexity_data'].get('overall_complexity_score')
                if overall_score >= 7:
                    pdf.set_text_color(192, 0, 0)  # Red for high
                elif overall_score >= 4:
                    pdf.set_text_color(255, 128, 0)  # Orange for medium
                else:
                    pdf.set_text_color(0, 128, 0)  # Green for low
                    
                pdf.cell(50, 8, f"{overall_score}/10", 1, ln=True)
                pdf.set_text_color(0, 0, 0)  # Reset to black
            
            pdf.ln(10)
            
            # Dimension comparison table
            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 10, "Dimension Comparison", ln=True)
            
            # Table header
            pdf.set_font("Arial", "B", 10)
            pdf.cell(60, 8, "Dimension", 1)
            
            # Add a column for each narrative
            for i, narrative in enumerate(narratives):
                pdf.cell(130/len(narratives), 8, f"Narrative {narrative['id']}", 1)
            pdf.ln()
            
            # Dimension rows
            dimensions = [
                ('Linguistic Complexity', 'linguistic_complexity'),
                ('Logical Structure', 'logical_structure'),
                ('Rhetorical Techniques', 'rhetorical_techniques'),
                ('Emotional Manipulation', 'emotional_manipulation')
            ]
            
            pdf.set_font("Arial", "", 10)
            for label, key in dimensions:
                pdf.cell(60, 8, label, 1)
                
                for narrative in narratives:
                    dim_data = narrative['complexity_data'].get(key, {})
                    score = dim_data.get('score', 'N/A')
                    
                    # Color code based on score
                    if score >= 7:
                        pdf.set_text_color(192, 0, 0)  # Red for high
                    elif score >= 4:
                        pdf.set_text_color(255, 128, 0)  # Orange for medium
                    else:
                        pdf.set_text_color(0, 128, 0)  # Green for low
                        
                    pdf.cell(130/len(narratives), 8, f"{score}/10", 1)
                    pdf.set_text_color(0, 0, 0)  # Reset to black
                
                pdf.ln()
            
            pdf.ln(10)
            
            # Comparison insights
            pdf.set_font("Arial", "B", 14)
            pdf.cell(190, 10, "Comparison Insights", ln=True)
            pdf.set_font("Arial", "", 12)
            
            # Generate simple insights
            insights = generate_comparison_insights(narratives)
            pdf.multi_cell(190, 8, insights)
            
            # Export to BytesIO
            pdf_output = BytesIO()
            pdf_output.write(pdf.output(dest='S').encode('latin1'))
            pdf_output.seek(0)
            
            return pdf_output
            
        except Exception as e:
            logger.error(f"Error exporting comparison to PDF: {e}")
            return None

    @staticmethod
    def export_trends_to_csv(days: int = 30) -> Optional[BytesIO]:
        """
        Export complexity trends over time to CSV.
        
        Args:
            days: Number of days to include in the analysis
            
        Returns:
            CSV data as BytesIO object, or None if export fails
        """
        try:
            # Calculate time cutoff
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Query narratives created/updated within the time period
            narratives = DetectedNarrative.query.filter(
                DetectedNarrative.last_updated >= cutoff_date
            ).order_by(DetectedNarrative.last_updated.asc()).all()
            
            # Process the data
            data = []
            for narrative in narratives:
                if not narrative.meta_data:
                    continue
                    
                try:
                    metadata = json.loads(narrative.meta_data)
                    complexity_data = metadata.get('complexity_analysis', {})
                    
                    if not complexity_data or 'overall_complexity_score' not in complexity_data:
                        continue
                    
                    # Add complexity scores
                    data.append({
                        'narrative_id': narrative.id,
                        'title': narrative.title,
                        'last_updated': narrative.last_updated.strftime('%Y-%m-%d'),
                        'overall_complexity': complexity_data.get('overall_complexity_score', 0),
                        'linguistic_complexity': complexity_data.get('linguistic_complexity', {}).get('score', 0),
                        'logical_structure': complexity_data.get('logical_structure', {}).get('score', 0),
                        'rhetorical_techniques': complexity_data.get('rhetorical_techniques', {}).get('score', 0),
                        'emotional_manipulation': complexity_data.get('emotional_manipulation', {}).get('score', 0)
                    })
                    
                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue
            
            if not data:
                logger.warning("No complexity data available for trends export")
                return None
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            # Create CSV file
            csv_output = StringIO()
            df.to_csv(csv_output, index=False)
            
            # Convert to BytesIO
            csv_bytes = BytesIO(csv_output.getvalue().encode('utf-8'))
            csv_bytes.seek(0)
            
            return csv_bytes
            
        except Exception as e:
            logger.error(f"Error exporting trends to CSV: {e}")
            return None

def generate_comparison_insights(narratives: List[Dict[str, Any]]) -> str:
    """
    Generate insights from narrative comparison.
    
    Args:
        narratives: List of narrative data dictionaries
        
    Returns:
        String with comparison insights
    """
    try:
        # Extract scores
        scores = []
        for narrative in narratives:
            complexity_data = narrative['complexity_data']
            scores.append({
                'id': narrative['id'],
                'overall': complexity_data.get('overall_complexity_score'),
                'linguistic': complexity_data.get('linguistic_complexity', {}).get('score'),
                'logical': complexity_data.get('logical_structure', {}).get('score'),
                'rhetorical': complexity_data.get('rhetorical_techniques', {}).get('score'),
                'emotional': complexity_data.get('emotional_manipulation', {}).get('score')
            })
        
        # Generate insights
        insights = []
        
        # Compare overall complexity
        max_overall = max(scores, key=lambda x: x['overall'])
        min_overall = min(scores, key=lambda x: x['overall'])
        
        if max_overall['overall'] - min_overall['overall'] >= 2:
            insights.append(
                f"Narrative {max_overall['id']} has significantly higher overall complexity "
                f"({max_overall['overall']}/10) compared to Narrative {min_overall['id']} "
                f"({min_overall['overall']}/10), suggesting more sophisticated information techniques."
            )
        else:
            insights.append(
                f"Narratives show similar overall complexity scores, suggesting comparable levels of sophistication."
            )
        
        # Identify dominant dimensions for each narrative
        for score in scores:
            dimensions = [
                ('linguistic', score['linguistic']),
                ('logical', score['logical']),
                ('rhetorical', score['rhetorical']),
                ('emotional', score['emotional'])
            ]
            
            # Sort dimensions by score
            dimensions.sort(key=lambda x: x[1], reverse=True)
            top_dimension = dimensions[0][0]
            
            # Map dimension to readable name
            dimension_names = {
                'linguistic': 'linguistic complexity',
                'logical': 'logical structure',
                'rhetorical': 'rhetorical techniques',
                'emotional': 'emotional manipulation'
            }
            
            insights.append(
                f"Narrative {score['id']} is strongest in {dimension_names[top_dimension]} "
                f"({dimensions[0][1]}/10), indicating a focus on this dimension."
            )
        
        # Check for similar patterns
        similar_patterns = False
        narrative_pairs = [(a, b) for a in scores for b in scores if a['id'] != b['id']]
        
        for a, b in narrative_pairs:
            # Check if dimension rankings are the same
            a_dims = [
                ('linguistic', a['linguistic']),
                ('logical', a['logical']),
                ('rhetorical', a['rhetorical']),
                ('emotional', a['emotional'])
            ]
            
            b_dims = [
                ('linguistic', b['linguistic']),
                ('logical', b['logical']),
                ('rhetorical', b['rhetorical']),
                ('emotional', b['emotional'])
            ]
            
            a_dims.sort(key=lambda x: x[1], reverse=True)
            b_dims.sort(key=lambda x: x[1], reverse=True)
            
            if [d[0] for d in a_dims] == [d[0] for d in b_dims]:
                similar_patterns = True
                insights.append(
                    f"Narratives {a['id']} and {b['id']} show similar complexity patterns, "
                    f"with the same ranking of dimensions, suggesting potential coordination or similar origins."
                )
                break
        
        if not similar_patterns:
            insights.append(
                "The narratives show distinct complexity patterns, indicating different approaches or origins."
            )
        
        return "\n\n".join(insights)
        
    except Exception as e:
        logger.error(f"Error generating comparison insights: {e}")
        return "Unable to generate comparison insights due to an error."