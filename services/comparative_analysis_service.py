"""
Service for comparative analysis of narratives and counter-narratives.
"""

import logging
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from sqlalchemy import desc, func
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import pearsonr
import networkx as nx

from app import db
from models import DetectedNarrative, NarrativeInstance, CounterMessage, DataSource
from utils.app_context import with_app_context
from utils.metrics import time_operation, Counter, Gauge

# Configure logger
logger = logging.getLogger(__name__)

# Prometheus metrics
comparative_counter = Counter('comparative_analyses_total', 'Total comparative analyses performed')
theme_detection_counter = Counter('theme_detection_total', 'Total theme detection operations')
correlation_gauge = Gauge('narrative_correlation_strength', 'Correlation strength between narratives')


class ComparativeAnalysisService:
    """Service for comparative analysis of narratives."""
    
    def __init__(self):
        """Initialize the comparative analysis service."""
        logger.info("Comparative analysis service initialized")
    
    @with_app_context
    def side_by_side_comparison(self, narrative_ids: List[int], dimensions: List[str] = None) -> Dict[str, Any]:
        """Perform side-by-side comparison of narratives across dimensions."""
        if not dimensions:
            dimensions = ['complexity_score', 'propagation_score', 'threat_score']
        
        # Get narratives data
        narratives = []
        for nid in narrative_ids:
            narrative = DetectedNarrative.query.get(nid)
            if narrative:
                narratives.append(narrative)
        
        if not narratives:
            return {'error': 'No valid narratives found for comparison'}
        
        # Build data for comparison
        data = {dim: [] for dim in dimensions}
        data['narrative'] = []
        
        for narrative in narratives:
            data['narrative'].append(narrative.title[:30])  # Truncate for display
            meta_data = narrative.get_meta_data()
            
            for dim in dimensions:
                # Get value from meta_data or use a default
                value = meta_data.get(dim, 0)
                data[dim].append(value)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(data)
        
        # Generate a bar chart using Plotly
        fig = go.Figure()
        for dim in dimensions:
            fig.add_trace(go.Bar(name=dim, x=df['narrative'], y=df[dim]))
        
        fig.update_layout(
            title='Side-by-Side Dimension Comparison',
            xaxis_title='Narratives',
            yaxis_title='Scores',
            barmode='group',
            height=600,
            width=800,
            margin=dict(l=50, r=50, t=80, b=100)
        )
        
        # Increment counter
        comparative_counter.inc()
        
        # Return both the plot HTML and the raw data
        return {
            'chart_html': plot(fig, output_type='div', include_plotlyjs=False),
            'data': data
        }
    
    @with_app_context
    def relative_growth_rate(self, narrative_id: int, days: int = 30) -> Dict[str, Any]:
        """Calculate and visualize the relative growth rate for a narrative."""
        # Get the narrative
        narrative = DetectedNarrative.query.get(narrative_id)
        if not narrative:
            return {'error': 'Narrative not found'}
        
        # Get instances for the narrative sorted by detection time
        instances = NarrativeInstance.query.filter_by(
            narrative_id=narrative_id
        ).order_by(
            NarrativeInstance.detected_at
        ).all()
        
        if not instances:
            return {'error': 'No instances found for narrative'}
        
        # Extract dates and calculate cumulative count
        dates = [instance.detected_at for instance in instances]
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame({
            'date': dates,
            'count': [1] * len(dates)  # Each instance counts as 1
        })
        
        # Resample to daily frequency and calculate cumulative count
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        daily = df.resample('D').sum().fillna(0)
        daily['cumulative'] = daily['count'].cumsum()
        
        # Calculate daily growth rate
        daily['growth_rate'] = daily['cumulative'].pct_change().fillna(0)
        
        # Limit to last N days
        if len(daily) > days:
            daily = daily.iloc[-days:]
        
        # Generate a line chart using Plotly
        fig = go.Figure()
        
        # Growth rate line
        fig.add_trace(go.Scatter(
            x=daily.index,
            y=daily['growth_rate'],
            mode='lines+markers',
            name='Growth Rate',
            line=dict(color='red')
        ))
        
        # Cumulative count line (secondary axis)
        fig.add_trace(go.Scatter(
            x=daily.index,
            y=daily['cumulative'],
            mode='lines',
            name='Cumulative Instances',
            line=dict(color='blue'),
            yaxis='y2'
        ))
        
        # Update layout for dual Y-axis
        fig.update_layout(
            title=f'Growth Rate for: {narrative.title[:50]}',
            xaxis_title='Date',
            yaxis=dict(
                title='Daily Growth Rate',
                titlefont=dict(color='red'),
                tickfont=dict(color='red')
            ),
            yaxis2=dict(
                title='Cumulative Instances',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.02, y=0.98),
            height=500,
            width=800,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Return both the plot HTML and the raw data
        return {
            'chart_html': plot(fig, output_type='div', include_plotlyjs=False),
            'data': {
                'dates': [d.strftime('%Y-%m-%d') for d in daily.index],
                'growth_rates': daily['growth_rate'].tolist(),
                'cumulative_counts': daily['cumulative'].tolist()
            }
        }
    
    @with_app_context
    def identify_correlation(self, narrative_id_1: int, narrative_id_2: int) -> Dict[str, Any]:
        """Identify correlation between two narratives over time."""
        # Get the narratives
        narrative1 = DetectedNarrative.query.get(narrative_id_1)
        narrative2 = DetectedNarrative.query.get(narrative_id_2)
        
        if not narrative1 or not narrative2:
            return {'error': 'One or both narratives not found'}
        
        # Get instances for both narratives
        instances1 = NarrativeInstance.query.filter_by(narrative_id=narrative_id_1).all()
        instances2 = NarrativeInstance.query.filter_by(narrative_id=narrative_id_2).all()
        
        if not instances1 or not instances2:
            return {'error': 'Insufficient instances for correlation analysis'}
        
        # Extract dates for both narratives
        dates1 = [instance.detected_at for instance in instances1]
        dates2 = [instance.detected_at for instance in instances2]
        
        # Convert to DataFrames
        df1 = pd.DataFrame({
            'date': dates1,
            'count': [1] * len(dates1)
        })
        df2 = pd.DataFrame({
            'date': dates2,
            'count': [1] * len(dates2)
        })
        
        # Resample to daily frequency
        df1['date'] = pd.to_datetime(df1['date'])
        df2['date'] = pd.to_datetime(df2['date'])
        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        
        # Resample and calculate daily counts
        daily1 = df1.resample('D').sum().fillna(0)
        daily2 = df2.resample('D').sum().fillna(0)
        
        # Determine the date range that includes both narratives
        start_date = max(daily1.index.min(), daily2.index.min())
        end_date = min(daily1.index.max(), daily2.index.max())
        
        # Filter to the common date range
        daily1 = daily1.loc[start_date:end_date]
        daily2 = daily2.loc[start_date:end_date]
        
        # Ensure both series have the same dates
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        daily1 = daily1.reindex(all_dates).fillna(0)
        daily2 = daily2.reindex(all_dates).fillna(0)
        
        # Calculate Pearson correlation
        corr_coef, p_value = pearsonr(daily1['count'], daily2['count'])
        
        # Update metric
        correlation_gauge.labels(
            narrative1_id=narrative_id_1, 
            narrative2_id=narrative_id_2
        ).set(corr_coef)
        
        # Generate a scatter plot with trend line
        fig = go.Figure()
        
        # Scatter plot of daily counts
        fig.add_trace(go.Scatter(
            x=daily1['count'],
            y=daily2['count'],
            mode='markers',
            name='Daily Instances',
            marker=dict(size=8)
        ))
        
        # Add trend line if there are enough data points
        if len(daily1) > 2:
            # Calculate linear regression
            from sklearn.linear_model import LinearRegression
            X = daily1['count'].values.reshape(-1, 1)
            y = daily2['count'].values
            reg = LinearRegression().fit(X, y)
            
            # Create trend line
            x_range = np.linspace(daily1['count'].min(), daily1['count'].max(), 100)
            y_pred = reg.predict(x_range.reshape(-1, 1))
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name=f'Trend (r={corr_coef:.2f})',
                line=dict(color='red')
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Correlation Analysis: {narrative1.title[:30]} vs {narrative2.title[:30]}',
            xaxis_title=f'{narrative1.title[:20]} Daily Instances',
            yaxis_title=f'{narrative2.title[:20]} Daily Instances',
            height=600,
            width=800,
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref='paper',
                    yref='paper',
                    text=f'Pearson r: {corr_coef:.2f}, p-value: {p_value:.4f}',
                    showarrow=False,
                    font=dict(size=14)
                )
            ],
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Return both the plot HTML and the raw data
        return {
            'chart_html': plot(fig, output_type='div', include_plotlyjs=False),
            'correlation': corr_coef,
            'p_value': p_value,
            'significance': 'significant' if p_value < 0.05 else 'not significant',
            'data': {
                'dates': [d.strftime('%Y-%m-%d') for d in daily1.index],
                'narrative1_counts': daily1['count'].tolist(),
                'narrative2_counts': daily2['count'].tolist()
            }
        }
    
    @with_app_context
    def shared_theme_detection(self, narrative_ids: List[int], n_topics: int = 5) -> Dict[str, Any]:
        """Detect shared themes across narratives using Latent Dirichlet Allocation."""
        # Get the narratives
        narratives = []
        for nid in narrative_ids:
            narrative = DetectedNarrative.query.get(nid)
            if narrative:
                narratives.append(narrative)
        
        if len(narratives) < 2:
            return {'error': 'At least two valid narratives are required for theme detection'}
        
        # Get instances for each narrative
        all_texts = []
        narrative_titles = []
        
        for narrative in narratives:
            instances = NarrativeInstance.query.filter_by(narrative_id=narrative.id).all()
            if instances:
                # Combine instance content for this narrative
                text = " ".join([instance.content for instance in instances if instance.content])
                all_texts.append(text)
                narrative_titles.append(narrative.title[:30])
            else:
                # Use the narrative description if no instances
                text = narrative.description or ""
                if text:
                    all_texts.append(text)
                    narrative_titles.append(narrative.title[:30])
        
        if len(all_texts) < 2:
            return {'error': 'Insufficient text content for theme detection'}
        
        # Increment counter
        theme_detection_counter.inc()
        
        try:
            # Create document-term matrix
            vectorizer = CountVectorizer(
                stop_words='english',
                max_df=0.9,  # Ignore terms that appear in >90% of documents
                min_df=2,    # Ignore terms that appear in fewer than 2 documents
                max_features=10000
            )
            X = vectorizer.fit_transform(all_texts)
            
            # Ensure we have enough features
            if X.shape[1] < 10:
                return {'error': 'Insufficient vocabulary for theme detection'}
            
            # Adjust number of topics if we have fewer documents than requested topics
            n_topics = min(n_topics, len(all_texts))
            
            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
            lda.fit(X)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                # Get top terms for this topic
                top_indices = topic.argsort()[:-11:-1]  # Top 10 terms
                top_terms = [feature_names[i] for i in top_indices]
                
                topics.append({
                    'id': topic_idx,
                    'top_terms': top_terms,
                    'weight': float(topic.sum())
                })
            
            # Get topic distribution for each narrative
            topic_distributions = lda.transform(X)
            
            # Create a heatmap of narrative-topic relationships
            fig = go.Figure(data=go.Heatmap(
                z=topic_distributions,
                x=[f'Topic {i+1}' for i in range(n_topics)],
                y=narrative_titles,
                colorscale='Viridis',
                colorbar=dict(title='Topic Weight')
            ))
            
            fig.update_layout(
                title='Narrative-Topic Distribution Heatmap',
                xaxis_title='Topics',
                yaxis_title='Narratives',
                height=600,
                width=800,
                margin=dict(l=100, r=50, t=80, b=50)
            )
            
            # Return both the plot HTML and the topic data
            return {
                'chart_html': plot(fig, output_type='div', include_plotlyjs=False),
                'topics': topics,
                'narrative_topic_distribution': topic_distributions.tolist(),
                'narratives': narrative_titles
            }
        
        except Exception as e:
            logger.error(f"Error in shared theme detection: {e}")
            return {'error': f'Theme detection failed: {str(e)}'}
    
    @with_app_context
    def coordinated_source_analysis(
        self, 
        narrative_ids: Optional[List[int]] = None, 
        edges: Optional[List[List[int]]] = None
    ) -> Dict[str, Any]:
        """Analyze source coordination using network analysis."""
        # Create directed graph
        G = nx.DiGraph()
        
        if edges:
            # Use provided edges
            for edge in edges:
                if len(edge) >= 2:
                    G.add_edge(edge[0], edge[1])
        
        elif narrative_ids:
            # Build graph from narrative instances
            for narrative_id in narrative_ids:
                instances = NarrativeInstance.query.filter_by(narrative_id=narrative_id).all()
                
                for instance in instances:
                    if instance.source_id:
                        # Add edge from source to narrative
                        G.add_edge(f"source_{instance.source_id}", f"narrative_{narrative_id}")
                        
                        # Add metadata if available
                        meta = instance.get_meta_data()
                        references = meta.get('references', [])
                        
                        for ref in references:
                            # Add edge from referenced source to this source
                            G.add_edge(f"source_{ref}", f"source_{instance.source_id}")
        
        else:
            # Use all sources and their relationships
            sources = DataSource.query.filter_by(is_active=True).all()
            
            for source in sources:
                G.add_node(f"source_{source.id}", 
                           type='source', 
                           name=source.name)
                
                # Add edges based on source metadata
                meta = source.get_meta_data()
                related = meta.get('related_sources', [])
                
                for related_id in related:
                    G.add_edge(f"source_{source.id}", f"source_{related_id}")
        
        # Check if we have enough nodes for analysis
        if len(G.nodes) < 3:
            return {'error': 'Insufficient nodes for network analysis'}
        
        try:
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Calculate eigenvector centrality if possible
            try:
                eigenvector_centrality = nx.eigenvector_centrality_numpy(G, max_iter=1000)
            except:
                eigenvector_centrality = {node: 0 for node in G.nodes}
            
            # Combine centrality scores
            combined_scores = {}
            for node in G.nodes:
                combined_scores[node] = (
                    degree_centrality.get(node, 0) +
                    betweenness_centrality.get(node, 0) +
                    eigenvector_centrality.get(node, 0)
                )
            
            # Get top sources by centrality
            top_sources = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            bottom_sources = sorted(combined_scores.items(), key=lambda x: x[1])[:5]
            
            # Prepare node and edge data for visualization
            node_sizes = [combined_scores.get(node, 0.1) * 1000 + 10 for node in G.nodes]
            edge_weights = [G.get_edge_data(u, v).get('weight', 1) for u, v in G.edges]
            
            # Create network visualization using Plotly
            pos = nx.spring_layout(G, seed=42)
            
            # Create edges
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            # Create nodes
            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=[],
                    color=[],
                    colorbar=dict(
                        thickness=15,
                        title='Node Centrality',
                        xanchor='left',
                        titleside='right'
                    ),
                    line=dict(width=2)
                )
            )
            
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
            
            # Add node attributes
            for node in G.nodes():
                node_trace['marker']['size'] += (combined_scores.get(node, 0.1) * 30 + 10,)
                node_trace['marker']['color'] += (combined_scores.get(node, 0),)
                node_info = f"Node: {node}<br>Centrality: {combined_scores.get(node, 0):.4f}"
                node_trace['text'] += (node_info,)
            
            # Create the figure
            fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Source Coordination Network Analysis',
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            width=800,
                            height=600
                        ))
            
            # Return network analysis results
            return {
                'chart_html': plot(fig, output_type='div', include_plotlyjs=False),
                'top_sources': [{'id': node, 'centrality': score} for node, score in top_sources],
                'bottom_sources': [{'id': node, 'centrality': score} for node, score in bottom_sources],
                'node_count': len(G.nodes),
                'edge_count': len(G.edges),
                'density': nx.density(G)
            }
        
        except Exception as e:
            logger.error(f"Error in coordinated source analysis: {e}")
            return {'error': f'Network analysis failed: {str(e)}'}