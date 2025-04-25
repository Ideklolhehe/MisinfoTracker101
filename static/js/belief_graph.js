/**
 * CIVILIAN Belief Graph Visualization
 * Renders and manages interactive visualizations of the belief graph
 */

class BeliefGraphViz {
    constructor(elementId, options = {}) {
        this.container = document.getElementById(elementId);
        if (!this.container) {
            console.error(`Element with ID "${elementId}" not found`);
            return;
        }
        
        // Default options
        this.options = {
            width: this.container.offsetWidth,
            height: 600,
            nodeSize: 10,
            linkDistance: 100,
            chargeStrength: -300,
            ...options
        };
        
        // Initialize D3 force simulation
        this.initializeSimulation();
        
        // Create SVG container
        this.svg = d3.select(this.container)
            .append("svg")
            .attr("width", this.options.width)
            .attr("height", this.options.height)
            .classed("belief-graph-svg", true);
            
        // Add zoom behavior
        this.svg.call(d3.zoom()
            .extent([[0, 0], [this.options.width, this.options.height]])
            .scaleExtent([0.25, 4])
            .on("zoom", (event) => {
                this.svgGroup.attr("transform", event.transform);
            }));
            
        // Create a group for the graph elements
        this.svgGroup = this.svg.append("g");
        
        // Initialize graph data structures
        this.nodes = [];
        this.links = [];
        this.nodeElements = null;
        this.linkElements = null;
        this.textElements = null;
        
        // Node color scale based on node type
        this.nodeColorScale = d3.scaleOrdinal()
            .domain(["narrative", "claim", "entity", "source", "instance"])
            .range(["#ff7675", "#74b9ff", "#55efc4", "#fdcb6e", "#a29bfe"]);
            
        // Add legend
        this.addLegend();
    }
    
    initializeSimulation() {
        // Create force simulation
        this.simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(this.options.linkDistance))
            .force("charge", d3.forceManyBody().strength(this.options.chargeStrength))
            .force("center", d3.forceCenter(this.options.width / 2, this.options.height / 2))
            .force("collision", d3.forceCollide(this.options.nodeSize * 1.5));
    }
    
    addLegend() {
        const legend = this.svg.append("g")
            .attr("class", "legend")
            .attr("transform", "translate(20, 20)");
            
        const types = ["narrative", "claim", "entity", "source", "instance"];
        const legendPadding = 5;
        const legendRectSize = 15;
        const legendSpacing = 4;
        
        const legendItems = legend.selectAll(".legend-item")
            .data(types)
            .enter()
            .append("g")
            .attr("class", "legend-item")
            .attr("transform", (d, i) => `translate(0, ${i * (legendRectSize + legendSpacing)})`);
            
        legendItems.append("rect")
            .attr("width", legendRectSize)
            .attr("height", legendRectSize)
            .style("fill", d => this.nodeColorScale(d))
            .style("stroke", "#fff");
            
        legendItems.append("text")
            .attr("x", legendRectSize + legendSpacing)
            .attr("y", legendRectSize - legendPadding)
            .text(d => d.charAt(0).toUpperCase() + d.slice(1)) // Capitalize
            .style("fill", "#e2e2e2")
            .style("font-size", "12px");
    }
    
    updateGraph(graphData) {
        if (!graphData || !graphData.connections) {
            console.error("Invalid graph data structure");
            return;
        }
        
        // Update nodes and links
        this.nodes = graphData.connections.nodes || [];
        this.links = graphData.connections.edges || [];
        
        // Update visualization
        this.renderGraph();
    }
    
    renderGraph() {
        // Clear previous elements
        this.svgGroup.selectAll("*").remove();
        
        // Create links
        this.linkElements = this.svgGroup.append("g")
            .selectAll("line")
            .data(this.links)
            .enter()
            .append("line")
            .attr("class", "belief-graph-link")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", d => Math.sqrt(d.weight || 1));
            
        // Create nodes
        this.nodeElements = this.svgGroup.append("g")
            .selectAll("circle")
            .data(this.nodes)
            .enter()
            .append("circle")
            .attr("class", "belief-graph-node")
            .attr("r", d => {
                // Size based on node type
                if (d.type === "narrative") return this.options.nodeSize * 1.5;
                if (d.type === "instance") return this.options.nodeSize * 0.8;
                return this.options.nodeSize;
            })
            .attr("fill", d => this.nodeColorScale(d.type || "entity"))
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
            .call(this.dragBehavior())
            .on("click", (event, d) => this.handleNodeClick(d))
            .append("title")
            .text(d => this.getNodeTooltip(d));
            
        // Create node labels
        this.textElements = this.svgGroup.append("g")
            .selectAll("text")
            .data(this.nodes)
            .enter()
            .append("text")
            .text(d => this.truncateText(d.content, 25))
            .attr("font-size", 12)
            .attr("dx", 15)
            .attr("dy", 4)
            .attr("fill", "#e2e2e2");
            
        // Update simulation
        this.simulation
            .nodes(this.nodes)
            .on("tick", () => this.ticked());
            
        this.simulation.force("link")
            .links(this.links);
            
        // Restart simulation
        this.simulation.alpha(1).restart();
    }
    
    ticked() {
        // Update positions
        this.linkElements
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
            
        this.nodeElements
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
            
        this.textElements
            .attr("x", d => d.x)
            .attr("y", d => d.y);
    }
    
    dragBehavior() {
        return d3.drag()
            .on("start", (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on("drag", (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on("end", (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
    
    handleNodeClick(node) {
        // Trigger custom event with node data
        const event = new CustomEvent("node-clicked", { detail: node });
        this.container.dispatchEvent(event);
        
        // Highlight node and its connections
        this.highlightNode(node.id);
    }
    
    highlightNode(nodeId) {
        // Reset all nodes and links
        this.nodeElements.attr("opacity", 0.3);
        this.linkElements.attr("opacity", 0.1);
        this.textElements.attr("opacity", 0.3);
        
        // Find connected nodes
        const connectedNodes = new Set();
        connectedNodes.add(nodeId);
        
        this.links.forEach(link => {
            if (link.source.id === nodeId || link.target.id === nodeId) {
                connectedNodes.add(link.source.id);
                connectedNodes.add(link.target.id);
            }
        });
        
        // Highlight connected nodes and links
        this.nodeElements
            .filter(d => connectedNodes.has(d.id))
            .attr("opacity", 1);
            
        this.linkElements
            .filter(d => 
                connectedNodes.has(d.source.id) && connectedNodes.has(d.target.id))
            .attr("opacity", 1);
            
        this.textElements
            .filter(d => connectedNodes.has(d.id))
            .attr("opacity", 1);
    }
    
    resetHighlight() {
        this.nodeElements.attr("opacity", 1);
        this.linkElements.attr("opacity", 0.6);
        this.textElements.attr("opacity", 1);
    }
    
    getNodeTooltip(node) {
        let tooltip = `ID: ${node.id}\nType: ${node.type || "Unknown"}\n`;
        if (node.content) {
            tooltip += `Content: ${node.content}`;
        }
        return tooltip;
    }
    
    truncateText(text, maxLength) {
        if (!text) return "";
        return text.length > maxLength 
            ? text.substring(0, maxLength) + "..." 
            : text;
    }
    
    updateLayout() {
        // Update SVG dimensions on container resize
        const width = this.container.offsetWidth;
        this.svg.attr("width", width);
        
        // Update simulation
        this.simulation.force("center", d3.forceCenter(width / 2, this.options.height / 2));
        this.simulation.alpha(0.3).restart();
    }
    
    loadNodeConnections(nodeId, depth = 1) {
        fetch(`/api/graph/node/${nodeId}?depth=${depth}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error("Error loading node connections:", data.error);
                    return;
                }
                this.updateGraph(data);
            })
            .catch(error => {
                console.error("Failed to load node connections:", error);
            });
    }
    
    loadNarrativeGraph(narrativeId) {
        // First, find the belief node for this narrative
        fetch(`/api/narratives/${narrativeId}`)
            .then(response => response.json())
            .then(data => {
                // Search for a node with this narrative title
                if (data.title) {
                    fetch(`/api/graph/search?query=${encodeURIComponent(data.title)}&type=narrative`)
                        .then(response => response.json())
                        .then(nodes => {
                            if (nodes.length > 0) {
                                this.loadNodeConnections(nodes[0].id, 2);
                            } else {
                                console.error("No belief node found for this narrative");
                            }
                        });
                }
            })
            .catch(error => {
                console.error("Failed to load narrative graph:", error);
            });
    }
}

// Initialize graph when DOM is ready
document.addEventListener("DOMContentLoaded", function() {
    // Check if graph container exists
    const graphContainer = document.getElementById("belief-graph-container");
    if (graphContainer) {
        // Initialize the graph visualization
        window.beliefGraph = new BeliefGraphViz("belief-graph-container");
        
        // Handle window resize
        window.addEventListener("resize", () => {
            if (window.beliefGraph) {
                window.beliefGraph.updateLayout();
            }
        });
        
        // Load narrative graph if narrative ID is specified
        const narrativeId = graphContainer.dataset.narrativeId;
        if (narrativeId) {
            window.beliefGraph.loadNarrativeGraph(narrativeId);
        }
    }
});
