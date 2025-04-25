/**
 * CIVILIAN Dashboard JavaScript
 * Handles interactive dashboard features including charts, data loading,
 * and user interaction.
 */

// Initialize dashboard when the DOM is loaded
document.addEventListener("DOMContentLoaded", function() {
    // Initialize charts if containers exist
    initCharts();
    
    // Set up interactive elements
    setupEventListeners();
    
    // Initialize datatables for tables
    setupDataTables();
    
    // Set up counter message approval functionality
    setupCounterMessageApproval();
    
    // Initialize narrative detail view if on detail page
    if (document.querySelector('.narrative-detail')) {
        initNarrativeDetail();
    }
    
    // Initialize real-time updates
    initRealTimeUpdates();
});

/**
 * Initialize dashboard charts
 */
function initCharts() {
    // Narratives Over Time Chart
    const narrativesChartContainer = document.getElementById('narratives-chart');
    if (narrativesChartContainer) {
        // Fetch data for chart
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                // Create chart
                const ctx = narrativesChartContainer.getContext('2d');
                
                // In a real application, we would fetch time-series data
                // For the proof of concept, we'll use placeholder data
                const labels = getDaysArray(7);
                
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'New Narratives',
                            data: [5, 8, 12, 7, 10, 15, 9],
                            borderColor: '#ff7675',
                            backgroundColor: 'rgba(255, 118, 117, 0.1)',
                            tension: 0.3,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'top',
                                labels: {
                                    color: '#e2e2e2'
                                }
                            },
                            title: {
                                display: true,
                                text: 'Narratives Detected (Last 7 Days)',
                                color: '#e2e2e2'
                            }
                        },
                        scales: {
                            x: {
                                grid: {
                                    color: 'rgba(255, 255, 255, 0.1)'
                                },
                                ticks: {
                                    color: '#e2e2e2'
                                }
                            },
                            y: {
                                beginAtZero: true,
                                grid: {
                                    color: 'rgba(255, 255, 255, 0.1)'
                                },
                                ticks: {
                                    color: '#e2e2e2'
                                }
                            }
                        }
                    }
                });
            })
            .catch(error => {
                console.error('Error fetching chart data:', error);
                narrativesChartContainer.innerHTML = '<div class="alert alert-danger">Failed to load chart data</div>';
            });
    }
    
    // Threat Distribution Chart
    const threatChartContainer = document.getElementById('threat-distribution-chart');
    if (threatChartContainer) {
        // Fetch data for chart
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                // Create chart
                const ctx = threatChartContainer.getContext('2d');
                
                new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Low Threat', 'Medium Threat', 'High Threat'],
                        datasets: [{
                            data: [60, 30, 10],
                            backgroundColor: [
                                '#55efc4',
                                '#fdcb6e',
                                '#ff7675'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'top',
                                labels: {
                                    color: '#e2e2e2'
                                }
                            },
                            title: {
                                display: true,
                                text: 'Threat Level Distribution',
                                color: '#e2e2e2'
                            }
                        }
                    }
                });
            })
            .catch(error => {
                console.error('Error fetching chart data:', error);
                threatChartContainer.innerHTML = '<div class="alert alert-danger">Failed to load chart data</div>';
            });
    }
    
    // Source Distribution Chart
    const sourceChartContainer = document.getElementById('source-distribution-chart');
    if (sourceChartContainer) {
        // Create chart
        const ctx = sourceChartContainer.getContext('2d');
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Twitter', 'Telegram', 'RSS', 'Other'],
                datasets: [{
                    label: 'Instances by Source',
                    data: [45, 30, 15, 10],
                    backgroundColor: [
                        '#74b9ff',
                        '#a29bfe',
                        '#ffeaa7',
                        '#81ecec'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Content Sources Distribution',
                        color: '#e2e2e2'
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#e2e2e2'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#e2e2e2'
                        }
                    }
                }
            }
        });
    }
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // Toggle counter message form
    const generateBtn = document.getElementById('generate-counter-btn');
    if (generateBtn) {
        generateBtn.addEventListener('click', function() {
            document.getElementById('counter-message-form').classList.toggle('d-none');
        });
    }
    
    // Show/hide advanced filters
    const advancedFiltersToggle = document.getElementById('toggle-advanced-filters');
    if (advancedFiltersToggle) {
        advancedFiltersToggle.addEventListener('click', function(e) {
            e.preventDefault();
            document.getElementById('advanced-filters').classList.toggle('d-none');
            this.textContent = this.textContent.includes('Show') 
                ? 'Hide Advanced Filters' 
                : 'Show Advanced Filters';
        });
    }
    
    // Narrative analysis trigger
    const analyzeButtons = document.querySelectorAll('.analyze-narrative-btn');
    analyzeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const narrativeId = this.dataset.narrativeId;
            analyzeNarrative(narrativeId, this);
        });
    });
    
    // Generate counter message button
    const counterMessageButtons = document.querySelectorAll('.generate-counter-btn');
    counterMessageButtons.forEach(button => {
        button.addEventListener('click', function() {
            const narrativeId = this.dataset.narrativeId;
            generateCounterMessage(narrativeId, this);
        });
    });
    
    // Filter form submission
    const filterForm = document.getElementById('filter-form');
    if (filterForm) {
        filterForm.addEventListener('submit', function(e) {
            // Remove empty fields before submitting
            const inputs = this.querySelectorAll('input, select');
            inputs.forEach(input => {
                if (input.value === '' || input.value === 'all') {
                    input.disabled = true;
                }
            });
        });
    }
}

/**
 * Setup DataTables for better table interaction
 */
function setupDataTables() {
    // Initialize DataTables if library is loaded
    if (typeof $.fn.DataTable !== 'undefined') {
        $('.datatable').DataTable({
            pageLength: 10,
            responsive: true,
            dom: 'Bfrtip',
            buttons: ['copy', 'csv', 'excel'],
            order: [[0, 'desc']]
        });
    }
}

/**
 * Setup counter message approval functionality
 */
function setupCounterMessageApproval() {
    const approvalButtons = document.querySelectorAll('.approve-counter-btn');
    approvalButtons.forEach(button => {
        button.addEventListener('click', function() {
            const messageId = this.dataset.messageId;
            approveCounterMessage(messageId, this);
        });
    });
}

/**
 * Initialize narrative detail view
 */
function initNarrativeDetail() {
    // Load belief graph if container exists
    const graphContainer = document.getElementById('belief-graph-container');
    if (graphContainer) {
        const narrativeId = graphContainer.dataset.narrativeId;
        
        // The belief graph is initialized in belief_graph.js
        // Here we just need to trigger loading the narrative-specific data
        if (window.beliefGraph && narrativeId) {
            window.beliefGraph.loadNarrativeGraph(narrativeId);
        }
    }
    
    // Load instances if container exists
    const instancesContainer = document.getElementById('narrative-instances');
    if (instancesContainer) {
        const narrativeId = instancesContainer.dataset.narrativeId;
        
        if (narrativeId) {
            loadNarrativeInstances(narrativeId, instancesContainer);
        }
    }
    
    // Load counter messages if container exists
    const counterMessagesContainer = document.getElementById('counter-messages');
    if (counterMessagesContainer) {
        const narrativeId = counterMessagesContainer.dataset.narrativeId;
        
        if (narrativeId) {
            loadCounterMessages(narrativeId, counterMessagesContainer);
        }
    }
}

/**
 * Initialize real-time updates
 */
function initRealTimeUpdates() {
    // In a production system, this would use websockets or SSE
    // For the proof of concept, we'll simulate updates with polling
    
    const updateInterval = 60000; // 60 seconds
    
    // Update dashboard stats
    setInterval(function() {
        updateDashboardStats();
    }, updateInterval);
    
    // Update narrative list if on narratives page
    if (document.querySelector('.narratives-list')) {
        setInterval(function() {
            updateNarrativesList();
        }, updateInterval);
    }
}

/**
 * Analyze a narrative
 */
function analyzeNarrative(narrativeId, button) {
    // Show loading state
    const originalText = button.textContent;
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
    
    // Call API to analyze narrative
    fetch(`/api/narratives/${narrativeId}/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        // Reset button
        button.disabled = false;
        button.textContent = originalText;
        
        if (data.error) {
            showAlert('danger', `Analysis failed: ${data.error}`);
            return;
        }
        
        // Show success message
        showAlert('success', 'Narrative analysis completed successfully.');
        
        // Update UI with analysis results
        if (data.propagation_score) {
            const scoreElement = document.getElementById(`propagation-score-${narrativeId}`);
            if (scoreElement) {
                scoreElement.textContent = `${(data.propagation_score * 100).toFixed(1)}%`;
            }
        }
        
        if (data.viral_threat) {
            const threatElement = document.getElementById(`viral-threat-${narrativeId}`);
            if (threatElement) {
                threatElement.textContent = data.viral_threat;
                
                // Update threat badge color
                const threatBadge = document.getElementById(`threat-badge-${narrativeId}`);
                if (threatBadge) {
                    threatBadge.className = 'badge';
                    if (data.viral_threat >= 4) {
                        threatBadge.classList.add('bg-danger');
                    } else if (data.viral_threat >= 2) {
                        threatBadge.classList.add('bg-warning');
                    } else {
                        threatBadge.classList.add('bg-success');
                    }
                }
            }
        }
        
        // Reload page to show all updated data
        setTimeout(() => {
            location.reload();
        }, 2000);
    })
    .catch(error => {
        console.error('Error analyzing narrative:', error);
        button.disabled = false;
        button.textContent = originalText;
        showAlert('danger', 'Failed to analyze narrative. Please try again.');
    });
}

/**
 * Generate a counter message for a narrative
 */
function generateCounterMessage(narrativeId, button) {
    // Show loading state
    const originalText = button.textContent;
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
    
    // Call API to generate counter message
    fetch('/api/counter-messages/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ narrative_id: narrativeId })
    })
    .then(response => response.json())
    .then(data => {
        // Reset button
        button.disabled = false;
        button.textContent = originalText;
        
        if (data.error) {
            showAlert('danger', `Generation failed: ${data.error}`);
            return;
        }
        
        // Show success message
        showAlert('success', 'Counter message generated successfully.');
        
        // Add the new counter message to the UI
        const counterMessagesContainer = document.getElementById('counter-messages');
        if (counterMessagesContainer && data.content) {
            const messageHtml = `
                <div class="card mb-3 border-secondary">
                    <div class="card-header d-flex justify-content-between">
                        <span>Counter Message (Draft)</span>
                        <span class="badge bg-secondary">New</span>
                    </div>
                    <div class="card-body">
                        <p class="card-text">${data.content}</p>
                    </div>
                    <div class="card-footer d-flex justify-content-between">
                        <small class="text-muted">Strategy: ${data.strategy || 'factual_correction'}</small>
                        <button class="btn btn-sm btn-primary approve-counter-btn" data-message-id="${data.counter_id}">
                            Approve for Deployment
                        </button>
                    </div>
                </div>
            `;
            
            counterMessagesContainer.insertAdjacentHTML('afterbegin', messageHtml);
            
            // Set up approval button for the new message
            const newApproveButton = counterMessagesContainer.querySelector(`[data-message-id="${data.counter_id}"]`);
            if (newApproveButton) {
                newApproveButton.addEventListener('click', function() {
                    approveCounterMessage(data.counter_id, this);
                });
            }
        }
        
        // Hide the generate button since we now have a counter message
        button.style.display = 'none';
    })
    .catch(error => {
        console.error('Error generating counter message:', error);
        button.disabled = false;
        button.textContent = originalText;
        showAlert('danger', 'Failed to generate counter message. Please try again.');
    });
}

/**
 * Approve a counter message
 */
function approveCounterMessage(messageId, button) {
    // Show loading state
    const originalText = button.textContent;
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Approving...';
    
    // Call API to approve counter message
    fetch(`/api/counter-messages/${messageId}/approve`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        // Reset button
        button.disabled = false;
        
        if (data.error) {
            button.textContent = originalText;
            showAlert('danger', `Approval failed: ${data.error}`);
            return;
        }
        
        // Show success message
        showAlert('success', 'Counter message approved successfully.');
        
        // Update UI to reflect approved state
        button.textContent = 'Approved';
        button.classList.remove('btn-primary');
        button.classList.add('btn-success');
        button.disabled = true;
        
        // Update status badge if exists
        const messageCard = button.closest('.card');
        if (messageCard) {
            const statusBadge = messageCard.querySelector('.badge');
            if (statusBadge) {
                statusBadge.textContent = 'Approved';
                statusBadge.classList.remove('bg-secondary');
                statusBadge.classList.add('bg-success');
            }
            
            const headerText = messageCard.querySelector('.card-header span:first-child');
            if (headerText) {
                headerText.textContent = 'Counter Message (Approved)';
            }
        }
    })
    .catch(error => {
        console.error('Error approving counter message:', error);
        button.disabled = false;
        button.textContent = originalText;
        showAlert('danger', 'Failed to approve counter message. Please try again.');
    });
}

/**
 * Load narrative instances
 */
function loadNarrativeInstances(narrativeId, container) {
    fetch(`/api/narratives/${narrativeId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                return;
            }
            
            if (!data.instances || data.instances.length === 0) {
                container.innerHTML = '<div class="alert alert-info">No instances found for this narrative.</div>';
                return;
            }
            
            // Display instances
            let instancesHtml = '';
            data.instances.forEach(instance => {
                let sourceText = instance.metadata?.user_name ? 
                    `@${instance.metadata.user_name}` : 
                    instance.metadata?.chat_name || 'Unknown Source';
                    
                instancesHtml += `
                    <div class="card mb-3">
                        <div class="card-header d-flex justify-content-between">
                            <span>${sourceText}</span>
                            <small class="text-muted">${new Date(instance.detected_at).toLocaleString()}</small>
                        </div>
                        <div class="card-body">
                            <p class="card-text">${instance.content}</p>
                        </div>
                        ${instance.url ? `
                        <div class="card-footer text-end">
                            <a href="${instance.url}" target="_blank" class="btn btn-sm btn-outline-info">
                                View Original <i class="feather icon-external-link"></i>
                            </a>
                        </div>
                        ` : ''}
                    </div>
                `;
            });
            
            container.innerHTML = instancesHtml;
        })
        .catch(error => {
            console.error('Error loading narrative instances:', error);
            container.innerHTML = '<div class="alert alert-danger">Failed to load instances. Please try again.</div>';
        });
}

/**
 * Load counter messages
 */
function loadCounterMessages(narrativeId, container) {
    fetch(`/api/narratives/${narrativeId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                return;
            }
            
            if (!data.counter_messages || data.counter_messages.length === 0) {
                container.innerHTML = '<div class="alert alert-info">No counter messages found for this narrative.</div>';
                return;
            }
            
            // Display counter messages
            let messagesHtml = '';
            data.counter_messages.forEach(message => {
                let statusBadgeClass = 'bg-secondary';
                if (message.status === 'approved') {
                    statusBadgeClass = 'bg-success';
                } else if (message.status === 'deployed') {
                    statusBadgeClass = 'bg-primary';
                }
                
                messagesHtml += `
                    <div class="card mb-3 border-secondary">
                        <div class="card-header d-flex justify-content-between">
                            <span>Counter Message (${message.status.charAt(0).toUpperCase() + message.status.slice(1)})</span>
                            <span class="badge ${statusBadgeClass}">${message.status.charAt(0).toUpperCase() + message.status.slice(1)}</span>
                        </div>
                        <div class="card-body">
                            <p class="card-text">${message.content}</p>
                        </div>
                        <div class="card-footer d-flex justify-content-between">
                            <small class="text-muted">Strategy: ${message.strategy}</small>
                            ${message.status === 'draft' ? `
                            <button class="btn btn-sm btn-primary approve-counter-btn" data-message-id="${message.id}">
                                Approve for Deployment
                            </button>
                            ` : ''}
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = messagesHtml;
            
            // Set up approval buttons
            setupCounterMessageApproval();
        })
        .catch(error => {
            console.error('Error loading counter messages:', error);
            container.innerHTML = '<div class="alert alert-danger">Failed to load counter messages. Please try again.</div>';
        });
}

/**
 * Update dashboard stats
 */
function updateDashboardStats() {
    const statsContainer = document.getElementById('dashboard-stats');
    if (!statsContainer) return;
    
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            // Update narrative count
            const narrativeCountElement = document.getElementById('total-narratives-count');
            if (narrativeCountElement && data.narratives) {
                narrativeCountElement.textContent = data.narratives.total;
            }
            
            // Update active narratives count
            const activeNarrativesElement = document.getElementById('active-narratives-count');
            if (activeNarrativesElement && data.narratives) {
                activeNarrativesElement.textContent = data.narratives.active;
            }
            
            // Update instances count
            const instancesCountElement = document.getElementById('total-instances-count');
            if (instancesCountElement && data.instances) {
                instancesCountElement.textContent = data.instances.total;
            }
            
            // Update counter messages stats
            const pendingMessagesElement = document.getElementById('pending-messages-count');
            if (pendingMessagesElement && data.counter_messages) {
                pendingMessagesElement.textContent = data.counter_messages.draft;
            }
        })
        .catch(error => {
            console.error('Error updating dashboard stats:', error);
        });
}

/**
 * Update narratives list
 */
function updateNarrativesList() {
    const narrativesContainer = document.querySelector('.narratives-list');
    if (!narrativesContainer) return;
    
    // Get current filter parameters from the URL
    const urlParams = new URLSearchParams(window.location.search);
    const status = urlParams.get('status') || 'all';
    const language = urlParams.get('language') || 'all';
    const days = urlParams.get('days') || '30';
    const search = urlParams.get('search') || '';
    
    // Construct API URL with filters
    let apiUrl = `/api/narratives?limit=20&offset=0`;
    if (status !== 'all') apiUrl += `&status=${status}`;
    if (language !== 'all') apiUrl += `&language=${language}`;
    if (days !== 'all') apiUrl += `&days=${days}`;
    if (search) apiUrl += `&search=${search}`;
    
    fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            // Check if we have new data to display
            if (!data.narratives || data.narratives.length === 0) return;
            
            // Get current displayed narratives
            const displayedNarratives = new Set();
            document.querySelectorAll('[data-narrative-id]').forEach(el => {
                displayedNarratives.add(el.dataset.narrativeId);
            });
            
            // Check for new narratives
            let hasNewNarratives = false;
            for (const narrative of data.narratives) {
                if (!displayedNarratives.has(narrative.id.toString())) {
                    hasNewNarratives = true;
                    break;
                }
            }
            
            // Show notification if new narratives are available
            if (hasNewNarratives) {
                const notification = document.getElementById('new-content-notification');
                if (notification) {
                    notification.classList.remove('d-none');
                }
            }
        })
        .catch(error => {
            console.error('Error checking for new narratives:', error);
        });
}

/**
 * Show an alert message
 */
function showAlert(type, message) {
    const alertsContainer = document.getElementById('alerts-container');
    if (!alertsContainer) return;
    
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.role = 'alert';
    
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertsContainer.appendChild(alertElement);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        alertElement.classList.remove('show');
        setTimeout(() => {
            alertElement.remove();
        }, 150);
    }, 5000);
}

/**
 * Get array of day labels for the last N days
 */
function getDaysArray(numDays) {
    const daysArray = [];
    for (let i = numDays - 1; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        daysArray.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    return daysArray;
}

/**
 * Format a date in a readable format
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}
