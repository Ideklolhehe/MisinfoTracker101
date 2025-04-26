#!/bin/bash
# CIVILIAN System Management Script

# Display usage information
show_help() {
    echo "CIVILIAN System Management Script"
    echo "=================================="
    echo ""
    echo "Usage: bash civilian_scripts.sh [command]"
    echo ""
    echo "Available Commands:"
    echo "  start             - Start the main CIVILIAN application"
    echo "  generate          - Generate adversarial training content"
    echo "  evaluate          - Evaluate the detector against adversarial content"
    echo "  fix-auth          - Fix authentication issues"
    echo "  migrate           - Run database migrations"
    echo "  sources           - Initialize data sources"
    echo "  help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash civilian_scripts.sh start"
    echo "  bash civilian_scripts.sh generate --topic health --type conspiracy_theory --batch 3"
    echo "  bash civilian_scripts.sh evaluate --limit 10"
    echo ""
}

# Execute the main application
start_app() {
    echo "Starting CIVILIAN application..."
    gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
}

# Generate adversarial content
generate_content() {
    if [ "$#" -eq 0 ]; then
        echo "Error: Missing required parameters for content generation."
        echo "Usage: bash civilian_scripts.sh generate --topic TOPIC --type TYPE [--batch COUNT] [--input-file FILE]"
        echo ""
        echo "Required parameters:"
        echo "  --topic TOPIC    Topic area for misinformation (health, science, politics, etc.)"
        echo "  --type TYPE      Type of misinformation (conspiracy_theory, misleading_statistics, etc.)"
        echo ""
        echo "Optional parameters:"
        echo "  --batch COUNT    Number of examples to generate (1-10, default: 1)"
        echo "  --input-file FILE Path to file with real content to base misinformation on"
        exit 1
    fi
    
    echo "Generating adversarial training content..."
    python generate_training_content.py "$@"
}

# Evaluate detector
evaluate_detector() {
    echo "Evaluating detector against adversarial content..."
    python evaluate_detector.py "$@"
}

# Fix authentication issues
fix_auth() {
    echo "Fixing authentication issues..."
    python fix_auth.py
}

# Run database migrations
run_migrations() {
    echo "Running database migrations..."
    python migrate_title_field.py "$@"
}

# Initialize data sources
init_sources() {
    echo "Initializing data sources..."
    python init_open_sources.py "$@"
}

# Main command processing
if [ "$#" -eq 0 ]; then
    show_help
    exit 0
fi

# Process command
command="$1"
shift

case "$command" in
    start)
        start_app
        ;;
    generate)
        generate_content "$@"
        ;;
    evaluate)
        evaluate_detector "$@"
        ;;
    fix-auth)
        fix_auth
        ;;
    migrate)
        run_migrations "$@"
        ;;
    sources)
        init_sources "$@"
        ;;
    help)
        show_help
        ;;
    *)
        echo "Error: Unknown command '$command'"
        show_help
        exit 1
        ;;
esac
