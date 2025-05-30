#!/bin/bash

echo "🚀 AI Chatbot Installation Script"
echo "================================="

# Check system requirements
check_requirements() {
    echo "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi
    
    # Check GPU (optional)
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        GPU_AVAILABLE=true
    else
        echo "⚠️  No NVIDIA GPU detected. Will use CPU mode."
        GPU_AVAILABLE=false
    fi
    
    echo "✅ All requirements met!"
}

# Setup environment
setup_environment() {
    echo "Setting up environment..."
    
    # Create .env file if not exists
    if [ ! -f .env ]; then
        echo "Creating .env file..."
        cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_URL=redis://redis:6379

# Models
MODEL_TIER=auto
USE_GPU=$GPU_AVAILABLE

# Security
SECRET_KEY=$(openssl rand -base64 32)
CORS_ORIGINS=["*"]
EOF
        echo "✅ Environment file created"
    fi
    
    # Create necessary directories
    mkdir -p models chroma_db logs
    chmod 777 models chroma_db logs
}

# Download models
download_models() {
    echo "Download AI models? (This may take 10-30 minutes)"
    read -p "Download now? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker run --rm \
            -v $(pwd)/models:/models \
            -e HF_HOME=/models \
            python:3.10-slim \
            bash -c "pip install huggingface_hub && python scripts/download_models.py"
    else
        echo "⚠️  Skipping model download. Models will be downloaded on first use."
    fi
}

# Start services
start_services() {
    echo "Starting services..."
    
    # Pull images
    docker-compose pull
    
    # Build custom images
    docker-compose build
    
    # Start services
    docker-compose up -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo "✅ API is healthy!"
    else
        echo "⚠️  API health check failed. Check logs with: docker-compose logs api"
    fi
}

# Print instructions
print_instructions() {
    echo ""
    echo "🎉 Installation Complete!"
    echo "========================"
    echo ""
    echo "To add the chatbot to your website, add this line to your HTML:"
    echo ""
    echo '<script src="http://YOUR_SERVER_IP/widget/widget.js"></script>'
    echo ""
    echo "Or use the advanced configuration:"
    echo ""
    echo '<script>'
    echo '  window.AI_CHATBOT_API_URL = "http://YOUR_SERVER_IP";'
    echo '  window.AI_CHATBOT_AUTO_START = true;'
    echo '</script>'
    echo '<script src="http://YOUR_SERVER_IP/widget/widget.js"></script>'
    echo ""
    echo "Dashboard: http://localhost:8000/api/docs"
    echo ""
    echo "To stop: docker-compose down"
    echo "To view logs: docker-compose logs -f"
    echo ""
}

# Main installation flow
main() {
    check_requirements
    setup_environment
    download_models
    start_services
    print_instructions
}

# Run main
main