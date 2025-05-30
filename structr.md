ai-chatbot/
├── backend/
│   ├── __init__.py  # Create empty file
│   ├── crawler/
│   │   ├── __init__.py  # Create empty file
│   │   ├── intelligent_crawler.py
│   │   └── discovery_strategies.py  # You need to create this
│   ├── processor/
│   │   ├── __init__.py  # Create empty file
│   │   ├── multimodal_parser.py
│   │   ├── visual_understanding.py  # You need to create this
│   │   ├── layout_analyzer.py  # You need to create this
│   │   └── knowledge_builder.py  # You need to create this
│   ├── chatbot/
│   │   ├── __init__.py  # Create empty file
│   │   ├── reasoning_engine.py
│   │   ├── retrieval_optimizer.py
│   │   └── complexity_classifier.py  # You need to create this
│   ├── api/
│   │   ├── __init__.py  # Create empty file
│   │   └── main.py
│   └── core/
│       ├── __init__.py  # Create empty file
│       └── config.py
├── frontend/
│   └── widget/
│       └── widget.js
├── docker/
│   └── Dockerfile.api
├── scripts/
│   ├── install.sh
│   └── download_models.py  # You need to create this
├── docker-compose.yml  # Move this to root
├── requirements.txt  # You need to create this
└── .env  # Will be created