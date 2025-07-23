# 🧵 FikseAgent - AI-Powered Tailor Assistant

An intelligent agent system that streamlines repair order creation through conversational AI and semantic search.

## 🎯 Overview

FikseAgent combines advanced semantic search with conversational AI to help tailors create repair orders efficiently. The system suggests relevant services based on natural language descriptions and guides users through a structured order creation process.

## ✨ Key Features

### 🔍 **Advanced Search Engine**
- **Semantic Search**: Uses sentence transformers and FAISS for contextual understanding
- **Hybrid Matching**: Combines semantic similarity with exact keyword matching  
- **Smart Prioritization**: Exact service matches first, then semantic suggestions
- **Price Filtering**: Optional price-based filtering with flexible ranges

### 🤖 **Conversational Agent**
- **Natural Language Processing**: Understands repair descriptions in plain English
- **Multi-step Workflow**: Guides users through service selection and order creation
- **Session Management**: Maintains conversation context across interactions
- **Intent Recognition**: Detects greetings, repair requests, confirmations, etc.

### 💬 **Modern Chat Interface**
- **Streamlit-based UI**: Clean, responsive chat interface
- **Service Selection**: Interactive service browsing with detailed information
- **Order Management**: Real-time order summaries and confirmations
- **Visual Feedback**: Clear status indicators and progress tracking

### 📋 **Order Management**
- **Structured Data**: Consistent order format with all necessary details
- **Price Calculation**: Automatic totaling with estimated time tracking
- **Order Export**: Download order details as JSON for integration
- **Session Persistence**: Orders maintained throughout conversation

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chat UI       │    │   Agent API     │    │   Search API    │
│  (Streamlit)    │◄──►│   (FastAPI)     │◄──►│   (FastAPI)     │
│  Port: 8501     │    │  Port: 8001     │    │  Port: 8000     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Session Storage │    │ FAISS Index +   │
                       │ (In-memory)     │    │ CSV Dataset     │
                       └─────────────────┘    └─────────────────┘
```

### Components:

1. **Search API** (`app.py`): 
   - FAISS-powered semantic search
   - Hybrid keyword + semantic matching
   - Dataset indexing and query processing

2. **Agent API** (`agent.py`):
   - Conversation management
   - Intent detection and handling
   - Order creation workflow
   - Session state management

3. **Chat UI** (`chat_ui.py`):
   - User interface with Streamlit
   - Real-time conversation display
   - Service selection interface
   - Order visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required dependencies (see `requirements.txt`)
- Precomputed dataset and FAISS index

### 🔧 Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset** (if not already done):
   ```bash
   python precompute_dataset.py
   ```

3. **Start the System**:
   ```bash
   python start_system.py
   ```

4. **Access the Chat Interface**:
   Open [http://localhost:8501](http://localhost:8501) in your browser

### Manual Startup (Alternative)

If you prefer to start services individually:

```bash
# Terminal 1 - Search API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Agent API  
uvicorn agent:app --host 0.0.0.0 --port 8001 --reload

# Terminal 3 - Chat UI
streamlit run chat_ui.py --server.port 8501
```

## 💡 Usage Examples

### Example Conversation Flow:

```
👤 User: "I have a silk dress with a small tear near the hem"

🤖 Agent: "Perfect! Based on your description, here are the top 5 suggested repair services:

1. **Hole** - Price per hole ($199)
2. **Seam** - Small seam opening, e.g., a loosened tag ($99)  
3. **Shorten dress** - Price per layer on dress ($399)
4. **Lengthen dress** - Price per layer on dress ($399)
5. **Dry cleaning** - Short or simple dress ($399)

Please select the services you need by typing the numbers (e.g., "1, 3, 5" or just "2"):"

👤 User: "1"

🤖 Agent: "✅ Great! You've selected:
• **Hole** - Price per hole ($199)

Would you like to add any other services manually? (Yes/No)"

👤 User: "No"

🤖 Agent: "📋 **Order Summary:**
• **Hole** - Price per hole ($199)

**Total Price:** $199

**Do you want to confirm and create this order?** (Yes/No)"

👤 User: "Yes"

🤖 Agent: "🎉 **Order Created Successfully!**

**Order ID:** ORD-A1B2C3D4
**Created:** 2024-01-15 14:30:25

**Services:**
• **Hole** - Price per hole ($199)

💰 **Total Price:** $199

✅ Your repair order has been created and is ready for processing!"
```

## 🔧 API Endpoints

### Search API (Port 8000)
- `GET /search?q={query}` - Search for services
- `GET /search_strategy` - Get search strategy info
- `GET /debug_search?q={query}` - Debug search processing
- `GET /health` - Health check

### Agent API (Port 8001)
- `POST /agent` - Main conversation endpoint
- `GET /agent/session/{session_id}` - Get session state
- `DELETE /agent/session/{session_id}` - Reset session
- `GET /health` - Health check

## 📊 Dataset Structure

The system uses a CSV dataset with the following fields:
- `Type of Repairer` - Who performs the service (Tailor, Cobbler, etc.)
- `Type of category` - Service category (Clothes, Shoes, Accessories)
- `Type of garment in category` - Specific garment type
- `Service` - Service name (Hole, Zipper, Seam, etc.)
- `Description` - Detailed service description
- `Price` - Service cost
- `Estimated time in hours` - Expected completion time

## 🎨 Conversation States

The agent manages conversations through these states:

1. **Greeting** - Initial welcome and problem description
2. **Searching** - Processing repair request and finding services
3. **Selecting** - User choosing from suggested services  
4. **Manual Addition** - Asking about additional services
5. **Confirming** - Order review and confirmation
6. **Completed** - Order created successfully

## 🔍 Search Algorithm

The hybrid search uses a two-stage approach:

### Stage 1: Semantic Candidate Retrieval
- Uses sentence transformers to embed queries
- FAISS index retrieval of top 100 candidates
- Captures semantic similarity and related concepts

### Stage 2: Keyword-Based Prioritization
- **Exact Service Matches** (Highest Priority)
- **Partial Service Matches** (High Priority)  
- **Description Matches** (Medium Priority)
- **General Field Matches** (Lower Priority)
- **Semantic Only** (Lowest Priority)

### Stage 3: Result Ranking
- Sort each priority group by semantic similarity
- Return top 10 results with priority ordering
- Apply optional price filtering

## 🛠️ Configuration

### Search Parameters (in `app.py`):
- `k=100` - Number of semantic candidates to retrieve
- Max results returned: 10
- Price filter tolerance: ±50

### Agent Parameters (in `agent.py`):
- Max suggested services shown: 5
- Session timeout: Until manually reset
- Order ID format: ORD-{8-char-hex}

## 🧪 Testing

Test the system with these example queries:
- "I have a silk dress with a small tear"
- "My jacket zipper is broken"  
- "Need to alter a suit jacket - take in the waist"
- "Shoes need new soles"
- "Button fell off my shirt"

## 📁 File Structure

```
FIKSE_AGENT/
├── app.py                 # Search API service
├── agent.py              # Agent API service  
├── chat_ui.py            # Streamlit chat interface
├── start_system.py       # System startup script
├── precompute_dataset.py # Dataset preprocessing
├── Dataset_categories.csv # Service dataset
├── faiss.index          # FAISS search index
├── precomputed_dataset/ # Processed dataset
├── templates/           # HTML templates
├── requirements.txt     # Python dependencies
└── README.md           # This documentation
```

## 🔧 Development

### Adding New Services
1. Update `Dataset_categories.csv` with new service entries
2. Run `python precompute_dataset.py` to reindex
3. Restart the search API

### Extending the Agent
- Modify intent detection in `detect_intent()`
- Add new conversation states in `handle_intent()`
- Update the UI to display new features

### Customizing the UI
- Edit `chat_ui.py` for interface changes
- Modify CSS styles in the Streamlit markdown
- Add new visualization components

## ❗ Troubleshooting

### Common Issues:
- **Port already in use**: Check if services are already running
- **FAISS index missing**: Run `python precompute_dataset.py`
- **Connection refused**: Ensure all services are started in order
- **No search results**: Check dataset and index files

### Debugging:
- Use `/debug_search` endpoint to trace search processing
- Check service logs for error messages  
- Verify session state with `/agent/session/{id}` endpoint

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Fikse platform for streamlining repair services.

---

🎉 **Ready to streamline your repair orders with AI?** Start the system and begin chatting! 