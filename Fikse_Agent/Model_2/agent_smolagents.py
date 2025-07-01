from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import json
import requests

from smolagents import CodeAgent, tool

app = FastAPI()

# === Session Handling ===
sessions = {}

class ServiceItem(BaseModel):
    id: str
    service: str
    description: str
    price: float
    garment_type: str
    repairer_type: str
    estimated_hours: Optional[float] = None

class OrderSummary(BaseModel):
    order_id: str
    services: List[ServiceItem]
    total_price: float
    estimated_total_hours: Optional[float] = None
    created_at: str

class SessionState:
    def __init__(self):
        self.user_name: Optional[str] = None
        self.current_query: Optional[str] = None
        self.suggested_services: List[ServiceItem] = []
        self.selected_services: List[ServiceItem] = []
        self.conversation_state: str = "greeting"
        self.pending_order: Optional[OrderSummary] = None

def get_session(session_id: str) -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return sessions[session_id]

# === SmolAgent-compatible Tools ===
@tool
def search_repair_services(query: str) -> List[Dict]:
    """
    Search repair services based on a query string.

    Args:
        query (str): The user's query describing the repair issue.

    Returns:
        List[Dict]: A list of service dictionaries, each containing:
            - id (str): Unique service identifier.
            - service (str): Name of the service.
            - description (str): Service description.
            - price (float): Price of the service.
            - garment_type (str): Type of garment category.
            - repairer_type (str): Type of repairer.
            - estimated_hours (Any): Estimated time to complete the service.
    """
    try:
        response = requests.get("http://localhost:8000/search", params={"q": query})
        response.raise_for_status()
        results = response.json()
        return [
            {
                "id": f"service_{i+1}",
                "service": r.get("Service", "Unknown Service"),
                "description": r.get("Description", ""),
                "price": float(r.get("Price", 0)),
                "garment_type": r.get("Type of garment in category", ""),
                "repairer_type": r.get("Type of Repairer", ""),
                "estimated_hours": r.get("Estimated time in hours")
            }
            for i, r in enumerate(results[:10])
        ]
    except:
        return []

@tool
def store_session_data(session_id: str, key: str, value: str) -> str:
    """
    Store a key-value pair in the user session.

    Args:
        session_id (str): Identifier for the user session.
        key (str): The key for the data to store (e.g., user_name, conversation_state).
        value (str): The value to store; for complex objects, it should be JSON stringified.

    Returns:
        str: Confirmation message indicating successful storage.
    """
    session = get_session(session_id)
    if key == "user_name":
        session.user_name = value
    elif key == "conversation_state":
        session.conversation_state = value
    elif key == "selected_services":
        session.selected_services = [ServiceItem(**s) for s in json.loads(value)]
    elif key == "suggested_services":
        session.suggested_services = [ServiceItem(**s) for s in json.loads(value)]
    elif key == "current_query":
        session.current_query = value
    return f"Stored {key} in session {session_id}"

@tool
def get_session_data(session_id: str, key: str) -> str:
    """
    Retrieve a value from the user session by key.

    Args:
        session_id (str): Identifier for the user session.
        key (str): The key of the data to retrieve.

    Returns:
        str: The requested data as a string; for complex objects, JSON stringified.
    """
    session = get_session(session_id)
    if key == "user_name":
        return session.user_name or ""
    elif key == "conversation_state":
        return session.conversation_state
    elif key == "selected_services":
        return json.dumps([s.dict() for s in session.selected_services])
    elif key == "suggested_services":
        return json.dumps([s.dict() for s in session.suggested_services])
    elif key == "current_query":
        return session.current_query or ""
    return ""

@tool
def create_repair_order(session_id: str, services_json: str) -> Dict:
    """
    Create a repair order from a list of selected services.

    Args:
        session_id (str): Identifier for the user session.
        services_json (str): JSON string representing a list of selected service items.

    Returns:
        Dict: Order summary including order ID, services, total price, estimated hours, and creation timestamp.
              If error occurs, returns a dict with an "error" key and error message.
    """
    try:
        services = [ServiceItem(**s) for s in json.loads(services_json)]
        total_price = sum(s.price for s in services)
        total_hours = sum(s.estimated_hours for s in services if s.estimated_hours)
        order = OrderSummary(
            order_id=f"ORD-{uuid.uuid4().hex[:8].upper()}",
            services=services,
            total_price=total_price,
            estimated_total_hours=total_hours,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        session = get_session(session_id)
        session.pending_order = order
        session.conversation_state = "completed"
        return order.dict()
    except Exception as e:
        return {"error": str(e)}


# === Ollama Integration ===
class OllamaModel:
    def __init__(self, model_name="phi3"):
        self.model_name = model_name

    def generate(self, messages, **kwargs):
        """Generate response using Ollama API - smolagents compatible"""
        # Convert messages to a single prompt
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list):
            # Handle list of messages
            prompt = ""
            for msg in messages:
                if hasattr(msg, 'content'):
                    prompt += f"{msg.content}\n"
                elif isinstance(msg, dict):
                    prompt += f"{msg.get('content', '')}\n"
                else:
                    prompt += str(msg) + "\n"
        else:
            # Handle single message object
            if hasattr(messages, 'content'):
                prompt = messages.content
            else:
                prompt = str(messages)
        
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        })
        
        # Return the response content
        response_text = r.json()["response"]
        
        # Create a simple response object that smolagents can handle
        class TokenUsage:
            def __init__(self):
                self.input_tokens = 100
                self.output_tokens = 50
                self.total_tokens = 150
        
        class SimpleResponse:
            def __init__(self, content):
                self.content = content
                self.token_usage = TokenUsage()  # Add token_usage object for smolagents compatibility
        
        return SimpleResponse(response_text)

agent = CodeAgent(
    tools=[search_repair_services, store_session_data, get_session_data, create_repair_order],
    model=OllamaModel(),
    add_base_tools=False,
    max_iterations=3  # Limit iterations to prevent infinite loops
)

# === FastAPI Routes ===
class AgentInput(BaseModel):
    session_id: str
    user_input: str

@app.post("/agent")
def fikse_agent(input: AgentInput):
    # Create a more structured prompt for the agent
    system_prompt = """You are a helpful assistant for a clothing repair service. 

When a user asks for help with repairs, you should:
1. Search for relevant services using search_repair_services()
2. Store user information using store_session_data()
3. Create repair orders using create_repair_order()

Keep your code simple and direct. Always wrap your code in <code> tags.

Example format:
<code>
services = search_repair_services("silk dress repair")
print(f"Found {len(services)} services")
</code>

Be concise and avoid complex nested code structures."""
    
    session_context = f"{system_prompt}\n\nSession ID: {input.session_id}\nUser input: {input.user_input}"
    try:
        result = agent.run(session_context)
        session = get_session(input.session_id)
        return {
            "intent": "processed_by_llm",
            "response": result,
            "conversation_state": session.conversation_state,
            "show_services": len(session.suggested_services) > 0,
            "services": [s.dict() for s in session.suggested_services],
            "selected_services": [s.dict() for s in session.selected_services],
            "order_summary": session.pending_order.dict() if session.pending_order else None,
            "order_created": session.pending_order.dict() if session.conversation_state == "completed" else None,
            "session_id": input.session_id
        }
    except Exception as e:
        # Fallback response when agent fails
        print(f"Agent error: {str(e)}")
        return {
            "intent": "error",
            "response": "I apologize, but I'm having trouble processing your request right now. Please try asking in a simpler way, like 'Help me fix a torn silk dress'.",
            "conversation_state": "greeting",
            "show_services": False,
            "services": [],
            "selected_services": [],
            "order_summary": None,
            "order_created": None,
            "session_id": input.session_id
        }

@app.get("/")
def root():
    return {"message": "Fikse Agent Service", "status": "running", "endpoints": ["/agent", "/health"]}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fikse-agent-smolagents", "model": "ollama"}

@app.get("/agent/session/{session_id}")
def get_session_state(session_id: str):
    """Get current session state for debugging"""
    if session_id not in sessions:
        return {"error": "Session not found"}
    
    session = sessions[session_id]
    return message.get("content", "No answer produced.")