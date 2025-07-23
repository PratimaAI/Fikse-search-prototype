from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import json

app = FastAPI()


# -----------------------------
# Data Models
# -----------------------------

class AgentInput(BaseModel):
    user_input: str

class ServiceItem(BaseModel):
    id: str
    service: str
    description: str
    price: float
    garment_type: str
    repairer_type: str
    estimated_hours: float


# -----------------------------
# In-Memory Session Storage
# -----------------------------

class Session:
    def __init__(self):
        self.user_name = ""
        self.conversation_state = ""
        self.selected_services = []
        self.suggested_services = []
        self.current_query = ""

_sessions = {}

def get_session(session_id: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session()
    return _sessions[session_id]


# -----------------------------
# API Endpoints
# -----------------------------

@app.post("/agent")
def fikse_agent(input: AgentInput):
    try:
        prompt = (
            "You are a helpful assistant, a user-friendly, solution-oriented, and modern tool built for real people "
            "who work with their hands — and who are often short on time. Respond conversationally.\n"
            f"User: {input.user_input}\nAssistant:"
        )
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        })
        response_text = r.json()["response"]
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Fikse Agent Service", "status": "running", "endpoints": ["/agent", "/health", "/search"]}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fikse-agent", "model": "ollama"}


@app.get("/search")
def search_services(q: str):
    # Dummy data for testing
    return [
        {
            "Service": "Zipper Replacement",
            "Description": "Replace broken zipper on jacket",
            "Price": "20",
            "Type of garment in category": "Jacket",
            "Type of Repairer": "Tailor",
            "Estimated time in hours": 2
        },
        {
            "Service": "Patch Hole",
            "Description": "Patch small hole on sleeve",
            "Price": "10",
            "Type of garment in category": "Sweater",
            "Type of Repairer": "Tailor",
            "Estimated time in hours": 1
        }
    ]


# -----------------------------
# Session Data Helpers
# -----------------------------

def store_session_data(session_id: str, key: str, value: str) -> str:
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

def get_session_data(session_id: str, key: str) -> str:
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


# -----------------------------
# Service Search Helper
# -----------------------------

def query_fikse_search(query: str) -> List[ServiceItem]:
    try:
        print(f"🔍 Searching for: {query}")
        response = requests.get("http://localhost:8000/search", params={"q": query})
        response.raise_for_status()
        results = response.json()
        print(f"📋 Found {len(results)} raw results")

        services = []
        for i, result in enumerate(results[:10]):
            service_item = ServiceItem(
                id=f"service_{i+1}",
                service=result.get("Service", "Unknown Service"),
                description=result.get("Description", ""),
                price=float(result.get("Price", 0)),
                garment_type=result.get("Type of garment in category", ""),
                repairer_type=result.get("Type of Repairer", ""),
                estimated_hours=float(result.get("Estimated time in hours", 0))
            )
            services.append(service_item)

        print(f"✅ Returning {len(services)} formatted services")
        return services
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return []
