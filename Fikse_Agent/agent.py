# agent_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import requests
from typing import List, Dict, Optional
import uuid
from datetime import datetime

app = FastAPI()

### -------------------------------
### 1. Data Models
### -------------------------------

class AgentInput(BaseModel):
    session_id: str
    user_input: str

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

### -------------------------------
### 2. Intent Detection (Enhanced)
### -------------------------------

def detect_intent(text: str) -> str:
    text = text.lower()
    
    # Greeting patterns
    if re.match(r"(hi|hello|hey|good (morning|afternoon|evening)|start|begin)", text):
        return "greeting"
    
    # Name introduction
    if "my name is" in text or "i am" in text:
        return "introduce_self"
    
    # Repair request patterns - more comprehensive
    repair_keywords = ["fix", "tear", "hole", "repair", "zipper", "seam", "broken", "damage", 
                      "torn", "rip", "worn", "stain", "alter", "adjust", "shorten", "lengthen",
                      "take in", "take out", "button", "hem", "sleeve"]
    if any(word in text for word in repair_keywords):
        return "repair_request"
    
    # Service selection - including plain numbers
    if (re.search(r"(select|choose|pick|add|remove|service [0-9]|option [0-9]|number [0-9])", text) or 
        re.match(r"^[0-9]+$", text)):  # Added pattern for plain numbers
        return "service_selection"
    
    # Confirmation patterns
    if any(phrase in text for phrase in ["yes", "confirm", "looks good", "that's right", 
                                        "correct", "approve", "create order", "proceed"]):
        return "confirmation"
    
    # Manual addition request
    if any(phrase in text for phrase in ["add more", "add other", "add additional", 
                                        "manually add", "other services"]):
        return "manual_addition_request"
    
    # Decline manual addition
    if any(phrase in text for phrase in ["no more", "that's all", "no additional", 
                                        "no other", "just these", "no thanks"]):
        return "decline_addition"
    
    # Cancel patterns
    if any(word in text for word in ["cancel", "nevermind", "stop", "quit", "exit"]):
        return "cancel"
    
    return "unknown"

### -------------------------------
### 3. Search Integration (Fixed)
### -------------------------------

def query_fikse_search(query: str) -> List[Dict]:
    """Query the Fikse search API and return structured results"""
    try:
        # Use GET request with query parameter as per app.py
        response = requests.get("http://localhost:8000/search", params={"q": query})
        response.raise_for_status()
        
        results = response.json()
        
        # Convert search results to ServiceItem format
        services = []
        for i, result in enumerate(results[:10]):  # Limit to top 10
            service_item = ServiceItem(
                id=f"service_{i+1}",
                service=result.get("Service", "Unknown Service"),
                description=result.get("Description", ""),
                price=float(result.get("Price", 0)),
                garment_type=result.get("Type of garment in category", ""),
                repairer_type=result.get("Type of Repairer", ""),
                estimated_hours=result.get("Estimated time in hours") if result.get("Estimated time in hours") else None
            )
            services.append(service_item)
        
        return services
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

### -------------------------------
### 4. Session Management (Enhanced)
### -------------------------------

# Enhanced session state
sessions = {}

class SessionState:
    def __init__(self):
        self.user_name: Optional[str] = None
        self.current_query: Optional[str] = None
        self.suggested_services: List[ServiceItem] = []
        self.selected_services: List[ServiceItem] = []
        self.conversation_state: str = "greeting"  # greeting, searching, selecting, confirming, completed
        self.pending_order: Optional[OrderSummary] = None

def get_session(session_id: str) -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return sessions[session_id]

### -------------------------------
### 5. Utility Functions
### -------------------------------

def extract_name(text: str) -> str:
    """Extract name from introduction text"""
    patterns = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"i'm (\w+)",
        r"call me (\w+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).capitalize()
    
    return "Friend"

def format_service_list(services: List[ServiceItem], show_numbers: bool = True) -> str:
    """Format services for display with selection numbers"""
    if not services:
        return "No services found."
    
    formatted = []
    for i, service in enumerate(services, 1):
        price_str = f"${service.price:.0f}" if service.price > 0 else "Price on request"
        if show_numbers:
            formatted.append(f"{i}. **{service.service}** - {service.description} ({price_str})")
        else:
            formatted.append(f"• **{service.service}** - {service.description} ({price_str})")
    
    return "\n".join(formatted)

def parse_service_selection(text: str, available_services: List[ServiceItem]) -> List[ServiceItem]:
    """Parse user selection and return selected services"""
    selected = []
    
    # Look for numbers in the text
    numbers = re.findall(r'\b(\d+)\b', text)
    
    for num_str in numbers:
        try:
            index = int(num_str) - 1  # Convert to 0-based index
            if 0 <= index < len(available_services):
                selected.append(available_services[index])
        except ValueError:
            continue
    
    return selected

def create_order_summary(services: List[ServiceItem]) -> OrderSummary:
    """Create order summary from selected services"""
    total_price = sum(service.price for service in services)
    total_hours = None
    
    # Calculate total hours if available
    hours_list = [s.estimated_hours for s in services if s.estimated_hours is not None]
    if hours_list:
        total_hours = sum(hours_list)
    
    return OrderSummary(
        order_id=f"ORD-{uuid.uuid4().hex[:8].upper()}",
        services=services,
        total_price=total_price,
        estimated_total_hours=total_hours,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

### -------------------------------
### 6. Intent Handler (Enhanced)
### -------------------------------

def handle_intent(session_id: str, intent: str, user_input: str) -> Dict:
    """Enhanced intent handler with full conversation flow"""
    session = get_session(session_id)
    
    if intent == "greeting":
        session.conversation_state = "greeting"
        return {
            "response": "👋 Hi there! I'm here to help you create repair orders with AI assistance.\n\nWhat garment needs fixing today? Please describe the item and what's wrong with it.",
            "state": "greeting",
            "show_services": False
        }
    
    elif intent == "introduce_self":
        name = extract_name(user_input)
        session.user_name = name
        return {
            "response": f"Nice to meet you, {name}! 😊\n\nWhat garment needs fixing today? Please describe the item and the damage you see.",
            "state": "greeting",
            "show_services": False
        }
    
    elif intent == "repair_request":
        session.current_query = user_input
        session.conversation_state = "searching"
        
        # Query the search API
        services = query_fikse_search(user_input)
        
        if not services:
            return {
                "response": "I couldn't find any matching services for your request. Could you try describing the garment and damage differently?",
                "state": "searching",
                "show_services": False
            }
        
        # Store suggested services and limit to top 5 for selection
        session.suggested_services = services[:5]
        session.conversation_state = "selecting"
        
        greeting = f"Great! " if not session.user_name else f"Perfect, {session.user_name}! "
        
        return {
            "response": f"{greeting}Based on your description, here are the top 5 suggested repair services:",
            "state": "selecting",
            "show_services": True,
            "services": [service.dict() for service in session.suggested_services]
        }
    
    elif intent == "service_selection":
        if session.conversation_state != "selecting" or not session.suggested_services:
            return {
                "response": "Please start by describing what needs to be repaired first.",
                "state": "greeting",
                "show_services": False
            }
        
        selected = parse_service_selection(user_input, session.suggested_services)
        
        if not selected:
            return {
                "response": "Please select a service from the options above.",
                "state": "selecting",
                "show_services": True,
                "services": [service.dict() for service in session.suggested_services]
            }
        
        session.selected_services = selected
        session.conversation_state = "manual_addition"
        
        return {
            "response": f"✅ Great! You've selected:\n\n{format_service_list(selected, show_numbers=False)}\n\n**Would you like to add any other services manually?** (Yes/No)",
            "state": "manual_addition",
            "show_services": False,
            "selected_services": [service.dict() for service in selected]
        }
    
    elif intent == "manual_addition_request":
        return {
            "response": "Please describe the additional service you'd like to add:",
            "state": "adding_manual",
            "show_services": False
        }
    
    elif intent == "decline_addition" or (intent == "unknown" and session.conversation_state == "manual_addition"):
        if not session.selected_services:
            return {
                "response": "No services selected. Please start over by describing what needs repair.",
                "state": "greeting",
                "show_services": False
            }
        
        # Create order summary
        session.pending_order = create_order_summary(session.selected_services)
        session.conversation_state = "confirming"
        
        hours_info = f"\n**Estimated Time:** {session.pending_order.estimated_total_hours:.1f} hours" if session.pending_order.estimated_total_hours else ""
        
        return {
            "response": f"📋 **Order Summary:**\n\n{format_service_list(session.selected_services, show_numbers=False)}\n\n**Total Price:** ${session.pending_order.total_price:.0f}{hours_info}\n\n**Do you want to confirm and create this order?**",
            "state": "confirming",
            "show_services": False,
            "order_summary": session.pending_order.dict()
        }
    
    elif intent == "confirmation":
        if session.conversation_state != "confirming" or not session.pending_order:
            return {
                "response": "No order to confirm. Please start by describing what needs repair.",
                "state": "greeting",
                "show_services": False
            }
        
        # "Create" the order (in a real system, this would save to database)
        order = session.pending_order
        session.conversation_state = "completed"
        
        hours_info = f"\n📅 **Estimated Time:** {order.estimated_total_hours:.1f} hours" if order.estimated_total_hours else ""
        
        return {
            "response": f"🎉 **Order Created Successfully!**\n\n**Order ID:** {order.order_id}\n**Created:** {order.created_at}\n\n**Services:**\n{format_service_list(session.selected_services, show_numbers=False)}\n\n💰 **Total Price:** ${order.total_price:.0f}{hours_info}\n\n✅ Your repair order has been created and is ready for processing!\n\n*Would you like to create another order?*",
            "state": "completed",
            "show_services": False,
            "order_created": order.dict()
        }
    
    elif intent == "cancel":
        # Reset session
        sessions[session_id] = SessionState()
        return {
            "response": "❌ Order cancelled. Feel free to start over whenever you're ready!",
            "state": "greeting",
            "show_services": False
        }
    
    else:  # unknown intent
        if session.conversation_state == "selecting":
            # If we're in selecting state and user input is a number, try to parse it as service selection
            if re.match(r"^[0-9]+$", user_input.strip()):
                selected = parse_service_selection(user_input, session.suggested_services)
                
                if selected:
                    session.selected_services = selected
                    session.conversation_state = "manual_addition"
                    
                    return {
                        "response": f"✅ Great! You've selected:\n\n{format_service_list(selected, show_numbers=False)}\n\n**Would you like to add any other services manually?** (Yes/No)",
                        "state": "manual_addition",
                        "show_services": False,
                        "selected_services": [service.dict() for service in selected]
                    }
            
            return {
                "response": "Please select a service from the options above.",
                "state": "selecting",
                "show_services": True,
                "services": [service.dict() for service in session.suggested_services]
            }
        elif session.conversation_state == "confirming":
            return {
                "response": "Please answer Yes or No to confirm the order.",
                "state": "confirming",
                "show_services": False
            }
        else:
            return {
                "response": "I didn't quite understand that. Could you describe the garment and what needs fixing?",
                "state": "greeting",
                "show_services": False
            }

### -------------------------------
### 7. FastAPI Endpoints
### -------------------------------

@app.post("/agent")
def fikse_agent(input: AgentInput):
    """Main agent endpoint with enhanced response structure"""
    try:
        intent = detect_intent(input.user_input)
        response_data = handle_intent(input.session_id, intent, input.user_input)
        
        return {
            "intent": intent,
            "response": response_data["response"],
            "conversation_state": response_data["state"],
            "show_services": response_data.get("show_services", False),
            "services": response_data.get("services", []),
            "selected_services": response_data.get("selected_services", []),
            "order_summary": response_data.get("order_summary"),
            "order_created": response_data.get("order_created"),
            "session_id": input.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/agent/session/{session_id}")
def get_session_state(session_id: str):
    """Get current session state for debugging"""
    if session_id not in sessions:
        return {"error": "Session not found"}
    
    session = sessions[session_id]
    return {
        "user_name": session.user_name,
        "conversation_state": session.conversation_state,
        "suggested_services_count": len(session.suggested_services),
        "selected_services_count": len(session.selected_services),
        "has_pending_order": session.pending_order is not None
    }

@app.delete("/agent/session/{session_id}")
def reset_session(session_id: str):
    """Reset a session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session reset"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fikse-agent"}
