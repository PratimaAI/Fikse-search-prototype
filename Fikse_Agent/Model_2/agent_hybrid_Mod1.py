# This is the modification of hybrid agent, it uses LLM(ollama model) for service selection and comment generation. 
# It also uses the search module. 
# Addition:(1) This code adds the comment section after order creation using LLM based on the Prompt file.
#(2) This code add a new function select_services which generates the single best service based on user input. 
#(3) Uses direct responses for better performance and reliability.
#(4) DATASET-AGNOSTIC: Works with any dataset without code changes
#(5) SHARED CONFIGURATION: Uses single shared config file

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import re
import requests
import json
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import json
import threading

# Constants for better performance and maintainability
GARMENT_TYPES = frozenset([
    # Clothing items
    "dress", "shirt", "pants", "jacket", "coat", "blouse", "skirt", "suit", 
    "jeans", "trousers", "sweater", "cardigan", "blazer", "shorts", "top", 
    "outfit", "clothing", "garment", "clothes", "arm", "backpack", "bag", 
    "belt", "boots", "bottom", "bunad", "bunad silver", "button", "chain", 
    "bracelet", "curtain", "dress shoes", "flats", "hat", "hole", "jewellery", 
    "necklace", "pearl", "ring", "seam", "sleeves", "sneakers", "wedding dress", 
    "winter shoes",
    # Service-related terms
    "repair", "alteration", "dry cleaning", "cleaning", "fix", "mend", "sew",
    "stitch", "adjust", "resize", "shorten", "shortened", "shortnened", "lengthen", "lengthened",
    "replace", "attach", "detach", "polish", "clean", "wash", "iron", "press",
    "take in", "take out", "needs", "help", "service"
])

FABRIC_TYPES = frozenset([
    "silk", "cotton", "wool", "linen", "polyester", "denim", "leather", 
    "cashmere", "satin", "chiffon", "velvet", "corduroy", "gold", "silver",
    "metal", "plastic", "wood", "glass", "ceramic", "stone", "gem", "diamond",
    "ruby", "emerald", "pearl", "crystal"
])

DAMAGE_TYPES = frozenset([
    "tear", "torn", "hole", "stain", "zipper", "button", "seam", "hem", "rip", 
    "worn", "faded", "shrunk", "stretched", "loose", "tight", "broken",
    "damaged", "ruined", "falling apart", "needs fixing", "polishing", 
    "replace", "adjustment", "overhaul", "dirty", "moldy", "mildew",
    "wrinkled", "creased", "scratched", "dented", "bent", "missing",
    "detached", "unraveled", "frayed", "discolored", "tarnished"
])

CATEGORY_TYPES = frozenset([
    "accessories", "clothes", "jewellery", "other textiles", "shoes", 
    "special occasion", "suit", "watch", "bags", "belts", "hats", "scarves",
    "gloves", "socks", "underwear", "lingerie", "formal wear", "casual wear",
    "sportswear", "workwear", "uniforms", "costumes", "traditional wear"
])

BUSINESS_TYPES = frozenset([
    "bridal seamstress", "bunadtilvirker", "cobbler", "dry cleaner", 
    "goldsmith", "leathermaker", "tailor", "watchmaker", "seamstress",
    "dressmaker", "alteration specialist", "repair shop", "cleaning service",
    "jewelry repair", "shoe repair", "leather repair", "watch repair"
])

SERVICE_TYPES = frozenset([
    "repair", "alteration", "dry cleaning", "other", "cleaning", "maintenance",
    "restoration", "polishing", "adjustment", "replacement", "installation",
    "assembly", "disassembly", "inspection", "testing", "calibration",
    # Common variations and misspellings
    "shorten", "shortened", "shortnened", "lengthen", "lengthened", "lenghten",
    "take in", "take out", "resize", "resized", "adjust", "adjusted",
    "fix", "fixed", "mend", "mended", "sew", "sewed", "stitch", "stitched"
])

VALID_INTENTS = frozenset([
    "repair_request", "greeting", "service_selection", "confirmation", "unknown"
])

# Compiled regex patterns for better performance
NUMBER_PATTERN = re.compile(r"^[0-9]+$")
CONFIRMATION_PATTERN = re.compile(r'\b(yes|confirm|okay|ok)\b|looks good')
CANCEL_PATTERN = re.compile(r'\b(no|cancel|nevermind|back)\b')
GREETING_PATTERN = re.compile(r'\b(hi|hello|hey|start|begin)\b')
DIGIT_PATTERN = re.compile(r"\b\d+\b")

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
SEARCH_URL = "http://localhost:8000/search"
DEFAULT_MODEL = "phi3"

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
    category: Optional[str] = None
    service_type: Optional[str] = None

class OrderSummary(BaseModel):
    order_id: str
    services: List[ServiceItem]
    total_price: float
    estimated_total_hours: Optional[float] = None
    created_at: str

### -------------------------------
### 2. Shared Configuration
### -------------------------------

def load_shared_config():
    """Load dataset configuration from JSON file"""
    try:
        with open("dataset.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  dataset.json not found. Using default configuration.")
        return {
            "search_priorities": {
                "exact_service_name": "service_name",
                "partial_service_name": "service_name", 
                "service_type": "service_type",
                "description": "description",
                "item_name": "item_name",
                "business_type": "business_type"
            },
            "price_column": "price",
            "hours_column": "hours",
            "standard_columns": [
                "business_type", "category", "item_name", "service_name", 
                "service_type", "description", "price", "hours"
            ]
        }

# Load shared configuration at startup
config = load_shared_config()
print(f"📋 Agent using shared configuration")

### -------------------------------
### 3. Enhanced Intent Detection
### -------------------------------

def detect_intent_and_context(text: str) -> Dict:
    """Enhanced intent detection with AI fallback for unknown intents"""
    text_lower = text.lower()
    print(f"Intent detection for: '{text}'")
    
    # Extract context first (always useful)
    context = {
        "garment_type": None,
        "damage_type": None,
        "fabric_type": None,
        "category_type": None,
        "business_type": None,
        "service_type": None
    }
    
    # Detect garment types using set intersection for better performance
    garment_intersection = GARMENT_TYPES.intersection(text_lower.split())
    if garment_intersection:
        context["garment_type"] = next(iter(garment_intersection))
        print(f"Found garment: {context['garment_type']}")
    
    # Detect fabric types
    fabric_intersection = FABRIC_TYPES.intersection(text_lower.split())
    if fabric_intersection:
        context["fabric_type"] = next(iter(fabric_intersection))
        print(f"Found fabric: {context['fabric_type']}")
        if context["garment_type"]:
            context["garment_type"] = f"{context['fabric_type']} {context['garment_type']}"
    
    # Detect damage/issue types
    damage_intersection = DAMAGE_TYPES.intersection(text_lower.split())
    if damage_intersection:
        context["damage_type"] = next(iter(damage_intersection))
        print(f"Found damage: {context['damage_type']}")
    
    # Detect category types
    category_intersection = CATEGORY_TYPES.intersection(text_lower.split())
    if category_intersection:
        context["category_type"] = next(iter(category_intersection))
        print(f"Found category: {context['category_type']}")
    
    # Detect business types
    business_intersection = BUSINESS_TYPES.intersection(text_lower.split())
    if business_intersection:
        context["business_type"] = next(iter(business_intersection))
        print(f"Found business type: {context['business_type']}")
    
    # Detect service types
    service_intersection = SERVICE_TYPES.intersection(text_lower.split())
    if service_intersection:
        context["service_type"] = next(iter(service_intersection))
        print(f"Found service type: {context['service_type']}")
    
    # Simple intent patterns first using compiled regex
    if NUMBER_PATTERN.match(text.strip()):
        print("Matched number pattern")
        return {"intent": "service_selection", "context": context}
    elif CONFIRMATION_PATTERN.search(text_lower):
        print("Matched confirmation pattern")
        return {"intent": "confirmation", "context": context}
    elif CANCEL_PATTERN.search(text_lower):
        print("Matched cancel pattern")
        return {"intent": "cancel", "context": context}
    elif GREETING_PATTERN.search(text_lower):
        print("Matched greeting pattern")
        return {"intent": "greeting", "context": context}
    
    # If we found any relevant context, likely a service request
    print(f"Context extracted: {context}")
    if (context["garment_type"] or context["fabric_type"] or context["damage_type"] or 
        context["category_type"] or context["business_type"] or context["service_type"]):
        print("Detected repair_request from context")
        return {"intent": "repair_request", "context": context}
    
    # For everything else, use AI to classify intent
    print("Using AI to classify intent")
    return ai_classify_intent(text, context)

def ai_classify_intent(text: str, context: Dict) -> Dict:
    """Use AI to classify intent when keyword matching fails"""
    try:
        prompt = f"""You are an intent classifier for a service assistant. 
        
User said: "{text}"

Based on this input, classify the intent as one of:
- repair_request: User needs service/repair/fixing
- greeting: User is saying hello or starting conversation  
- service_selection: User is selecting from options
- confirmation: User is confirming something
- unknown: Doesn't fit any category

Respond with ONLY the intent name, nothing else."""

        response = requests.post(OLLAMA_URL, json={
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False
        })
        
        ai_intent = response.json()["response"].strip().lower()
        print(f"AI classified intent as: '{ai_intent}'")
        
        # Validate AI response
        if ai_intent in VALID_INTENTS:
            print(f"Valid AI intent: {ai_intent}")
            return {"intent": ai_intent, "context": context}
        else:
            print(f"Invalid AI intent: {ai_intent}, using fallback")
            # Fallback: if AI response is invalid, assume repair request if any context found
            if context["garment_type"] or context["fabric_type"] or context["damage_type"] or context["category_type"]:
                return {"intent": "repair_request", "context": context}
            return {"intent": "unknown", "context": context}
            
    except Exception:
        # If AI fails, use smart fallback
        if context["garment_type"] or context["fabric_type"] or context["damage_type"] or context["category_type"]:
            return {"intent": "repair_request", "context": context}
        return {"intent": "unknown", "context": context}

### -------------------------------
### 4. AI Response Generation
### -------------------------------

class AIResponseGenerator:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for response generation"""
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            })
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Please describe what item needs service."

    def select_services_with_llm(self, user_input: str, services: List[ServiceItem]) -> List[ServiceItem]:
        service_options = "\n".join(
            [
                f"{i+1}. {s.repairer_type} | {s.category} | {s.garment_type} | {s.service} | {s.service_type} | {s.description} | {s.price}"
                for i, s in enumerate(services)
            ]
        )

        # Detect if this is a repair request
        repair_keywords = ["damaged", "broken", "torn", "hole", "rip", "repair", "fix", "mend"]
        is_repair_request = any(keyword in user_input.lower() for keyword in repair_keywords)
        
        repair_instruction = ""
        if is_repair_request:
            repair_instruction = "\nIMPORTANT: The user mentioned damage/repair. Prioritize REPAIR services over cleaning services. Choose repair, alteration, or maintenance services when available."

        prompt = f"""
You are an expert service assistant. A user has requested the following:

User: \"{user_input}\"

Here are 10 available services (each row shows: Business Type | Item Category | Item Name | Service Name | Service Type | Description | Price):

{service_options}{repair_instruction}

Please choose the single most relevant service for this request. Respond only with the number. Example: 4
Only select from the numbers shown above. Do not invent new options or numbers.
"""

        try:
            response = self._call_ollama(prompt).strip()
            print(f"LLM raw response: {response}")

            # Extract numbers from the response using compiled regex
            indices = [int(i) - 1 for i in DIGIT_PATTERN.findall(response)]
            selected = [services[i] for i in indices if 0 <= i < len(services)]
            
            # Fallback if nothing was selected
            if not selected:
                print("No valid services extracted from LLM response.")
                return services[:1]

            # Only return the first valid selection
            selected_service = selected[:1]
            
            # Print detailed information about selected service
            print("Selected services:")
            for s in selected_service:
                print(f"- {s.repairer_type} | {s.category} | {s.garment_type} | {s.service} | {s.service_type} | {s.description} | {s.price}")
            sys.stdout.flush()
            
            return selected_service
        except Exception as e:
            print(f"LLM service selection failed: {e}")
            return services[:1]  # fallback to first if LLM fails

### -------------------------------
### 5. Session Management
### -------------------------------

sessions = {}

class SessionState:
    def __init__(self):
        self.user_name: Optional[str] = None
        self.conversation_state: str = "greeting"  # greeting, searching, selecting, confirming, completed
        self.conversation_history: List[Dict] = []
        self.context: Dict = {}
        self.suggested_services: List[ServiceItem] = []
        self.selected_services: List[ServiceItem] = []
        self.pending_order: Optional[OrderSummary] = None
        self.current_query: Optional[str] = None

def get_session(session_id: str) -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return sessions[session_id]

### -------------------------------
### 6. Service Functions (Using Shared Configuration)
### -------------------------------

def query_fikse_search(query: str) -> List[ServiceItem]:
    """Query the search service for services using shared configuration"""
    try:
        print(f"Searching for: {query}")
        response = requests.get(SEARCH_URL, params={"q": query})
        response.raise_for_status()
        results = response.json()
        print(f"Found {len(results)} raw results")
        
        # Use standardized column names from shared configuration
        priorities = config["search_priorities"]
        price_col = config["price_column"]
        hours_col = config["hours_column"]
        
        services = []
        for i, result in enumerate(results[:10]):
            service_item = ServiceItem(
                id=f"service_{i+1}",
                service=result.get(priorities["exact_service_name"], "Unknown Service"),
                description=result.get(priorities["description"], ""),
                price=float(result.get(price_col, 0)),
                garment_type=result.get(priorities["item_name"], ""),
                repairer_type=result.get(priorities["business_type"], ""),
                estimated_hours=result.get(hours_col, None),
                category=result.get("category", ""),
                service_type=result.get("service_type", "")  # Add service type
            )
            services.append(service_item)
        
        print(f"Returning {len(services)} formatted services")
        print("Available services:")
        for i, service in enumerate(services):
            print(f"  {i+1}. {service.repairer_type} | {service.category} | {service.garment_type} | {service.service} | {service.service_type} | {service.description} | {service.price}")
        sys.stdout.flush()
        
        return services
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

### -------------------------------
### 7. Main Agent Logic
### -------------------------------

ai_generator = AIResponseGenerator()

# Utility to load and cache the TONE_GUIDELINE from the Prompt file
def get_tone_guideline(prompt_path="Prompt"):
    if not hasattr(get_tone_guideline, "_cache"):
        get_tone_guideline._cache = None
        get_tone_guideline._lock = threading.Lock()
    if get_tone_guideline._cache is None:
        with get_tone_guideline._lock:
            if get_tone_guideline._cache is None:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Extract the TONE_GUIDELINE string if present
                if "TONE_GUIDELINE" in content:
                    local_vars = {}
                    exec(content, {}, local_vars)
                    get_tone_guideline._cache = local_vars.get("TONE_GUIDELINE", content)
                else:
                    get_tone_guideline._cache = content
    return get_tone_guideline._cache

def generate_order_comment(user_prompt: str, selected_services: list, ai_generator) -> str:
    TONE_GUIDELINE = get_tone_guideline()  # Cached, loaded once
    
    # Create full service row information for each selected service
    service_rows = []
    for s in selected_services:
        service_row = f"{s.repairer_type} | {s.category} | {s.garment_type} | {s.service} | {s.service_type} | {s.description} | {s.price}"
        service_rows.append(service_row)
    
    selected_services_text = "\n".join(service_rows)
    
    comment_prompt = (
        f"{TONE_GUIDELINE.strip()}\n\n"
        f"Customer: \"{user_prompt}\"\n"
        f"Service: {selected_services_text}\n\n"
        f"Note:"
    )
    
    try:
        llm_response = ai_generator._call_ollama(comment_prompt).strip()
        print(f"Generated comment: {llm_response}")
        sys.stdout.flush()
        
        # Better post-processing for more natural comments
        if 'no additional instructions' in llm_response.lower():
            return 'No additional instructions.'
        
        # Clean up the response
        comment = llm_response.strip()
        
        # Remove common prefixes that LLM might add
        prefixes_to_remove = ['comment:', 'response:', 'answer:']
        for prefix in prefixes_to_remove:
            if comment.lower().startswith(prefix):
                comment = comment[len(prefix):].strip()
        
        # Limit to reasonable length (max 2 sentences, ~50 words)
        sentences = comment.split('.')
        if len(sentences) > 2:
            comment = '. '.join(sentences[:2]) + '.'
        
        # Ensure it's not empty
        if not comment or comment.isspace():
            return 'No additional instructions.'
        
        print(f"Final comment: {comment}")
        sys.stdout.flush()
        return comment
        
    except Exception as e:
        print(f"Comment generation failed: {e}")
        return "No additional instructions."

@app.post("/agent")
def hybrid_agent(input: AgentInput):
    """Main agent endpoint with enhanced context management"""
    try:
        print(f"\n=== NEW REQUEST ===")
        print(f"User input: {input.user_input}")
        print(f"Session ID: {input.session_id}")
        sys.stdout.flush()
        
        session = get_session(input.session_id)
        print(f"Session state: conversation_state={session.conversation_state}, suggested_services={len(session.suggested_services)}, selected_services={len(session.selected_services)}")
        sys.stdout.flush()
        
        # Detect intent with context from previous conversation
        intent_data = detect_intent_and_context(input.user_input)
        intent = intent_data.get("intent", "unknown")
        context = intent_data.get("context", {})
        
        # Maintain context from previous conversation if this is a follow-up
        if session.context and (intent == "repair_request" or intent == "service_selection"):
            # Merge new context with existing context
            for key, value in context.items():
                if value is not None:
                    session.context[key] = value
            context = session.context
        else:
            session.context = context
        
        print(f"Final intent: {intent}, context: {context}")
        
        # Handle different conversation states
        if session.conversation_state == "greeting":
            if intent == "repair_request":
                # User described what they need
                session.current_query = input.user_input
                session.conversation_state = "searching"
                
                # Search for services
                services = query_fikse_search(input.user_input)
                session.suggested_services = services
                
                if services:
                    # Use LLM to select the best service
                    selected_services = ai_generator.select_services_with_llm(input.user_input, services)
                    session.selected_services = selected_services
                    session.conversation_state = "confirming"
                    
                    # Generate order preview
                    total_price = sum(s.price for s in selected_services)
                    total_hours = sum(s.estimated_hours or 0 for s in selected_services)
                    
                    order = OrderSummary(
                        order_id=str(uuid.uuid4()),
                        services=selected_services,
                        total_price=total_price,
                        estimated_total_hours=total_hours if total_hours > 0 else None,
                        created_at=datetime.now().isoformat()
                    )
                    session.pending_order = order
                    
                    # Generate comment using LLM
                    comment = generate_order_comment(input.user_input, selected_services, ai_generator)
                    
                    response_text = f"**Service Found!**\n\n**Selected Service:** {selected_services[0].service}\n**Price:** ${total_price:.0f}\n**Estimated Hours:** {total_hours:.1f}h\n\n**Comment:** {comment}\n\nWould you like to proceed with this service?"
                    
                    return {
                        "intent": intent,
                        "response": response_text,
                        "conversation_state": "confirming",
                        "show_services": False,
                        "order_created": order.dict(),
                        "context": context
                    }
                else:
                    response_text = "I couldn't find any services matching your request. Could you try describing it differently?"
                    return {
                        "intent": intent,
                        "response": response_text,
                        "conversation_state": "greeting",
                        "show_services": False,
                        "context": context
                    }
            elif intent == "greeting":
                response_text = "Hi! How can I help you today?"
                return {
                    "intent": intent,
                    "response": response_text,
                    "conversation_state": "greeting",
                    "show_services": False,
                    "context": context
                }
            else:
                response_text = "I'm not sure how to help with that. Could you describe what item needs service?"
                return {
                    "intent": intent,
                    "response": response_text,
                    "conversation_state": "greeting",
                    "show_services": False,
                    "context": context
                }
        
        elif session.conversation_state == "confirming":
            if intent == "confirmation":
                # User confirmed the order
                if session.pending_order:
                    final_order = session.pending_order
                    order_dict = final_order.dict()
                    
                    response_text = f"**Order Created Successfully!**\n\n**Order ID:** {final_order.order_id}\n**Service:** {session.selected_services[0].service}\n**Price:** ${final_order.total_price:.0f}\n**Created:** {final_order.created_at}\n\nYour service order is ready for processing! Is there anything else I can help you with?"
                    return {
                        "intent": intent,
                        "response": response_text,
                        "conversation_state": "completed",
                        "show_services": False,
                        "order_created": final_order.dict(),
                        "context": context
                    }
            elif intent == "cancel":
                # Reset to service selection
                session.conversation_state = "greeting"
                session.selected_services = []
                session.pending_order = None
                session.context = {}  # Clear context
                print("Order cancelled, reset to greeting state")
                response_text = "Order cancelled. What item would you like to get serviced?"
                
                return {
                    "intent": intent,
                    "response": response_text,
                    "conversation_state": "greeting",
                    "show_services": False,
                    "context": context
                }
            elif intent == "repair_request":
                # User is providing additional details or changing the request
                session.current_query = input.user_input
                session.conversation_state = "searching"
                
                # Search with new query
                services = query_fikse_search(input.user_input)
                session.suggested_services = services
                
                if services:
                    selected_services = ai_generator.select_services_with_llm(input.user_input, services)
                    session.selected_services = selected_services
                    session.conversation_state = "confirming"
                    
                    total_price = sum(s.price for s in selected_services)
                    total_hours = sum(s.estimated_hours or 0 for s in selected_services)
                    
                    order = OrderSummary(
                        order_id=str(uuid.uuid4()),
                        services=selected_services,
                        total_price=total_price,
                        estimated_total_hours=total_hours if total_hours > 0 else None,
                        created_at=datetime.now().isoformat()
                    )
                    session.pending_order = order
                    
                    comment = generate_order_comment(input.user_input, selected_services, ai_generator)
                    
                    response_text = f"**Updated Service Found!**\n\n**Selected Service:** {selected_services[0].service}\n**Price:** ${total_price:.0f}\n**Estimated Hours:** {total_hours:.1f}h\n\n**Comment:** {comment}\n\nWould you like to proceed with this service?"
                    
                    return {
                        "intent": intent,
                        "response": response_text,
                        "conversation_state": "confirming",
                        "show_services": False,
                        "order_created": order.dict(),
                        "context": context
                    }
                else:
                    response_text = "I couldn't find any services for that. Could you try describing it differently?"
                    return {
                        "intent": intent,
                        "response": response_text,
                        "conversation_state": "confirming",
                        "show_services": False,
                        "context": context
                    }
            else:
                response_text = "I'm not sure how to help with that. Could you describe what item needs service?"
                return {
                    "intent": intent,
                    "response": response_text,
                    "conversation_state": "confirming",
                    "show_services": False,
                    "context": context
                }
        
        # Handle cancellation
        elif intent == "cancel":
            if session.conversation_state == "confirming":
                # Reset to service selection
                session.conversation_state = "greeting"
                session.selected_services = []
                session.pending_order = None  # Clear the order preview
                session.context = {}  # Clear context
                print("Order cancelled, reset to greeting state")
                response_text = "Order cancelled. What item would you like to get serviced?"
                
                return {
                    "intent": intent,
                    "response": response_text,
                    "conversation_state": "greeting",
                    "show_services": False,
                    "context": context
                }
            else:
                # General reset
                session.conversation_state = "greeting"
                session.context = {}
                response_text = "No problem! What item would you like to get serviced?"
                
                return {
                    "intent": intent,
                    "response": response_text,
                    "conversation_state": "greeting",
                    "show_services": False,
                    "context": context
                }
        
        # Handle greetings with direct response
        elif intent == "greeting":
            session.conversation_state = "greeting"
            session.context = {}  # Clear context
            response_text = "Hi! How can I help you today?"
            
            return {
                "intent": intent,
                "response": response_text,
                "conversation_state": "greeting",
                "show_services": False,
                "context": context
            }
        
        # Handle unknown or other intents with a simple response
        else:
            response_text = "I'm not sure how to help with that. Could you describe what item needs service?"
            
            return {
                "intent": intent,
                "response": response_text,
                "conversation_state": session.conversation_state,
                "show_services": False,
                "context": context
            }
    
    except Exception as e:
        return {
            "intent": "error",
            "response": "I apologize, but I'm having trouble right now. Please describe what item needs service.",
            "conversation_state": "greeting",
            "show_services": False,
            "error": str(e)
        }

@app.get("/")
def root():
    return {"message": "AI-Powered Service Agent", "specialty": "general service and repair services"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Agent is running"}

@app.post("/detect_intent")
def detect_intent_endpoint(input: AgentInput):
    """Test endpoint for intent detection"""
    try:
        intent_data = detect_intent_and_context(input.user_input)
        return {
            "input": input.user_input,
            "intent": intent_data.get("intent"),
            "context": intent_data.get("context"),
            "confidence": intent_data.get("confidence", "high")
        }
    except Exception as e:
        return {"error": str(e)} 