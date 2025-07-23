import json

# Mock agent_prototype module as a simple object for this test
import types
agent_prototype = types.SimpleNamespace()

# In-memory sessions storage
_sessions = {}

# Mock ServiceItem class
class ServiceItem:
    def __init__(self, service, description, price, garment_type, repairer_type, estimated_hours):
        self.service = service
        self.description = description
        self.price = price
        self.garment_type = garment_type
        self.repairer_type = repairer_type
        self.estimated_hours = estimated_hours

    def dict(self):
        return self.__dict__

# Mock Session class to hold session data attributes
class Session:
    def __init__(self):
        self.user_name = ""
        self.conversation_state = ""
        self.selected_services = []
        self.suggested_services = []
        self.current_query = ""

# Function to get or create session by ID
def get_session(session_id: str):
    if session_id not in _sessions:
        _sessions[session_id] = Session()
    return _sessions[session_id]

# Inject mocks into agent_prototype namespace
agent_prototype.get_session = get_session
agent_prototype.ServiceItem = ServiceItem

# Implement store_session_data and get_session_data using get_session
def store_session_data(session_id, key, value):
    session = agent_prototype.get_session(session_id)
    setattr(session, key, value)

def get_session_data(session_id, key):
    session = agent_prototype.get_session(session_id)
    return getattr(session, key, None)

agent_prototype.store_session_data = store_session_data
agent_prototype.get_session_data = get_session_data

# The test function
def test_session_persistence():
    session_id = "test-session"

    # Store session data
    agent_prototype.store_session_data(session_id, "user_name", "Alice")
    agent_prototype.store_session_data(session_id, "conversation_state", "selecting")

    services = [{
        "service": "Fix zipper",
        "description": "Repair zipper",
        "price": 15.0,
        "garment_type": "jacket",
        "repairer_type": "zip specialist",
        "estimated_hours": 1
    }]
    # Store selected_services as JSON string
    agent_prototype.store_session_data(session_id, "selected_services", json.dumps(services))

    # Retrieve session data
    user_name = agent_prototype.get_session_data(session_id, "user_name")
    conversation_state = agent_prototype.get_session_data(session_id, "conversation_state")
    selected_services_json = agent_prototype.get_session_data(session_id, "selected_services")
    selected_services = json.loads(selected_services_json)

    # Print results
    print("Stored and Retrieved Session Data:")
    print(f"user_name: {user_name}")
    print(f"conversation_state: {conversation_state}")
    print(f"selected_services: {selected_services}")

    # Assertions with feedback
    try:
        assert user_name == "Alice"
        assert conversation_state == "selecting"
        assert selected_services[0]["service"] == "Fix zipper"
        assert selected_services[0]["price"] == 15.0
        print("\nTest Passed: Session data persisted correctly.")
    except AssertionError:
        print("\nTest Failed: Session data did not persist correctly.")
        raise

if __name__ == "__main__":
    test_session_persistence()
