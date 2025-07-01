# chat_ui.py

import streamlit as st
import requests
import uuid
import json
from typing import Dict, List, Optional

st.set_page_config(
    page_title="FikseAgent - AI Tailor Assistant", 
    page_icon="🧵",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .service-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #1f77b4;
    }
    .selected-service {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
    }
    .order-summary {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .order-created {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧵 FikseAgent")
st.markdown("*Streamline your repair order creation with AI assistance*")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_services" not in st.session_state:
    st.session_state.current_services = []
if "selected_services" not in st.session_state:
    st.session_state.selected_services = []
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "greeting"
if "pending_order" not in st.session_state:
    st.session_state.pending_order = None

# Function to send message to agent
def send_agent_message(message: str):
    try:
        with st.spinner("🤔 Thinking..."):
            response = requests.post("http://localhost:8001/agent", json={
                "session_id": st.session_state.session_id,
                "user_input": message
            })
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ **Connection Error**: Make sure the agent service is running on port 8001")
        return None
    except Exception as e:
        st.error(f"❌ **Error**: {str(e)}")
        return None

# Function to handle service selection
def handle_service_selection(service_index):
    response_data = send_agent_message(str(service_index + 1))
    if response_data:
        # Update session state
        st.session_state.conversation_state = response_data.get("conversation_state", "greeting")
        st.session_state.current_services = response_data.get("services", [])
        st.session_state.selected_services = response_data.get("selected_services", [])
        st.session_state.pending_order = response_data.get("order_summary")
        
        # Add agent response to chat history
        message_data = {
            "role": "assistant", 
            "content": response_data.get("response", ""),
            "services": response_data.get("services", []),
            "selected_services": response_data.get("selected_services", []),
            "order_summary": response_data.get("order_summary"),
            "order_created": response_data.get("order_created")
        }
        st.session_state.chat_history.append(message_data)
        st.rerun()

# Function to handle order confirmation
def handle_order_action(action):
    response_data = send_agent_message(action)
    if response_data:
        # Update session state
        st.session_state.conversation_state = response_data.get("conversation_state", "greeting")
        st.session_state.current_services = response_data.get("services", [])
        st.session_state.selected_services = response_data.get("selected_services", [])
        st.session_state.pending_order = response_data.get("order_summary")
        
        # Add agent response to chat history
        message_data = {
            "role": "assistant", 
            "content": response_data.get("response", ""),
            "services": response_data.get("services", []),
            "selected_services": response_data.get("selected_services", []),
            "order_summary": response_data.get("order_summary"),
            "order_created": response_data.get("order_created")
        }
        st.session_state.chat_history.append(message_data)
        st.rerun()

# Sidebar with session info and controls
with st.sidebar:
    st.header("Session Info")
    st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    st.write(f"**State:** {st.session_state.conversation_state}")
    
    if st.button("🔄 Reset Session"):
        try:
            response = requests.delete(f"http://localhost:8001/agent/session/{st.session_state.session_id}")
            st.session_state.chat_history = []
            st.session_state.current_services = []
            st.session_state.selected_services = []
            st.session_state.conversation_state = "greeting"
            st.session_state.pending_order = None
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting session: {e}")
    
    # Show current services if any
    if st.session_state.current_services:
        st.header("Available Services")
        for i, service in enumerate(st.session_state.current_services, 1):
            st.write(f"{i}. {service['service']} - ${service['price']:.0f}")
    
    # Show selected services if any
    if st.session_state.selected_services:
        st.header("Selected Services")
        total_price = sum(service['price'] for service in st.session_state.selected_services)
        for service in st.session_state.selected_services:
            st.write(f"• {service['service']} - ${service['price']:.0f}")
        st.write(f"**Total: ${total_price:.0f}**")

# Main chat interface
st.header("💬 Chat with FikseAgent")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display services if included in the message (for history only)
        if message.get("services") and st.session_state.conversation_state != "selecting":
            st.subheader("Available Services:")
            for i, service in enumerate(message["services"], 1):
                with st.expander(f"{i}. {service['service']} - ${service['price']:.0f}"):
                    st.write(f"**Description:** {service['description']}")
                    st.write(f"**Garment Type:** {service['garment_type']}")
                    st.write(f"**Repairer:** {service['repairer_type']}")
                    if service.get('estimated_hours'):
                        st.write(f"**Estimated Time:** {service['estimated_hours']} hours")
        
        # Display selected services if included
        if message.get("selected_services"):
            st.subheader("Selected Services:")
            total = sum(service['price'] for service in message["selected_services"])
            for service in message["selected_services"]:
                st.markdown(f"✅ **{service['service']}** - ${service['price']:.0f}")
            st.markdown(f"**Subtotal: ${total:.0f}**")
        
        # Display order summary if included
        if message.get("order_summary"):
            order = message["order_summary"]
            st.markdown("### 📋 Order Summary")
            st.markdown(f"**Total Price:** ${order['total_price']:.0f}")
            if order.get('estimated_total_hours'):
                st.markdown(f"**Estimated Time:** {order['estimated_total_hours']:.1f} hours")
            
            with st.expander("View All Services"):
                for service in order['services']:
                    st.markdown(f"• {service['service']} - ${service['price']:.0f}")
        
        # Display created order if included
        if message.get("order_created"):
            order = message["order_created"]
            st.markdown("### 🎉 Order Created!")
            st.markdown(f"**Order ID:** `{order['order_id']}`")
            st.markdown(f"**Created:** {order['created_at']}")
            st.markdown(f"**Total Price:** ${order['total_price']:.0f}")
            
            if order.get('estimated_total_hours'):
                st.markdown(f"**Estimated Time:** {order['estimated_total_hours']:.1f} hours")
            
            with st.expander("View Order Details"):
                for service in order['services']:
                    st.markdown(f"• {service['service']} - ${service['price']:.0f}")
                    st.markdown(f"  *{service['description']}*")
            
            # Option to download order as JSON
            st.download_button(
                label="📥 Download Order Details",
                data=json.dumps(order, indent=2),
                file_name=f"order_{order['order_id']}.json",
                mime="application/json"
            )

# Service selection section (appears when in selecting state)
if st.session_state.conversation_state == "selecting" and st.session_state.current_services:
    st.subheader("🛠️ Available Services - Click to Select:")
    
    # Display service buttons
    for i, service in enumerate(st.session_state.current_services):
        col1, col2 = st.columns([3, 1])
        with col1:
            service_text = f"**{service['service']}** - ${service['price']:.0f}"
            if service['description']:
                service_text += f"\n*{service['description']}*"
            st.markdown(service_text)
        with col2:
            if st.button(f"Select", key=f"select_service_{i}", type="primary"):
                handle_service_selection(i)

# Order confirmation section (appears when in confirming state)
if st.session_state.conversation_state == "confirming" and st.session_state.pending_order:
    st.subheader("📋 Order Confirmation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm Order", key="confirm_order_main", type="primary"):
            handle_order_action("Yes")
    with col2:
        if st.button("❌ Cancel Order", key="cancel_order_main"):
            handle_order_action("Cancel")

# Manual service addition section (appears when in manual_addition state)
if st.session_state.conversation_state == "manual_addition":
    st.subheader("🔧 Additional Services")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add More Services", key="add_more_services", type="primary"):
            handle_order_action("Yes")
    with col2:
        if st.button("➡️ Proceed to Order", key="proceed_to_order"):
            handle_order_action("No")

# User input (only show when not in selection states)
if st.session_state.conversation_state not in ["selecting", "confirming", "manual_addition"]:
    user_input = st.chat_input("Describe the garment and the issue...")
    
    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Call the agent API
        response_data = send_agent_message(user_input)
        if response_data:
            # Update session state
            st.session_state.conversation_state = response_data.get("conversation_state", "greeting")
            st.session_state.current_services = response_data.get("services", [])
            st.session_state.selected_services = response_data.get("selected_services", [])
            st.session_state.pending_order = response_data.get("order_summary")
            
            # Add agent response to chat history
            message_data = {
                "role": "assistant", 
                "content": response_data.get("response", ""),
                "services": response_data.get("services", []),
                "selected_services": response_data.get("selected_services", []),
                "order_summary": response_data.get("order_summary"),
                "order_created": response_data.get("order_created")
            }
            st.session_state.chat_history.append(message_data)
            st.rerun()

# Status indicator
if st.session_state.conversation_state == "greeting":
    st.info("👋 Ready to help! Describe what needs repair.")
elif st.session_state.conversation_state == "selecting":
    st.info("🛠️ Select a service from the options above.")
elif st.session_state.conversation_state == "manual_addition":
    st.info("🔧 Would you like to add more services or proceed with your order?")
elif st.session_state.conversation_state == "confirming":
    st.info("✅ Review your order above and click confirm to proceed.")
elif st.session_state.conversation_state == "completed":
    st.success("🎉 Order created successfully! Start a new conversation for another order.")
