#!/usr/bin/env python3
"""
FikseAgent System Startup Script

This script starts both the search API and agent API services,
ensuring proper initialization and dependency checking.
"""

import subprocess
import time
import sys
import requests
import signal
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# Global process references for cleanup
search_process = None
agent_process = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down FikseAgent system...")
    
    if agent_process:
        print("  Stopping Agent API...")
        agent_process.terminate()
        agent_process.wait()
    
    if search_process:
        print("  Stopping Search API...")
        search_process.terminate()
        search_process.wait()
    
    print("✅ System shutdown complete")
    sys.exit(0)

def check_port_available(port):
    """Check if a port is available"""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=2)
        return False  # Port is occupied
    except requests.exceptions.RequestException:
        return True  # Port is available or service not responding

def wait_for_search_service(max_attempts=30):
    """Wait for search service to become available (using /search endpoint)"""
    print(f"⏳ Waiting for Search API to start on port 8000...")
    
    for attempt in range(max_attempts):
        try:
            # Try the search endpoint with a simple query
            response = requests.get("http://localhost:8000/search", params={"q": "test"}, timeout=2)
            if response.status_code == 200:
                print(f"✅ Search API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        if attempt % 5 == 0:
            print(f"   Still waiting... ({attempt + 1}/{max_attempts})")
    
    print(f"❌ Search API failed to start within {max_attempts} seconds")
    return False

def wait_for_service(port, service_name, max_attempts=30):
    """Wait for a service to become available"""
    print(f"⏳ Waiting for {service_name} to start on port {port}...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        if attempt % 5 == 0:
            print(f"   Still waiting... ({attempt + 1}/{max_attempts})")
    
    print(f"❌ {service_name} failed to start within {max_attempts} seconds")
    return False

def start_search_api():
    """Start the search API service"""
    global search_process
    
    print("🔍 Starting Fikse Search API (port 8000)...")
    
    if not check_port_available(8000):
        print("⚠️  Port 8000 is already in use. Trying to use existing service...")
        try:
            response = requests.get("http://localhost:8000/search", params={"q": "test"}, timeout=5)
            if response.status_code == 200:
                print("✅ Using existing Search API service")
                return True
        except:
            print("❌ Port 8000 is occupied by non-compatible service")
            return False
    
    try:
        search_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return wait_for_search_service()
        
    except Exception as e:
        print(f"❌ Failed to start Search API: {e}")
        return False

def start_agent_api():
    """Start the agent API service"""
    global agent_process
    
    print("🤖 Starting Fikse Agent API (port 8001)...")
    
    if not check_port_available(8001):
        print("⚠️  Port 8001 is already in use. Trying to use existing service...")
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print("✅ Using existing Agent API service")
                return True
        except:
            print("❌ Port 8001 is occupied by non-compatible service")
            return False
    
    try:
        agent_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "agent:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return wait_for_service(8001, "Agent API")
        
    except Exception as e:
        print(f"❌ Failed to start Agent API: {e}")
        return False

def start_chat_ui():
    """Start the Streamlit chat UI"""
    print("💬 Starting Chat UI (port 8501)...")
    
    try:
        # Run streamlit in a separate process
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "chat_ui.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        
        print("✅ Chat UI should be starting at http://localhost:8501")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start Chat UI: {e}")
        return False

def check_dependencies():
    """Check if all required files and dependencies exist"""
    print("🔍 Checking system dependencies...")
    
    required_files = ["app.py", "agent.py", "chat_ui.py", "Dataset_categories.csv", "faiss.index"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check if precomputed dataset exists
    if not os.path.exists("precomputed_dataset"):
        print("❌ Missing precomputed_dataset directory")
        print("   Run: python precompute_dataset.py")
        return False
    
    print("✅ All dependencies found")
    return True

def show_system_status():
    """Show the status of all services"""
    print("\n" + "="*60)
    print("🎯 FIKSE AGENT SYSTEM STATUS")
    print("="*60)
    
    services = [
        ("Search API", "http://localhost:8000/search?q=test"),
        ("Agent API", "http://localhost:8001/health"),
        ("Chat UI", "http://localhost:8501")
    ]
    
    for service_name, url in services:
        try:
            if "8501" in url:  # Streamlit doesn't have health endpoint
                print(f"💬 {service_name:<15} - Available at http://localhost:8501")
            elif "8000" in url:  # Search API - test with search endpoint
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    print(f"✅ {service_name:<15} - Running at http://localhost:8000")
                else:
                    print(f"❌ {service_name:<15} - Error: {response.status_code}")
            else:  # Agent API - has health endpoint
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    print(f"✅ {service_name:<15} - Running at http://localhost:8001")
                else:
                    print(f"❌ {service_name:<15} - Error: {response.status_code}")
        except requests.exceptions.RequestException:
            print(f"❌ {service_name:<15} - Not responding")
    
    print("="*60)
    print("🌐 Access the Chat UI at: http://localhost:8501")
    print("📚 Search API docs at: http://localhost:8000/docs")
    print("🤖 Agent API docs at: http://localhost:8001/docs")
    print("="*60)

def main():
    """Main startup sequence"""
    print("🚀 Starting FikseAgent System...")
    print("="*50)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Dependency check failed. Please resolve issues and try again.")
        sys.exit(1)
    
    # Start services in sequence
    print("\n📋 Starting services...")
    
    # 1. Start Search API first (Agent depends on it)
    if not start_search_api():
        print("❌ Failed to start Search API. Aborting.")
        sys.exit(1)
    
    # 2. Start Agent API
    if not start_agent_api():
        print("❌ Failed to start Agent API. Aborting.")
        sys.exit(1)
    
    # 3. Start Chat UI
    if not start_chat_ui():
        print("⚠️  Chat UI failed to start, but APIs are running")
    
    # Show system status
    time.sleep(2)  # Give services a moment to fully initialize
    show_system_status()
    
    print("\n🎉 FikseAgent system is ready!")
    print("📝 To test the system:")
    print("   1. Open http://localhost:8501 in your browser")
    print("   2. Try: 'I have a silk dress with a small tear'")
    print("   3. Select services by typing numbers like '1, 3'")
    print("   4. Follow the conversation to create an order")
    
    print("\n⏹️  Press Ctrl+C to stop all services")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main() 