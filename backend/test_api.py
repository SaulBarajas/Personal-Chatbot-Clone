import requests
import json

BASE_URL = "http://localhost:8502"

def print_response(response):
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    print("\n")

# Test login
print("Testing login...")
response = requests.post(f"{BASE_URL}/login", params={"username": "testuser"})
print_response(response)
user_id = response.json()["user_id"]

# Test create chat
print("Testing create chat...")
create_chat_data = {"role": "user", "content": "Hello, can you help me with a coding problem?"}
response = requests.post(f"{BASE_URL}/create_chat", json=create_chat_data, params={"user_id": user_id})
print_response(response)
chat_id = response.json()["chat_id"]

# Test chat
print("Testing chat...")
chat_data = {"role": "user", "content": "How do I reverse a string in Python?"}
response = requests.post(f"{BASE_URL}/chat/{chat_id}", json=chat_data, params={"user_id": user_id})
print_response(response)

# Test get chat summaries
print("Testing get chat summaries...")
response = requests.get(f"{BASE_URL}/chat_summaries", params={"user_id": user_id})
print_response(response)

# Test get chat history
print("Testing get chat history...")
response = requests.get(f"{BASE_URL}/chat_history/{chat_id}", params={"user_id": user_id})
print_response(response)

# Test edit message
print("Testing edit message...")
edit_data = {"role": "user", "content": "How do I reverse a string in Python efficiently?"}
node_id = response.json()["messages"][2]["id"]  # Assuming this is the message we want to edit
response = requests.put(f"{BASE_URL}/edit_message/{chat_id}/{node_id}", json=edit_data, params={"user_id": user_id})
print_response(response)

# Test get chat history
print("Testing get chat history 2...")
response = requests.get(f"{BASE_URL}/chat_history/{chat_id}", params={"user_id": user_id})
print_response(response)

# Test switch edit
print("Testing switch edit...")
response = requests.put(f"{BASE_URL}/switch_edit/{chat_id}/{node_id}", params={"user_id": user_id})
print_response(response)

# Test get chat history
print("Testing get chat history 3...")
response = requests.get(f"{BASE_URL}/chat_history/{chat_id}", params={"user_id": user_id})
print_response(response)

# Test get node edits
print("Testing get node edits...")
response = requests.get(f"{BASE_URL}/node_edits/{chat_id}/{node_id}", params={"user_id": user_id})
print_response(response)

# Test set LLM
print("Testing set LLM...")
response = requests.put(f"{BASE_URL}/set_llm", params={"llm_name": "llama8bLLM", "user_id": user_id})
print_response(response)

# Test set system prompt
print("Testing set system prompt...")
response = requests.put(f"{BASE_URL}/set_system_prompt/{chat_id}", params={"prompt": "You are a helpful coding assistant.", "user_id": user_id})
print_response(response)

# Test set chat name
print("Testing set chat name...")
response = requests.put(f"{BASE_URL}/set_chat_name/{chat_id}", params={"name": "Python Coding Help", "user_id": user_id})
print_response(response)


# Test get running summary
print("Testing get running summary...")
response = requests.get(f"{BASE_URL}/running_summary/{chat_id}", params={"user_id": user_id})
print_response(response)