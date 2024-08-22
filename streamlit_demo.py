import streamlit as st
from chatbot_clone import Chatbot, llamaLLM, Llama4bitLLM, LLMInterface
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'current_llm' not in st.session_state:
    st.session_state.current_llm = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Available LLMs
LLM_OPTIONS = {
    "llamaLLM": llamaLLM,
    "Llama4bitLLM": Llama4bitLLM
}

# Function to create a new chat
def create_new_chat(initial_message=None):
    new_chat_id = str(uuid.uuid4())
    st.session_state.chatbot.create_chat(new_chat_id, st.session_state.user_id)
    st.session_state.current_chat_id = new_chat_id
    if initial_message:
        st.session_state.chatbot.chat(new_chat_id, st.session_state.user_id, initial_message)
    return new_chat_id

# Chat Info page (formerly System Prompt page)
def chat_info_page():
    st.title("Chat Information")
    
    if st.session_state.current_chat_id:
        chat = st.session_state.chatbot.get_chat(st.session_state.current_chat_id, st.session_state.user_id)
        
        # Chat Name Section
        st.header("Chat Name")
        current_name = chat.name
        new_name = st.text_input("Chat Name", value=current_name)
        if st.button("Update Chat Name"):
            st.session_state.chatbot.set_name(st.session_state.user_id, st.session_state.current_chat_id, new_name)
            st.success("Chat name updated successfully!")
        
        # System Prompt Section
        st.header("System Prompt")
        current_prompt = chat.system_prompt
        new_prompt = st.text_area("System Prompt", value=current_prompt, height=200)
        if st.button("Update System Prompt"):
            st.session_state.chatbot.set_system_prompt(st.session_state.user_id, st.session_state.current_chat_id, new_prompt)
            st.success("System prompt updated successfully!")
        
        # Chat Statistics
        st.header("Chat Statistics")
        st.write(f"Chat ID: {chat.chat_id}")
        st.write(f"Message Count: {len(chat.full_conversation_history)}")
        st.write(f"Running Summary: {chat.running_summary}")
        st.write(f"Recent Messages: {chat.recent_messages}")
        st.write(f"Full Conversation History: {chat.full_conversation_history}")
        #st.write(f"Created: {chat.created_at}")  # Assuming you have a created_at attribute
        #st.write(f"Last Updated: {chat.updated_at}")  # Assuming you have an updated_at attribute
    else:
        st.warning("Please select or create a chat first.")
    
    if st.button("Back to Chat"):
        st.session_state.page = 'home'
        st.experimental_rerun()

# Login page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Here you should implement proper authentication
        # For this example, we'll just use the username as the user_id
        st.session_state.user_id = username
        st.session_state.current_llm = llamaLLM()
        st.session_state.chatbot = Chatbot(st.session_state.current_llm, st.session_state.user_id)
        st.success("Logged in successfully!")
        st.experimental_rerun()

# Homepage
def homepage():
    st.title("Chat Application")
    
    # Sidebar
    with st.sidebar:
        st.header("User Information")
        st.write(f"Signed in as: {st.session_state.user_id}")
        if st.button("Change User"):
            st.session_state.user_id = None
            st.session_state.chatbot = None
            st.session_state.current_chat_id = None
            st.session_state.current_llm = None
            st.experimental_rerun()
                
        st.header("Current Chat")
        if st.session_state.current_chat_id:
            chat = st.session_state.chatbot.get_chat(st.session_state.current_chat_id, st.session_state.user_id)
            st.write(f"Chat ID: {st.session_state.current_chat_id[:8]}...")
            st.write(f"Chat Name: {chat.name}")
            message_count = len(chat.full_conversation_history)
            st.write(f"Messages: {message_count}")
        else:
            st.write("No chat selected")
            
        st.header("LLM Selection")
        current_llm_name = st.session_state.current_llm.__class__.__name__
        selected_llm = st.selectbox(
            "Choose LLM",
            list(LLM_OPTIONS.keys()),
            index=list(LLM_OPTIONS.keys()).index(current_llm_name)
        )
        if selected_llm != current_llm_name:
            st.session_state.current_llm = LLM_OPTIONS[selected_llm]()
            st.session_state.chatbot.set_llm(st.session_state.current_llm)
            st.success(f"Changed LLM to {selected_llm}")
            st.experimental_rerun()
            
        st.header("System Prompt")
        if st.session_state.current_chat_id:
            chat = st.session_state.chatbot.get_chat(st.session_state.current_chat_id, st.session_state.user_id)
            st.write(chat.system_prompt)
        else:
            st.write("No chat selected")
            
        st.header("Chat Actions")
        if st.button("Chat Info"):
            st.session_state.page = 'chat_info'
            st.experimental_rerun()
        
            
            
        st.header("Create New Chat")
        if st.button("New Chat"):
            create_new_chat()
            st.experimental_rerun()
    
        st.header("Previous Chats")
        chat_summaries = st.session_state.chatbot.get_user_chat_summaries()
        for chat in chat_summaries:
            if st.button(f"{chat['name']} ({chat['chat_id']})"):
                st.session_state.current_chat_id = chat['chat_id']
                st.experimental_rerun()
        
    # Main chat area
    if st.session_state.current_chat_id:
        chat = st.session_state.chatbot.get_chat(st.session_state.current_chat_id, st.session_state.user_id)
        st.header(f"{chat.name} (Chat {st.session_state.current_chat_id[:8]})")
        
        # Display chat history
        for message in chat.full_conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        st.header("Start a New Chat")
        st.write("Type your message below to start a new chat.")
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        if not st.session_state.current_chat_id:
            create_new_chat(user_input)
        else:
            st.session_state.chatbot.chat(st.session_state.current_chat_id, st.session_state.user_id, user_input)
            print(st.session_state.chatbot.llm)
        st.experimental_rerun()
        
        
# Main app logic
def main():
    if st.session_state.user_id is None:
        login_page()
    elif st.session_state.page == 'chat_info':
        chat_info_page()
    else:
        homepage()

if __name__ == "__main__":
    main()