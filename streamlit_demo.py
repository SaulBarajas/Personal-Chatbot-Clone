import streamlit as st
from chatbot_clone import Chatbot, llama8bLLM, Llama4bitLLM, llama70bLLM, llama405bLLM, LLMInterface
import os
from dotenv import load_dotenv
import uuid
import graphviz

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
    "llama8bLLM": llama8bLLM,
    "llama70bLLM": llama70bLLM,
    "llama405bLLM": llama405bLLM,
    "Llama4bitLLM": Llama4bitLLM
}

# Helper Function to create a new chat
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
        st.write(f"Message Count: {len(chat.get_full_conversation_history())}")
        st.write(f"Running Summary: {chat.running_summary}")
        st.write(f"Recent Messages: {chat.recent_messages}")
        st.write(f"Full Conversation History: {chat.get_full_conversation_history()}")
        #st.write(f"Created: {chat.created_at}")  # Assuming you have a created_at attribute
        #st.write(f"Last Updated: {chat.updated_at}")  # Assuming you have an updated_at attribute
    else:
        st.warning("Please select or create a chat first.")
    
    if st.button("Back to Chat"):
        st.session_state.page = 'home'
        st.experimental_rerun()

# Login page gets username and starts the chatbot
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Here you should implement proper authentication
        # For this example, we'll just use the username as the user_id
        st.session_state.user_id = username
        st.session_state.current_llm = llama8bLLM()
        st.session_state.chatbot = Chatbot(st.session_state.current_llm, st.session_state.user_id)
        st.success("Logged in successfully!")
        st.experimental_rerun()

# Add this function to handle message editing
def edit_message(chat_id, node_id, new_content):
    chat = st.session_state.chatbot.get_chat(chat_id, st.session_state.user_id)
    chat.edit_message(node_id, new_content)
    st.success("Message edited successfully!")

def create_conversation_graph(chat):
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')

    def add_node_and_edges(node):
        if node is None:
            return

        node_id = node.id
        label = f"{node.role}: {node.content[:20]}..."
        graph.node(node_id, label)

        # Add vertical edges (parent-child relationships)
        if node.parent:
            graph.edge(node.parent.id, node_id)

        # Add horizontal edges (edit relationships)
        if node.prev_edit:
            graph.edge(node.prev_edit.id, node_id, color="blue", constraint="false")
        if node.next_edit:
            graph.edge(node_id, node.next_edit.id, color="blue", constraint="false")

        # Recursively add children
        for child in node.children:
            add_node_and_edges(child)

    # Add all nodes and edges
    for node in chat.node_map.values():
        add_node_and_edges(node)

    # Highlight the current conversation path
    current_conversation = chat.get_current_conversation()
    for message in current_conversation:
        graph.node(message["id"], color="darkgreen", style="filled", fillcolor="#90EE90", fontcolor="black")

    return graph

def display_conversation_graph():
    st.title("Conversation Graph")
    
    if st.session_state.current_chat_id:
        chat = st.session_state.chatbot.get_chat(st.session_state.current_chat_id, st.session_state.user_id)
        
        graph = create_conversation_graph(chat)
        
        # Render the graph
        st.graphviz_chart(graph)
        
        if st.button("Back to Chat"):
            st.session_state.page = 'home'
            st.experimental_rerun()
    else:
        st.warning("Please select or create a chat first.")

# Add this new function for the new chat creation page
def new_chat_page():
    st.title("Create New Chat")
    
    # Add a back button
    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.experimental_rerun()
    
    initial_message = st.text_input("Enter your first message to start the chat:")
    if st.button("Start Chat"):
        if initial_message:
            new_chat_id = str(uuid.uuid4())
            st.session_state.chatbot.create_chat(new_chat_id, st.session_state.user_id)
            chat = st.session_state.chatbot.get_chat(new_chat_id, st.session_state.user_id)
            chat.add_message("user", initial_message)
            chat.generate_name(initial_message)
            response = chat.get_response()
            st.session_state.chatbot.save_chat(new_chat_id, st.session_state.user_id)
            st.session_state.current_chat_id = new_chat_id
            st.session_state.page = 'home'
            st.experimental_rerun()
        else:
            st.warning("Please enter a message to start the chat.")

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
        
        # Add this button at the top of the sidebar
        if st.button("Show Current Graph"):
            st.session_state.page = 'graph'
            st.experimental_rerun()
        
        st.header("Current Chat")
        if st.session_state.current_chat_id:
            chat = st.session_state.chatbot.get_chat(st.session_state.current_chat_id, st.session_state.user_id)
            st.write(f"Chat ID: {st.session_state.current_chat_id[:8]}...")
            st.write(f"Chat Name: {chat.name}")
            message_count = len(chat.get_full_conversation_history())
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
            st.session_state.page = 'new_chat'
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
        
        # Get the full conversation history
        full_conversation = chat.get_current_conversation()
        
        # Display chat history with edit buttons and version arrows
        for message in full_conversation:
            with st.chat_message(message["role"]):
                # Create an empty container for each message
                message_container = st.empty()
                
                # Get all edits for this message
                edits = chat.get_node_edits(message["id"])
                current_edit_index = message["current_edit_index"]
                
                # Create arrow buttons for version navigation
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    left_disabled = current_edit_index == 0
                    if st.button("←", key=f"left_{message['id']}", disabled=left_disabled):
                        if current_edit_index > 0:
                            new_node_id = edits[current_edit_index - 1].id
                            try:
                                chat.switch_to_edit(new_node_id)
                                st.session_state.chatbot.save_chat(st.session_state.current_chat_id, st.session_state.user_id)
                                st.experimental_rerun()
                            except ValueError as e:
                                st.error(f"Error switching to previous edit: {str(e)}")
                
                with col2:
                    message_container.write(message["content"])
                
                with col3:
                    right_disabled = current_edit_index == len(edits) - 1
                    if st.button("→", key=f"right_{message['id']}", disabled=right_disabled):
                        if current_edit_index < len(edits) - 1:
                            new_node_id = edits[current_edit_index + 1].id
                            try:
                                chat.switch_to_edit(new_node_id)
                                st.session_state.chatbot.save_chat(st.session_state.current_chat_id, st.session_state.user_id)
                                st.experimental_rerun()
                            except ValueError as e:
                                st.error(f"Error switching to next edit: {str(e)}")
                
                # Display edit button for user messages
                if message["role"] == "user":
                    if st.button("Edit", key=f"edit_{message['id']}"):
                        st.session_state.editing_message = message["id"]
                        st.experimental_rerun()

                # Display edit form if editing
                if 'editing_message' in st.session_state and st.session_state.editing_message == message["id"]:
                    with message_container.container():
                        new_content = st.text_input("Edit your message", value=message["content"], key=f"edit_input_{message['id']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Update", key=f"update_{message['id']}"):
                                try:
                                    st.session_state.chatbot.edit_message(st.session_state.current_chat_id, st.session_state.user_id, message["id"], new_content)
                                    del st.session_state.editing_message
                                    st.experimental_rerun()
                                except ValueError as e:
                                    st.error(f"Error updating message: {str(e)}")
                        with col2:
                            if st.button("Cancel", key=f"cancel_{message['id']}"):
                                del st.session_state.editing_message
                                st.experimental_rerun()

    # Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        if not st.session_state.current_chat_id:
            new_chat_id = str(uuid.uuid4())
            st.session_state.chatbot.create_chat(new_chat_id, st.session_state.user_id)
            st.session_state.current_chat_id = new_chat_id
        chat = st.session_state.chatbot.get_chat(st.session_state.current_chat_id, st.session_state.user_id)
        chat.add_message("user", user_input)
        if not chat.name:
            chat.generate_name(user_input)
        response = chat.get_response()
        st.session_state.chatbot.save_chat(st.session_state.current_chat_id, st.session_state.user_id)
        st.experimental_rerun()
        
        
# Main app logic
def main():
    if st.session_state.user_id is None:
        login_page()
    elif st.session_state.page == 'chat_info':
        chat_info_page()
    elif st.session_state.page == 'graph':
        display_conversation_graph()
    elif st.session_state.page == 'new_chat':
        new_chat_page()
    else:
        homepage()

if __name__ == "__main__":
    main()