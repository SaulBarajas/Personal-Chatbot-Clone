from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
from chatbot_clone import Chatbot, llama8bLLM, Llama4bitLLM, llama70bLLM, llama405bLLM
import os
from dotenv import load_dotenv
import uuid
import graphviz

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Available LLMs
LLM_OPTIONS = {
    "llama8bLLM": llama8bLLM,
    "llama70bLLM": llama70bLLM,
    "llama405bLLM": llama405bLLM,
    "Llama4bitLLM": Llama4bitLLM
}

# Pydantic models for request/response bodies
class ChatMessage(BaseModel):
    role: str
    content: str
    id: Optional[str] = None
    edits: Optional[List[Dict[str, Any]]] = None
    current_edit_index: Optional[int] = None
    timestamp: Optional[datetime] = None  # Add timestamp
    node_created: Optional[datetime] = None  # Add node_created
    last_accessed: Optional[datetime] = None  # Add last_accessed

class ChatSummary(BaseModel):
    chat_id: str
    name: str
    last_message: str
    message_count: int
    last_accessed: datetime

class ChatResponse(BaseModel):
    chat_id: str
    response: str
    user_message_id: str
    assistant_message_id: str
    user_message_current_edit_index: int
    assistant_message_current_edit_index: int

class ChatHistory(BaseModel):
    messages: List[ChatMessage]
    running_summary: str
    system_prompt: str  # Add this line to include the system prompt

class LoginRequest(BaseModel):
    username: str

# In-memory store for chatbots (replace with database in production)
chatbots = {}

# Dependency to get the current chatbot
def get_chatbot(user_id: str):
    if user_id not in chatbots:
        chatbots[user_id] = Chatbot(llama8bLLM(), user_id)
    return chatbots[user_id]

@app.post("/login")
def login(username: str = Query(..., description="Username for login")):
    print(f"Received login request for username: {username}")  # Add this line for debugging
    if username not in chatbots:
        chatbots[username] = Chatbot(llama8bLLM(), username)
    return {"user_id": username}

@app.post("/chat/{chat_id}", response_model=ChatResponse)
def chat(chat_id: str, message: ChatMessage, user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    chat = chatbot.get_chat(chat_id, user_id)
    user_message = chat.add_message(message.role, message.content)
    chat.update_last_accessed()
    assistant_message = chat.get_response()
    chatbot.save_chat(chat_id, user_id)
    return ChatResponse(
        chat_id=chat_id,
        response=assistant_message.content,
        user_message_id=user_message.id,
        assistant_message_id=assistant_message.id,
        user_message_current_edit_index=0,
        assistant_message_current_edit_index=0
    )

@app.post("/create_chat", response_model=ChatResponse)
def create_chat(message: ChatMessage, user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    new_chat_id = str(uuid.uuid4())
    chat = chatbot.create_chat(new_chat_id, user_id)
    user_message = chat.add_message(message.role, message.content)
    chat.generate_name(message.content)
    chat.update_last_accessed()
    assistant_message = chat.get_response()
    chatbot.save_chat(new_chat_id, user_id)
    return ChatResponse(
        chat_id=new_chat_id,
        response=assistant_message.content,
        user_message_id=user_message.id,
        assistant_message_id=assistant_message.id,
        user_message_current_edit_index=0,
        assistant_message_current_edit_index=0
    )

@app.get("/chat_summaries", response_model=List[ChatSummary])
def get_chat_summaries(user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    summaries = chatbot.get_user_chat_summaries()
    sorted_summaries = sorted(summaries, key=lambda x: x['last_accessed'], reverse=True)
    for summary in sorted_summaries:
        print(f"Chat Name: {summary['name']}, Last Accessed: {summary['last_accessed']}")
    return [
        ChatSummary(
            chat_id=summary['chat_id'],
            name=summary['name'],
            last_message=summary['last_message'],
            message_count=summary['message_count'],
            last_accessed=summary['last_accessed']
        )
        for summary in sorted_summaries
    ]

@app.get("/chat_history/{chat_id}", response_model=ChatHistory)
def get_chat_history(chat_id: str, user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    chat = chatbot.get_chat(chat_id, user_id)
    history = chat.get_current_conversation()
    chatbot.save_chat(chat_id, user_id)
    return ChatHistory(  
        messages=[
            ChatMessage(
                role=msg["role"],
                content=msg["content"],
                id=msg["id"],
                edits=[{"id": edit["id"], "content": edit["content"]} for edit in msg.get("edits", [])],
                current_edit_index=msg["current_edit_index"],  # Ensure this is always present
                timestamp=msg.get("timestamp"),
                node_created=msg.get("node_created"),
                last_accessed=msg.get("last_accessed")
            ) for msg in history
        ],
        running_summary=chat.running_summary,
        system_prompt=chat.system_prompt  # Add this line to include the system prompt
    )

@app.put("/edit_message/{chat_id}/{node_id}")
def edit_message(chat_id: str, node_id: str, message: ChatMessage = Body(...), user_id: str = Query(...), chatbot: Chatbot = Depends(get_chatbot)):
    print(f"Received edit request for chat_id: {chat_id}, node_id: {node_id}, new content: {message.content}, user_id: {user_id}")
    edited_message, new_response = chatbot.edit_message(chat_id, user_id, node_id, message.content)
    chat = chatbot.get_chat(chat_id, user_id)
    edits = chat.get_node_edits(node_id)
    updated_conversation = chat.get_current_conversation()
    chatbot.save_chat(chat_id, user_id)
    return {
        "edited_message": {
            "id": edited_message.id,
            "content": edited_message.content,
            "edits": [{"id": edit.id, "content": edit.content} for edit in edits],
            "edit_index": edits.index(edited_message)
        },
        "new_response": {"id": new_response.id, "content": new_response.content},
        "updated_conversation": [
            {
                "id": msg["id"],
                "content": msg["content"],
                "role": msg["role"],
                "edits": msg["edits"],
                "current_edit_index": msg["current_edit_index"]
            } for msg in updated_conversation
        ]
    }

@app.put("/switch_edit/{chat_id}/{node_id}")
def switch_to_edit(chat_id: str, node_id: str, user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    chat = chatbot.get_chat(chat_id, user_id)
    chat.switch_to_edit(node_id)
    chatbot.save_chat(chat_id, user_id)
    return {"message": "Switched to edit successfully"}

@app.get("/node_edits/{chat_id}/{node_id}")
def get_node_edits(chat_id: str, node_id: str, user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    chat = chatbot.get_chat(chat_id, user_id)
    edits = chat.get_node_edits(node_id)
    return [{"id": edit.id, "content": edit.content} for edit in edits]

@app.put("/switch_edit/{chat_id}/{node_id}/{edit_index}")
def switch_edit(chat_id: str, node_id: str, edit_index: int, user_id: str = Query(...), chatbot: Chatbot = Depends(get_chatbot)):
    chat = chatbot.get_chat(chat_id, user_id)
    switched_node = chat.switch_to_edit(node_id, edit_index)
    chatbot.save_chat(chat_id, user_id)
    edits = chat.get_node_edits(node_id)
    updated_conversation = chat.get_current_conversation()
    return {
        "switched_node": {
            "id": switched_node.id,
            "content": switched_node.content,
            "edit_index": edit_index,
            "edits": [{"id": edit.id, "content": edit.content} for edit in edits]
        },
        "updated_conversation": [
            {
                "id": msg["id"],
                "content": msg["content"],
                "role": msg["role"],
                "edits": msg["edits"],
                "current_edit_index": msg["current_edit_index"]
            } for msg in updated_conversation
        ]
    }

@app.put("/set_llm")
def set_llm(llm: str = Body(...), user_id: str = Body(...)):
    if llm not in LLM_OPTIONS:
        raise HTTPException(status_code=400, detail="Invalid LLM name")
    try:
        new_llm = LLM_OPTIONS[llm]()
        chatbot = get_chatbot(user_id)
        chatbot.set_llm(new_llm)
        print(f"LLM changed to {llm} for user {user_id}")  # Add this log
        return {"message": "LLM changed successfully"}
    except Exception as e:
        print(f"Error changing LLM: {str(e)}")  # Add this log
        raise HTTPException(status_code=500, detail=f"Failed to change LLM: {str(e)}")

@app.put("/set_system_prompt/{chat_id}")
def set_system_prompt(
    chat_id: str,
    prompt: str = Body(..., embed=True),
    user_id: str = Query(...),
    chatbot: Chatbot = Depends(get_chatbot)
):
    chat = chatbot.get_chat(chat_id, user_id)
    chat.set_system_prompt(prompt)
    chatbot.save_chat(chat_id, user_id)
    return {"message": "System prompt updated successfully"}

@app.put("/set_chat_name/{chat_id}")
def set_chat_name(
    chat_id: str,
    name: str = Body(..., embed=True),
    user_id: str = Query(...),
    chatbot: Chatbot = Depends(get_chatbot)
):
    chat = chatbot.get_chat(chat_id, user_id)
    chat.set_name(name)
    chatbot.save_chat(chat_id, user_id)
    return {"message": "Chat name updated successfully"}

@app.get("/conversation_graph/{chat_id}")
def get_conversation_graph(chat_id: str, user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    chat = chatbot.get_chat(chat_id, user_id)
    graph = create_conversation_graph(chat)
    return {"graph": graph.source}

@app.get("/running_summary/{chat_id}")
def get_running_summary(chat_id: str, user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    chat = chatbot.get_chat(chat_id, user_id)
    return {"running_summary": chat.running_summary}

@app.put("/update_last_accessed/{chat_id}")
def update_last_accessed(chat_id: str, user_id: str, chatbot: Chatbot = Depends(get_chatbot)):
    chat = chatbot.get_chat(chat_id, user_id)
    chat.update_last_accessed()  # Update last accessed timestamp
    chatbot.save_chat(chat_id, user_id)
    return {"message": "Last accessed timestamp updated successfully"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8502)