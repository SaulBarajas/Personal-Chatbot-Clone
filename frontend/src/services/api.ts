import axios from 'axios';

export const API_BASE_URL = 'http://localhost:8502';  // Adjust this to match your backend URL

export interface LoginResponse {
  user_id: string;
}

export interface ChatSummary {
  chat_id: string;
  name: string;
  last_message: string;
  message_count: number;
}

export interface CreateChatResponse {
  chat_id: string;
  response: string;
  user_message_id: string;
  assistant_message_id: string;
  user_message_current_edit_index: number;
  assistant_message_current_edit_index: number;
}

export interface SendMessageResponse {
  response: string;
  user_message_id: string;
  assistant_message_id: string;
  user_message_current_edit_index: number;
  assistant_message_current_edit_index: number;
}

export interface ChatHistoryResponse {
  messages: Array<{
    id: string;
    content: string;
    role: 'user' | 'assistant';
    edits?: Array<{ id: string; content: string }>;
    current_edit_index: number;  // Change this to be non-optional
    timestamp?: string;
    node_created?: string;
    last_accessed?: string;
  }>;
  running_summary: string;
  system_prompt: string;  // Add this line
}

export interface EditMessageResponse {
  edited_message: {
    id: string;
    content: string;
    edits: Array<{ id: string; content: string }>;
    edit_index: number;
  };
  new_response: {
    id: string;
    content: string;
  };
  updated_conversation: Array<{
    id: string;
    content: string;
    role: 'user' | 'assistant';
    edits?: Array<{ id: string; content: string }>;
    current_edit_index: number;  // Add this line
  }>;
}

export interface SwitchEditResponse {
  switched_node: {
    id: string;
    content: string;
    edit_index: number;
    edits: Array<{ id: string; content: string }>;
  };
  updated_conversation: Array<{
    id: string;
    content: string;
    role: 'user' | 'assistant';
    edits?: Array<{ id: string; content: string }>;
    current_edit_index: number;
  }>;
}

export const login = async (username: string): Promise<LoginResponse> => {
  const response = await axios.post<LoginResponse>(`${API_BASE_URL}/login?username=${encodeURIComponent(username)}`);
  return response.data;
};

export const getChatSummaries = async (userId: string): Promise<ChatSummary[]> => {
  const response = await axios.get<ChatSummary[]>(`${API_BASE_URL}/chat_summaries?user_id=${userId}`);
  return response.data;
};

export const createChat = async (message: string, userId: string): Promise<CreateChatResponse> => {
  const response = await axios.post<CreateChatResponse>(
    `${API_BASE_URL}/create_chat?user_id=${encodeURIComponent(userId)}`,
    {
      role: 'user',
      content: message
    }
  );
  return response.data;
};

export const sendMessage = async (chatId: string, message: string, userId: string): Promise<SendMessageResponse> => {
  const response = await axios.post<SendMessageResponse>(
    `${API_BASE_URL}/chat/${chatId}?user_id=${encodeURIComponent(userId)}`,
    {
      role: 'user',
      content: message
    }
  );
  return response.data;
};

export const setLLM = async (llm: string, userId: string): Promise<void> => {
  try {
    const response = await axios.put(`${API_BASE_URL}/set_llm`, { llm, user_id: userId });
    console.log('setLLM response:', response.data); // Add this log
    if (response.data.message !== "LLM changed successfully") {
      throw new Error('Failed to change LLM');
    }
  } catch (error) {
    console.error('Error in setLLM:', error);
    throw error;
  }
};

export const getChatHistory = async (chatId: string, userId: string): Promise<ChatHistoryResponse> => {
  const response = await axios.get<ChatHistoryResponse>(`${API_BASE_URL}/chat_history/${chatId}?user_id=${userId}`);
  return response.data;
};

export const editMessage = async (chatId: string, messageId: string, content: string, userId: string): Promise<EditMessageResponse> => {
  try {
    console.log('Sending edit request to API for message:', messageId, 'with content:', content);
    const response = await axios.put<EditMessageResponse>(`${API_BASE_URL}/edit_message/${chatId}/${messageId}?user_id=${userId}`, {
      role: 'user',  // Assuming the role is 'user' for edited messages
      content: content,
      id: messageId
    });
    console.log('Edit request successful');
    return response.data;
  } catch (error) {
    console.error('Error in editMessage API call:', error);
    throw error;
  }
};

export const updateLastAccessed = async (chatId: string, userId: string): Promise<void> => {
  await axios.put(`${API_BASE_URL}/update_last_accessed/${chatId}?user_id=${userId}`);
};

export const switchEdit = async (chatId: string, nodeId: string, editIndex: number, userId: string): Promise<SwitchEditResponse> => {
  const response = await axios.put<SwitchEditResponse>(
    `${API_BASE_URL}/switch_edit/${chatId}/${nodeId}/${editIndex}?user_id=${encodeURIComponent(userId)}`
  );
  return response.data;
};

// Add this new function to set the system prompt
export const updateSystemPrompt = async (chatId: string, prompt: string, userId: string): Promise<void> => {
  try {
    const response = await axios.put(
      `${API_BASE_URL}/set_system_prompt/${chatId}`,
      { prompt: prompt },
      { params: { user_id: userId } }
    );
    if (response.data.message !== "System prompt updated successfully") {
      throw new Error('Failed to update system prompt');
    }
  } catch (error) {
    console.error('Error in updateSystemPrompt:', error);
    throw error;
  }
};

export const updateChatName = async (chatId: string, name: string, userId: string): Promise<void> => {
  try {
    const response = await axios.put(
      `${API_BASE_URL}/set_chat_name/${chatId}`,
      { name },
      { params: { user_id: userId } }
    );
    if (response.data.message !== "Chat name updated successfully") {
      throw new Error('Failed to update chat name');
    }
  } catch (error) {
    console.error('Error in updateChatName:', error);
    throw error;
  }
};