export interface Conversation {
    id: string;
    title: string;
    lastAccessed: Date;  // Change this from string to Date
  }
  
  export interface ChatSummary {
    chat_id: string;
    name: string;
    last_message: string;
    message_count: number;
    last_accessed: string;
  }
  
  export interface Message {
    id: string;
    content: string;
    sender: 'user' | 'assistant';
    edits?: Array<{ id: string; content: string }>;
    current_edit_index: number;  // Make this non-optional
  }
  
  export interface ChatHistoryResponse {
    messages: Message[];
    running_summary: string;
    system_prompt: string;  // Add this line
  }