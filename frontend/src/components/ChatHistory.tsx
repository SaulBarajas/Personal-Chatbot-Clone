import React from 'react';

interface ChatHistoryProps {
  currentChatId: string | null;
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ currentChatId }) => {
  return (
    <div className="flex-1 overflow-y-auto p-4">
      {currentChatId ? (
        <p>Chat history for chat {currentChatId}</p>
      ) : (
        <p>Select a chat to view history</p>
      )}
    </div>
  );
};

export default ChatHistory;