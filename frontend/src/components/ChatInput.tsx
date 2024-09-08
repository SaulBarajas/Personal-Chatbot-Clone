import React, { useState } from 'react';
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";

interface ChatInputProps {
  currentChatId: string | null;
  onSendMessage: (message: string) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ currentChatId, onSendMessage }) => {
  const [message, setMessage] = useState('');

  return (
    <div className="flex space-x-2">
      <Input
        value={message}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMessage(e.target.value)}
        placeholder="Type your message..."
        disabled={!currentChatId}
      />
      {/* ... rest of the component */}
    </div>
  );
};

export default ChatInput;