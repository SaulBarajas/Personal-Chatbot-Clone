import React from 'react';
import { Button } from "../components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";

interface SidebarProps {
  user: string;
  onLogout: () => void;
  currentLLM: string;
  onLLMChange: (llm: string) => void;
  currentChatId: string | null;
  setCurrentChatId: (chatId: string | null) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  user,
  onLogout,
  currentLLM,
  onLLMChange,
  currentChatId,
  setCurrentChatId
}) => {
  // Mock chat list (replace with actual data fetching)
  const chatList = [
    { id: '1', name: 'Chat 1' },
    { id: '2', name: 'Chat 2' },
    { id: '3', name: 'Chat 3' },
  ];

  return (
    <div className="w-64 bg-gray-100 p-4 flex flex-col">
      <div className="mb-4">
        <h2 className="text-lg font-semibold">Welcome, {user}</h2>
        <Button onClick={onLogout} className="mt-2">Logout</Button>
      </div>
      <div className="mb-4">
        <SelectTrigger>
          <Select value={currentLLM} onValueChange={onLLMChange}>
            <SelectItem value="llama8bLLM">llama8bLLM</SelectItem>
            <SelectItem value="llama70bLLM">llama70bLLM</SelectItem>
            <SelectItem value="llama405bLLM">llama405bLLM</SelectItem>
            <SelectItem value="Llama4bitLLM">Llama4bitLLM</SelectItem>
          </Select>
          <SelectValue placeholder="Select LLM" />
        </SelectTrigger>
      </div>
      <div className="flex-1 overflow-y-auto">
        <h3 className="text-md font-semibold mb-2">Chats</h3>
        {chatList.map((chat) => (
          <Button
            key={chat.id}
            className={`w-full justify-start mb-1 ${currentChatId === chat.id ? "bg-blue-500 text-white" : ""}`}
            onClick={() => setCurrentChatId(chat.id)}
          >
            {chat.name}
          </Button>
        ))}
      </div>
      <Button onClick={() => setCurrentChatId(null)} className="mt-2">New Chat</Button>
    </div>
  );
};

export default Sidebar;