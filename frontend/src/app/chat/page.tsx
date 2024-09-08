import { useState, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { PlusIcon, SendIcon, UserIcon, BrainCircuitIcon, SettingsIcon, EditIcon, CopyIcon, ThumbsUpIcon, ThumbsDownIcon, CheckIcon, XIcon, ArrowLeftIcon, ArrowRightIcon, LogOutIcon, PencilIcon } from "lucide-react"
import { User } from '@/types/user'
import { Conversation, Message } from '@/types/chat'
import { getChatSummaries, editMessage, createChat, sendMessage, setLLM, getChatHistory, CreateChatResponse, SendMessageResponse, updateLastAccessed, API_BASE_URL, switchEdit as apiSwitchEdit, SwitchEditResponse, EditMessageResponse, updateSystemPrompt, updateChatName } from '@/services/api';
import { Spinner } from "@/components/ui/spinner";
import { CodeBlock } from "@/components/CodeBlock";
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { Textarea } from "@/components/ui/textarea"

interface CodeProps extends React.HTMLAttributes<HTMLElement> {
    node?: any;
    inline?: boolean;
    className?: string;
    children?: React.ReactNode;
}

interface ChatPageProps {
  userId: string;
}

const EditControls = ({ message, onEdit, editIndices, onSwitchEdit }: { 
  message: Message; 
  onEdit: (id: string, content: string) => void; 
  editIndices: { [messageId: string]: number };
  onSwitchEdit: (messageId: string, direction: 'prev' | 'next') => void;
}) => {
  const currentEditIndex = editIndices[message.id] || 0;
  const totalEdits = message.edits?.length || 1;

  return (
    <div className="flex items-center space-x-1">
      <Button
        size="sm"
        variant="ghost"
        onClick={() => onSwitchEdit(message.id, 'prev')}
        disabled={currentEditIndex === 0}
      >
        <ArrowLeftIcon className="h-3 w-3" />
      </Button>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={() => onEdit(message.id, message.content)}>
              <EditIcon className="h-4 w-4 mr-1" /> Edit {`(${currentEditIndex + 1}/${totalEdits})`}
            </Button>
          </TooltipTrigger>
          <TooltipContent>Edit message</TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <Button
        size="sm"
        variant="ghost"
        onClick={() => onSwitchEdit(message.id, 'next')}
        disabled={currentEditIndex === totalEdits - 1}
      >
        <ArrowRightIcon className="h-3 w-3" />
      </Button>
    </div>
  );
};

export default function ChatPage({ userId }: ChatPageProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [selectedLLM, setSelectedLLM] = useState<string>("llama8bLLM")
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false)
  const [isNotificationsEnabled, setIsNotificationsEnabled] = useState<boolean>(true)
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false)
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [editedContent, setEditedContent] = useState<string>("")
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [inputMessage, setInputMessage] = useState<string>("")
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [streamResponse, setStreamResponse] = useState(true)
  const [autoSendTyping, setAutoSendTyping] = useState(false)
  const [editIndices, setEditIndices] = useState<{ [messageId: string]: number }>({});
  const [systemPrompt, setSystemPrompt] = useState<string>("You are a helpful assistant.");
  const [isChatNameDialogOpen, setIsChatNameDialogOpen] = useState(false);
  const [newChatName, setNewChatName] = useState("");
  const navigate = useNavigate();

  const llmOptions = [
    { value: "llama8bLLM", label: "Llama 8B" },
    { value: "llama70bLLM", label: "Llama 70B" },
    { value: "llama405bLLM", label: "Llama 405B" },
    { value: "Llama4bitLLM", label: "Llama 4-bit" },
  ]

  useEffect(() => {
    const initializeChats = async () => {
      const sortedConversations = await fetchChatSummaries();
      setConversations(sortedConversations);
    };
    initializeChats();
  }, [userId]);

  useEffect(() => {
    if (currentChatId) {
      loadChatHistory(currentChatId);
    }
  }, [currentChatId]);

  const fetchChatSummaries = async () => {
    try {
      // Add a small delay before fetching summaries
      await new Promise(resolve => setTimeout(resolve, 100));
      const summaries = await getChatSummaries(userId);
      // Ensure summaries are sorted by last_accessed
      const sortedConversations: Conversation[] = summaries
        .map((summary: any) => ({
          id: summary.chat_id,
          title: summary.name,
          lastAccessed: new Date(summary.last_accessed)
        }))
        .sort((a, b) => b.lastAccessed.getTime() - a.lastAccessed.getTime());
      return sortedConversations;
    } catch (error) {
      console.error('Error fetching chat summaries:', error);
      return [];
    }
  };

  const handleEditMessage = async (id: string, content: string) => {
    const message = messages.find(msg => msg.id === id);
    const currentEditIndex = editIndices[id] || 0;
    setEditingMessageId(id);
    setEditedContent(message?.edits?.[currentEditIndex]?.content || content);
  }

  const handleSaveEdit = async () => {
    if (!currentChatId || !editingMessageId) return;
    console.log('Saving edit for message:', editingMessageId, 'with content:', editedContent);
    try {
      const response = await editMessage(currentChatId, editingMessageId, editedContent, userId);
      console.log('Edit saved successfully, response:', response);
      
      const updatedMessages = response.updated_conversation.map(msg => ({
        id: msg.id,
        content: msg.content,
        sender: msg.role as 'user' | 'assistant',
        edits: msg.edits,
        current_edit_index: msg.current_edit_index
      }));
      setMessages(updatedMessages);
      updateEditIndices(updatedMessages);
      
      setEditingMessageId(null);
      setEditedContent("");
      
      // Fetch updated chat summaries after editing a message
      const updatedConversations = await fetchChatSummaries();
      setConversations(updatedConversations);
    } catch (error) {
      console.error('Error editing message:', error);
      setError('Failed to edit message. Please try again.');
    }
  }

  const handleCancelEdit = () => {
    setEditingMessageId(null)
    setEditedContent("")
  }

  const handleCopyMessage = (content: string) => {
    navigator.clipboard.writeText(content)
  }

  const handleSendMessage = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (inputMessage.trim()) {
      setIsLoading(true);
      setError(null);
      try {
        let response: CreateChatResponse | SendMessageResponse;
        if (!currentChatId) {
          response = await createChat(inputMessage, userId);
          if ('chat_id' in response) {
            setCurrentChatId(response.chat_id);
            setMessages([
              { id: response.user_message_id, content: inputMessage, sender: 'user', current_edit_index: 0 },
              { id: response.assistant_message_id, content: response.response, sender: 'assistant', current_edit_index: 0 }
            ]);
          }
        } else {
          response = await sendMessage(currentChatId, inputMessage, userId);
          setMessages(prevMessages => [
            ...prevMessages,
            { id: response.user_message_id, content: inputMessage, sender: 'user', current_edit_index: 0 },
            { id: response.assistant_message_id, content: response.response, sender: 'assistant', current_edit_index: 0 }
          ]);
        }
        setInputMessage('');
        // Fetch updated chat summaries after sending a message
        const updatedConversations = await fetchChatSummaries();
        setConversations(updatedConversations);
      } catch (error) {
        console.error('Error sending message:', error);
        if (axios.isAxiosError(error) && error.response) {
          setError(`Failed to send message: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
        } else if (error instanceof Error) {
          setError(`Failed to send message: ${error.message}`);
        } else {
          setError('Failed to send message. Please try again.');
        }
      } finally {
        setIsLoading(false);
      }
    }
  }

  const handleLLMChange = async (llm: string) => {
    try {
      setIsLoading(true);
      await setLLM(llm, userId);
      setSelectedLLM(llm);
      console.log(`LLM changed to: ${llm}`);
      if (currentChatId) {
        await loadChatHistory(currentChatId);
      }
    } catch (error) {
      console.error('Error changing LLM:', error);
      setError('Failed to change LLM. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }

  const updateEditIndices = (messages: Message[]) => {
    const newEditIndices: { [messageId: string]: number } = {};
    messages.forEach(msg => {
      if (msg.edits && msg.edits.length > 0) {
        newEditIndices[msg.id] = msg.current_edit_index ?? 0;
      }
    });
    setEditIndices(newEditIndices);
  };

  const loadChatHistory = async (chatId: string) => {
    try {
      setMessages([]);  // Clear messages state before loading new chat history
      const history = await getChatHistory(chatId, userId);
      const newMessages = history.messages.map(msg => ({
        id: msg.id,
        content: msg.content,
        sender: msg.role as 'user' | 'assistant',
        edits: msg.edits || [],
        current_edit_index: msg.current_edit_index
      }));
      setMessages(newMessages);
      updateEditIndices(newMessages);
      setCurrentChatId(chatId);
      setSystemPrompt(history.system_prompt); // Add this line to update the system prompt
      
      // Fetch and update all conversations
      const updatedConversations = await fetchChatSummaries();
      setConversations(updatedConversations);
      
      // Update last accessed timestamp on the backend
      await updateLastAccessed(chatId, userId);
    } catch (error) {
      console.error('Error loading chat history:', error);
      setError('Failed to load chat history. Please try again.');
    }
  };

  const handleNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
    setInputMessage('');
    setError(null);
  };

  const handleSwitchEdit = async (messageId: string, direction: 'prev' | 'next') => {
    const message = messages.find(msg => msg.id === messageId);
    if (!message || !message.edits) return;

    let newIndex = editIndices[messageId] || 0;
    if (direction === 'prev') {
      newIndex = (newIndex - 1 + message.edits.length) % message.edits.length;
    } else {
      newIndex = (newIndex + 1) % message.edits.length;
    }

    try {
      if (!currentChatId) throw new Error("No active chat");
      const response = await apiSwitchEdit(currentChatId, messageId, newIndex, userId);
      
      // Update the entire conversation
      const updatedMessages = response.updated_conversation.map(msg => ({
        id: msg.id,
        content: msg.content,
        sender: msg.role as 'user' | 'assistant',
        edits: msg.edits,
        current_edit_index: msg.current_edit_index
      }));
      setMessages(updatedMessages);
      
      // Update editIndices
      const newEditIndices = { ...editIndices };
      updatedMessages.forEach(msg => {
        if (msg.edits && msg.edits.length > 0) {
          newEditIndices[msg.id] = msg.current_edit_index;
        }
      });
      setEditIndices(newEditIndices);

      // If we're currently editing this message, update the editedContent
      if (editingMessageId === messageId) {
        setEditedContent(response.switched_node.content);
      }
    } catch (error) {
      console.error('Error switching edit:', error);
      setError('Failed to switch edit. Please try again.');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>, messageId: string) => {
    if (e.key === 'ArrowLeft') {
      handleSwitchEdit(messageId, 'prev');
    } else if (e.key === 'ArrowRight') {
      handleSwitchEdit(messageId, 'next');
    }
  }

  const handleLogout = () => {
    // Clear user session (you might want to call an API to invalidate the session on the server)
    // For now, we'll just clear the user ID from local storage
    localStorage.removeItem('userId');
    
    // Redirect to login page
    navigate('/login');
  };

  const handleSystemPromptChange = async (newPrompt: string) => {
    if (!currentChatId) {
      setError("No active chat to update system prompt");
      return;
    }
    try {
      setIsLoading(true);
      await updateSystemPrompt(currentChatId, newPrompt, userId);
      setSystemPrompt(newPrompt);
      console.log(`System prompt updated for chat ${currentChatId}`);
    } catch (error) {
      console.error('Error updating system prompt:', error);
      setError('Failed to update system prompt. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleChatNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewChatName(e.target.value);
  };

  const handleChatNameSave = async () => {
    if (!currentChatId) return;
    try {
      await updateChatName(currentChatId, newChatName, userId);
      const updatedConversations = await fetchChatSummaries();
      setConversations(updatedConversations);
      setIsChatNameDialogOpen(false);
    } catch (error) {
      console.error('Error updating chat name:', error);
      setError('Failed to update chat name. Please try again.');
    }
  };

  return (
    <div className={`flex h-screen ${isDarkMode ? 'bg-slate-900 text-slate-100' : 'bg-green-50 text-black'}`}>
      {/* Sidebar */}
      <div className={`w-64 ${isDarkMode ? 'bg-slate-800' : 'bg-green-100'} p-4 flex flex-col`}>
        <Button 
          className={`mb-4 ${isDarkMode ? 'bg-slate-700 hover:bg-slate-600 text-slate-100' : 'bg-green-200 hover:bg-green-300 text-black'}`} 
          onClick={handleNewChat}
        >
          <PlusIcon className="mr-2 h-4 w-4" />
          New chat
        </Button>
        <div className="flex-1 overflow-auto">
          {conversations.map((conversation) => (
            <Button key={conversation.id} variant="ghost" className={`w-full justify-start mb-1 ${isDarkMode ? 'text-slate-100 hover:bg-slate-700' : 'text-black hover:bg-green-200'}`} onClick={() => loadChatHistory(conversation.id)}>
              {conversation.title}
            </Button>
          ))}
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className={`w-full justify-start ${isDarkMode ? 'text-slate-100 hover:bg-slate-700' : 'text-slate-800 hover:bg-green-200'}`}>
              <UserIcon className="mr-2 h-4 w-4" />
              Profile
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className={`w-56 ${isDarkMode ? 'bg-slate-800 text-slate-100' : 'bg-green-100 text-slate-800'}`}>
            <div className="flex items-center justify-start p-2">
              <Avatar className="h-10 w-10 mr-2">
                <AvatarFallback>{userId.slice(0, 2).toUpperCase()}</AvatarFallback>
              </Avatar>
              <div className="flex flex-col">
                <p className="text-sm font-medium leading-none">{userId}</p>
              </div>
            </div>
            <DropdownMenuSeparator />
            <DropdownMenuLabel>My Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
              <DialogTrigger asChild>
                <DropdownMenuItem 
                  className={isDarkMode ? 'focus:bg-slate-700' : 'focus:bg-green-200'}
                  onSelect={(event) => {
                    event.preventDefault()
                    setIsSettingsOpen(true)
                  }}
                >
                  <SettingsIcon className="mr-2 h-4 w-4" />
                  Settings
                </DropdownMenuItem>
              </DialogTrigger>
                <DialogContent className={`sm:max-w-[625px] ${isDarkMode ? 'bg-slate-800 text-slate-100' : 'bg-green-100 text-slate-800'}`}>
                <DialogHeader>
                  <DialogTitle>Settings</DialogTitle>
                  <DialogDescription>Adjust your chat and LLM preferences here.</DialogDescription>
                </DialogHeader>
                <Tabs defaultValue="llm" className="w-full">
                  <TabsList className={`grid w-full grid-cols-3 ${isDarkMode ? 'bg-slate-700' : 'bg-green-200'}`}>
                    <TabsTrigger value="llm" className={`${isDarkMode ? 'data-[state=active]:bg-slate-600' : 'data-[state=active]:bg-green-300'}`}>LLM Settings</TabsTrigger>
                    <TabsTrigger value="chat" className={`${isDarkMode ? 'data-[state=active]:bg-slate-600' : 'data-[state=active]:bg-green-300'}`}>Chat Settings</TabsTrigger>
                    <TabsTrigger value="system" className={`${isDarkMode ? 'data-[state=active]:bg-slate-600' : 'data-[state=active]:bg-green-300'}`}>System Prompt</TabsTrigger>
                  </TabsList>
                  <TabsContent value="llm">
                    <div className="grid gap-4 py-4">
                      <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="llm-model" className="text-right">
                          Model
                        </Label>
                        <Select value={selectedLLM} onValueChange={handleLLMChange}>
                          <SelectTrigger id="llm-model" className={`${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-black'}`}>
                            <SelectValue placeholder="Select LLM" />
                          </SelectTrigger>
                          <SelectContent className={isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-black'}>
                            {llmOptions.map((option) => (
                              <SelectItem key={option.value} value={option.value}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="temperature" className="text-right">
                          Temperature
                        </Label>
                        <div className="col-span-3 flex items-center gap-4">
                          <Slider
                            id="temperature"
                            min={0}
                            max={1}
                            step={0.1}
                            value={[temperature]}
                            onValueChange={(value: number[]) => setTemperature(value[0])}
                            className={`flex-grow ${isDarkMode ? '[&_[role=slider]]:bg-slate-100' : '[&_[role=slider]]:bg-green-800 border border-gray-400'}`}
                          />
                          <span className="w-12 text-center">{temperature.toFixed(1)}</span>
                        </div>
                      </div>
                      <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="max-tokens" className="text-right">
                          Max Tokens
                        </Label>
                        <div className="col-span-3 flex items-center gap-4">
                          <Slider
                            id="max-tokens"
                            min={1}
                            max={4096}
                            step={1}
                            value={[maxTokens]}
                            onValueChange={(value: number[]) => setMaxTokens(value[0])}
                            className={`flex-grow ${isDarkMode ? '[&_[role=slider]]:bg-slate-100' : '[&_[role=slider]]:bg-green-800 border border-gray-400'}`}
                            style={{
                              '--slider-track-color': isDarkMode ? '#4B5563' : '#D1D5DB',
                              '--slider-range-color': isDarkMode ? '#60A5FA' : '#3B82F6',
                              '--slider-thumb-color': isDarkMode ? '#F9FAFB' : '#1F2937',
                            } as React.CSSProperties}
                          />
                          <span className="w-12 text-center">{maxTokens}</span>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                  <TabsContent value="chat">
                    <div className="grid gap-4 py-4">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="dark-mode">Winter Mode</Label>
                        <Switch
                          id="dark-mode"
                          checked={isDarkMode}
                          onCheckedChange={setIsDarkMode}
                          className={`${isDarkMode ? 'bg-slate-600' : 'bg-green-300'} data-[state=checked]:bg-green-600`}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="notifications">Notifications</Label>
                        <Switch
                          id="notifications"
                          checked={isNotificationsEnabled}
                          onCheckedChange={setIsNotificationsEnabled}
                          className={`${isDarkMode ? 'bg-slate-600' : 'bg-green-300'} data-[state=checked]:bg-green-600`}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="stream-response">Stream Response</Label>
                        <Switch
                          id="stream-response"
                          checked={streamResponse}
                          onCheckedChange={setStreamResponse}
                          className={`${isDarkMode ? 'bg-slate-600' : 'bg-green-300'} data-[state=checked]:bg-green-600`}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="auto-send-typing">Auto-send Typing</Label>
                        <Switch
                          id="auto-send-typing"
                          checked={autoSendTyping}
                          onCheckedChange={setAutoSendTyping}
                          className={`${isDarkMode ? 'bg-slate-600' : 'bg-green-300'} data-[state=checked]:bg-green-600`}
                        />
                      </div>
                    </div>
                  </TabsContent>
                  <TabsContent value="system">
                    <div className="grid gap-4 py-4">
                      <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="system-prompt" className="text-right">
                          System Prompt
                        </Label>
                        <Textarea
                          id="system-prompt"
                          value={systemPrompt}
                          onChange={(e) => setSystemPrompt(e.target.value)}
                          className={`col-span-3 ${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-black'}`}
                          rows={5}
                        />
                      </div>
                      <Button onClick={() => handleSystemPromptChange(systemPrompt)} className="ml-auto">
                        Update System Prompt
                      </Button>
                    </div>
                  </TabsContent>
                </Tabs>
              </DialogContent>
            </Dialog>
            <DropdownMenuItem 
              className={isDarkMode ? 'focus:bg-slate-700' : 'focus:bg-green-200'}
              onSelect={handleLogout}
            >
              <LogOutIcon className="mr-2 h-4 w-4" />
              <span>Log out</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        <header className={`border-b p-4 flex justify-between items-center ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-green-100 border-green-200'}`}>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold">
              {conversations.find(conv => conv.id === currentChatId)?.title || "AI Assistant"}
            </h1>
            <Dialog open={isChatNameDialogOpen} onOpenChange={setIsChatNameDialogOpen}>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm">
                  <PencilIcon className="h-4 w-4" />
                  <span className="sr-only">Edit chat name</span>
                </Button>
              </DialogTrigger>
              <DialogContent className={`sm:max-w-[425px] ${isDarkMode ? 'bg-slate-800 text-slate-100' : 'bg-green-100 text-slate-800'}`}>
                <DialogHeader>
                  <DialogTitle>Edit Chat Name</DialogTitle>
                  <DialogDescription>
                    Change the name of the current chat.
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <div className="grid grid-cols-4 items-center gap-4">
                    <Label htmlFor="chat-name" className="text-right">
                      Name
                    </Label>
                    <Input
                      id="chat-name"
                      value={newChatName}
                      onChange={handleChatNameChange}
                      className={`col-span-3 ${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-white text-black'}`}
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button type="submit" onClick={handleChatNameSave}>Save changes</Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
          <div className="flex items-center space-x-2">
            <BrainCircuitIcon className={`h-5 w-5 ${isDarkMode ? 'text-slate-400' : 'text-green-600'}`} />
            <Select value={selectedLLM} onValueChange={handleLLMChange}>
              <SelectTrigger className={`w-[180px] ${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-black'}`}>
                <SelectValue placeholder="Select LLM" />
              </SelectTrigger>
              <SelectContent className={isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-black'}>
                {llmOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm" className={isDarkMode ? 'bg-slate-700 hover:bg-slate-600' : 'bg-green-200 hover:bg-green-300'}>
                  <SettingsIcon className="h-4 w-4" />
                  <span className="sr-only">Open settings</span>
                </Button>
              </DialogTrigger>
              <DialogContent className={`sm:max-w-[625px] ${isDarkMode ? 'bg-slate-800 text-slate-100' : 'bg-green-100 text-slate-800'}`}>
                <DialogHeader>
                  <DialogTitle>Settings</DialogTitle>
                  <DialogDescription>Adjust your chat and LLM preferences here.</DialogDescription>
                </DialogHeader>
                <Tabs defaultValue="llm" className="w-full">
                  <TabsList className={`grid w-full grid-cols-3 ${isDarkMode ? 'bg-slate-700' : 'bg-green-200'}`}>
                    <TabsTrigger value="llm" className={`${isDarkMode ? 'data-[state=active]:bg-slate-600' : 'data-[state=active]:bg-green-300'}`}>LLM Settings</TabsTrigger>
                    <TabsTrigger value="chat" className={`${isDarkMode ? 'data-[state=active]:bg-slate-600' : 'data-[state=active]:bg-green-300'}`}>Chat Settings</TabsTrigger>
                    <TabsTrigger value="system" className={`${isDarkMode ? 'data-[state=active]:bg-slate-600' : 'data-[state=active]:bg-green-300'}`}>System Prompt</TabsTrigger>
                  </TabsList>
                  <TabsContent value="llm">
                    <div className="grid gap-4 py-4">
                      <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="llm-model" className="text-right">
                          Model
                        </Label>
                        <Select value={selectedLLM} onValueChange={handleLLMChange}>
                          <SelectTrigger id="llm-model" className={`${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-black'}`}>
                            <SelectValue placeholder="Select LLM" />
                          </SelectTrigger>
                          <SelectContent className={isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-black'}>
                            {llmOptions.map((option) => (
                              <SelectItem key={option.value} value={option.value}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="temperature" className="text-right">
                          Temperature
                        </Label>
                        <div className="col-span-3 flex items-center gap-4">
                          <Slider
                            id="temperature"
                            min={0}
                            max={1}
                            step={0.1}
                            value={[temperature]}
                            onValueChange={(value: number[]) => setTemperature(value[0])}
                            className={`flex-grow ${isDarkMode ? '[&_[role=slider]]:bg-slate-100' : '[&_[role=slider]]:bg-green-800 border border-gray-400'}`}
                          />
                          <span className="w-12 text-center">{temperature.toFixed(1)}</span>
                        </div>
                      </div>
                      <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="max-tokens" className="text-right">
                          Max Tokens
                        </Label>
                        <div className="col-span-3 flex items-center gap-4">
                          <Slider
                            id="max-tokens"
                            min={1}
                            max={4096}
                            step={1}
                            value={[maxTokens]}
                            onValueChange={(value: number[]) => setMaxTokens(value[0])}
                            className={`flex-grow ${isDarkMode ? '[&_[role=slider]]:bg-slate-100' : '[&_[role=slider]]:bg-green-800 border border-gray-400'}`}
                            style={{
                              '--slider-track-color': isDarkMode ? '#4B5563' : '#D1D5DB',
                              '--slider-range-color': isDarkMode ? '#60A5FA' : '#3B82F6',
                              '--slider-thumb-color': isDarkMode ? '#F9FAFB' : '#1F2937',
                            } as React.CSSProperties}
                          />
                          <span className="w-12 text-center">{maxTokens}</span>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                  <TabsContent value="chat">
                    <div className="grid gap-4 py-4">
                      <div className="flex items-center justify-between">
                        <Label htmlFor="dark-mode">Winter Mode</Label>
                        <Switch
                          id="dark-mode"
                          checked={isDarkMode}
                          onCheckedChange={setIsDarkMode}
                          className={`${isDarkMode ? 'bg-slate-600' : 'bg-green-300'} data-[state=checked]:bg-green-600`}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="notifications">Notifications</Label>
                        <Switch
                          id="notifications"
                          checked={isNotificationsEnabled}
                          onCheckedChange={setIsNotificationsEnabled}
                          className={`${isDarkMode ? 'bg-slate-600' : 'bg-green-300'} data-[state=checked]:bg-green-600`}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="stream-response">Stream Response</Label>
                        <Switch
                          id="stream-response"
                          checked={streamResponse}
                          onCheckedChange={setStreamResponse}
                          className={`${isDarkMode ? 'bg-slate-600' : 'bg-green-300'} data-[state=checked]:bg-green-600`}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label htmlFor="auto-send-typing">Auto-send Typing</Label>
                        <Switch
                          id="auto-send-typing"
                          checked={autoSendTyping}
                          onCheckedChange={setAutoSendTyping}
                          className={`${isDarkMode ? 'bg-slate-600' : 'bg-green-300'} data-[state=checked]:bg-green-600`}
                        />
                      </div>
                    </div>
                  </TabsContent>
                  <TabsContent value="system">
                    <div className="grid gap-4 py-4">
                      <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="system-prompt" className="text-right">
                          System Prompt
                        </Label>
                        <Textarea
                          id="system-prompt"
                          value={systemPrompt}
                          onChange={(e) => setSystemPrompt(e.target.value)}
                          className={`col-span-3 ${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-black'}`}
                          rows={5}
                        />
                      </div>
                      <Button onClick={() => handleSystemPromptChange(systemPrompt)} className="ml-auto">
                        Update System Prompt
                      </Button>
                    </div>
                  </TabsContent>
                </Tabs>
              </DialogContent>
            </Dialog>
          </div>
        </header>
        <main className={`flex-1 overflow-auto p-4 space-y-4 ${isDarkMode ? 'bg-slate-900' : 'bg-green-50'}`}>
          {messages.length > 0 ? (
            messages.map((message, index) => (
              <div key={message.id} className={`flex flex-col ${message.sender === 'user' ? 'items-end' : 'items-start'} w-full`}>
                <div className={`flex items-start space-x-2 ${message.sender === 'user' ? 'flex-row-reverse' : ''} ${editingMessageId === message.id ? 'w-full' : 'max-w-[75%]'}`}>
                  {message.sender === 'assistant' && (
                    <Avatar>
                      <AvatarFallback>AI</AvatarFallback>
                    </Avatar>
                  )}
                  <div className={`rounded-lg p-4 ${
                    message.sender === 'user'
                      ? isDarkMode ? 'bg-blue-600 text-slate-100' : 'bg-green-200 text-slate-800'
                      : isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-100 text-slate-800'
                  } ${editingMessageId === message.id ? 'w-full' : 'inline-block'}`}>
                    {editingMessageId === message.id ? (
                      <div className="flex flex-col space-y-2 w-full">
                        <div className="flex items-center space-x-2 w-full">
                          {message.edits && message.edits.length > 1 && (
                            <>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => handleSwitchEdit(message.id, 'prev')}
                                disabled={editIndices[message.id] === 0}
                              >
                                <ArrowLeftIcon className="h-4 w-4" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => handleSwitchEdit(message.id, 'next')}
                                disabled={editIndices[message.id] === message.edits.length - 1}
                              >
                                <ArrowRightIcon className="h-4 w-4" />
                              </Button>
                            </>
                          )}
                        </div>
                        <Textarea
                          value={editedContent}
                          onChange={(e) => setEditedContent(e.target.value)}
                          onKeyDown={(e) => handleKeyDown(e, message.id)}
                          className={`w-full min-h-[100px] resize-y ${
                            isDarkMode ? 'bg-slate-600 text-slate-100' : 'bg-white text-black'
                          }`}
                        />
                        <div className="flex justify-end space-x-2">
                          <Button size="sm" onClick={handleSaveEdit}>
                            <CheckIcon className="h-4 w-4 mr-1" /> Save
                          </Button>
                          <Button size="sm" variant="outline" onClick={handleCancelEdit}>
                            <XIcon className="h-4 w-4 mr-1" /> Cancel
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      components={{
                          code: ({ node, inline, className, children, ...props }: CodeProps) => {
                          const match = /language-(\w+)/.exec(className || '')
                          return !inline && match ? (
                              <CodeBlock
                              className={className}
                              {...props}
                              >
                              {String(children).replace(/\n$/, '')}
                              </CodeBlock>
                          ) : (
                              <code className={className} {...props}>
                              {children}
                              </code>
                          )
                          },
                      }}
                      className={`prose ${isDarkMode ? 'dark:prose-invert' : 'prose-black'} max-w-none break-words`}
                      >
                      {message.content}
                      </ReactMarkdown>
                    )}
                  </div>
                  {message.sender === 'user' && (
                    <Avatar>
                      <AvatarImage src="/placeholder-user.jpg" alt="User" />
                      <AvatarFallback>U</AvatarFallback>
                    </Avatar>
                  )}
                </div>
                <div className={`flex mt-1 space-x-2 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {message.sender === 'user' && editingMessageId !== message.id && (
                    <EditControls message={message} onEdit={handleEditMessage} editIndices={editIndices} onSwitchEdit={handleSwitchEdit} />
                  )}
                  {message.sender === 'assistant' && (
                    <>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button variant="ghost" size="sm" onClick={() => handleCopyMessage(message.content)}>
                              <CopyIcon className="h-4 w-4 mr-1" /> Copy
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Copy message</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button variant="ghost" size="sm">
                              <ThumbsUpIcon className="h-4 w-4 mr-1" /> Like
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Like response</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button variant="ghost" size="sm">
                              <ThumbsDownIcon className="h-4 w-4 mr-1" /> Dislike
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Dislike response</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500">Start a new conversation</p>
            </div>
          )}
          {isLoading && (
            <div className="flex justify-center">
              <Spinner />
            </div>
          )}
          {error && (
            <div className="text-red-500 text-center">{error}</div>
          )}
        </main>
        <footer className={`p-4 border-t ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-green-100 border-green-200'}`}>
          <form className="flex space-x-2" onSubmit={handleSendMessage}>
            <Input 
              className={`flex-1 ${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-white text-black'}`} 
              placeholder="Type your message..." 
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              disabled={isLoading}
            />
            <Button type="submit" size="sm" className={isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-green-500 hover:bg-green-600'} disabled={isLoading}>
              {isLoading ? <Spinner size="sm" /> : <SendIcon className="h-4 w-4" />}
              <span className="sr-only">Send</span>
            </Button>
          </form>
        </footer>
      </div>
    </div>
  )
}