'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { PlusIcon, SendIcon, UserIcon, BrainCircuitIcon, SettingsIcon, EditIcon, CopyIcon, ThumbsUpIcon, ThumbsDownIcon, CheckIcon, XIcon } from "lucide-react"

export function AppChatPage() {
  const [conversations, setConversations] = useState([
    { id: 1, title: "Getting started with AI" },
    { id: 2, title: "Explaining quantum computing" },
    { id: 3, title: "The future of renewable energy" },
  ])

  const [messages, setMessages] = useState([
    { id: 1, content: "Hello! How can I assist you today?", sender: "ai" },
    { id: 2, content: "Hi there! I'd like to learn about artificial intelligence.", sender: "user" },
    { id: 3, content: "Great! I'd be happy to help you learn about artificial intelligence. What specific aspects of AI would you like to know more about?", sender: "ai" },
  ])

  const [selectedLLM, setSelectedLLM] = useState("gpt-3.5-turbo")
  const [isDarkMode, setIsDarkMode] = useState(false)
  const [isNotificationsEnabled, setIsNotificationsEnabled] = useState(true)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [editingMessageId, setEditingMessageId] = useState(null)
  const [editedContent, setEditedContent] = useState("")

  const llmOptions = [
    { value: "gpt-3.5-turbo", label: "GPT-3.5 Turbo" },
    { value: "gpt-4", label: "GPT-4" },
    { value: "claude-v1", label: "Claude v1" },
    { value: "claude-instant-v1", label: "Claude Instant v1" },
  ]

  const handleEditMessage = (id, content) => {
    setEditingMessageId(id)
    setEditedContent(content)
  }

  const handleSaveEdit = () => {
    setMessages(messages.map(msg => msg.id === editingMessageId ? { ...msg, content: editedContent } : msg))
    setEditingMessageId(null)
    setEditedContent("")
  }

  const handleCancelEdit = () => {
    setEditingMessageId(null)
    setEditedContent("")
  }

  const handleCopyMessage = (content) => {
    navigator.clipboard.writeText(content)
  }

  return (
    <div className={`flex h-screen ${isDarkMode ? 'bg-slate-900 text-slate-100' : 'bg-green-50 text-slate-800'}`}>
      {/* Sidebar */}
      <div className={`w-64 ${isDarkMode ? 'bg-slate-800' : 'bg-green-100'} p-4 flex flex-col`}>
        <Button className={`mb-4 ${isDarkMode ? 'bg-slate-700 hover:bg-slate-600 text-slate-100' : 'bg-green-200 hover:bg-green-300 text-slate-800'}`} onClick={() => setConversations([...conversations, { id: conversations.length + 1, title: "New conversation" }])}>
          <PlusIcon className="mr-2 h-4 w-4" />
          New chat
        </Button>
        <div className="flex-1 overflow-auto">
          {conversations.map((conversation) => (
            <Button key={conversation.id} variant="ghost" className={`w-full justify-start mb-1 ${isDarkMode ? 'text-slate-100 hover:bg-slate-700' : 'text-slate-800 hover:bg-green-200'}`}>
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
          <DropdownMenuContent className={isDarkMode ? 'bg-slate-800 text-slate-100' : 'bg-green-100 text-slate-800'}>
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
              <DialogContent className={`sm:max-w-[425px] ${isDarkMode ? 'bg-slate-800 text-slate-100' : 'bg-green-100 text-slate-800'}`}>
                <DialogHeader>
                  <DialogTitle>Settings</DialogTitle>
                  <DialogDescription>Adjust your chat preferences here.</DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="dark-mode">Winter Mode</Label>
                    <Switch
                      id="dark-mode"
                      checked={isDarkMode}
                      onCheckedChange={setIsDarkMode}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="notifications">Notifications</Label>
                    <Switch
                      id="notifications"
                      checked={isNotificationsEnabled}
                      onCheckedChange={setIsNotificationsEnabled}
                    />
                  </div>
                </div>
              </DialogContent>
            </Dialog>
            <DropdownMenuItem className={isDarkMode ? 'focus:bg-slate-700' : 'focus:bg-green-200'}>Logout</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        <header className={`border-b p-4 flex justify-between items-center ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-green-100 border-green-200'}`}>
          <h1 className="text-xl font-bold">AI Assistant</h1>
          <div className="flex items-center space-x-2">
            <BrainCircuitIcon className={`h-5 w-5 ${isDarkMode ? 'text-slate-400' : 'text-green-600'}`} />
            <Select value={selectedLLM} onValueChange={setSelectedLLM}>
              <SelectTrigger className={`w-[180px] ${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-slate-800'}`}>
                <SelectValue placeholder="Select LLM" />
              </SelectTrigger>
              <SelectContent className={isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-green-200 text-slate-800'}>
                {llmOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </header>
        <main className={`flex-1 overflow-auto p-4 space-y-4 ${isDarkMode ? 'bg-slate-900' : 'bg-green-50'}`}>
          {messages.map((message, index) => (
            <div key={message.id} className={`flex flex-col ${message.sender === 'user' ? 'items-end' : 'items-start'}`}>
              <div className={`flex items-start space-x-2 ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}>
                {message.sender === 'ai' && (
                  <Avatar>
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                )}
                <div className={`rounded-lg p-2 max-w-[70%] ${
                  message.sender === 'user'
                    ? isDarkMode ? 'bg-blue-600 text-slate-100' : 'bg-green-500 text-white'
                    : isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-white text-slate-800'
                }`}>
                  {editingMessageId === message.id ? (
                    <div className="flex flex-col space-y-2">
                      <Input
                        value={editedContent}
                        onChange={(e) => setEditedContent(e.target.value)}
                        className={isDarkMode ? 'bg-slate-600 text-slate-100' : 'bg-white text-slate-800'}
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
                    <p>{message.content}</p>
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
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="sm" onClick={() => handleEditMessage(message.id, message.content)}>
                          <EditIcon className="h-4 w-4 mr-1" /> Edit
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Edit message</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
                {message.sender === 'ai' && (
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
          ))}
        </main>
        <footer className={`p-4 border-t ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-green-100 border-green-200'}`}>
          <form className="flex space-x-2" onSubmit={(e) => {
            e.preventDefault()
            const input = e.currentTarget.elements.namedItem('message') as HTMLInputElement
            if (input.value.trim()) {
              setMessages([...messages, { id: messages.length + 1, content: input.value, sender: 'user' }])
              input.value = ''
            }
          }}>
            <Input className={`flex-1 ${isDarkMode ? 'bg-slate-700 text-slate-100' : 'bg-white text-slate-800'}`} placeholder="Type your message..." name="message" />
            <Button type="submit" size="icon" className={isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-green-500 hover:bg-green-600'}>
              <SendIcon className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </Button>
          </form>
        </footer>
      </div>
    </div>
  )
}