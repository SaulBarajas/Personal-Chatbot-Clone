import React, { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './app/login/page';
import ChatPage from './app/chat/page';
import { User } from './types/user';
import './styles/global.css';

function App() {
  const [user, setUser] = useState<User | null>(null);

  const handleSetUser = (userId: string) => {
    setUser({ id: userId, name: userId }); // Using userId as name for simplicity
  };

  return (
    <Routes>
      <Route path="/login" element={<LoginPage setUser={handleSetUser} />} />
      <Route 
        path="/chat" 
        element={user ? <ChatPage userId={user.id} /> : <Navigate to="/login" replace />} 
      />
      <Route path="/" element={<Navigate to="/login" replace />} />
    </Routes>
  );
}

export default App;