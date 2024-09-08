import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import RootLayout from './app/layout';
import './styles/global.css';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <RootLayout>
      <App />
    </RootLayout>
  </React.StrictMode>
);