import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import '../index.css';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Router>{children}</Router>
      </body>
    </html>
  );
}