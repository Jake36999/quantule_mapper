import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; 
// FIX: Import from the correct file name (Dashboard.tsx)
import IRERDashboard from './Dashboard'; 

const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('Failed to find the root element');

const root = ReactDOM.createRoot(rootElement);

root.render(
  <React.StrictMode>
    <IRERDashboard />
  </React.StrictMode>
);