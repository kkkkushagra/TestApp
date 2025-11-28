// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

// Ensure default theme is dark unless user explicitly chose light
(function initTheme() {
  try {
    const saved = localStorage.getItem('theme'); // 'dark' or 'light' or null
    if (saved === 'light') {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    } else {
      // default -> dark
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    }
  } catch (e) { /* ignore storage errors */ }
})();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
