import React from 'react';
import ChatWidget from './ChatWidget';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>🌸 Trợ lý thông minh 🌸</h1>
        <p>Hỏi gì cũng được, tớ sẽ giúp bạn hết mình!</p>
      </header>
      <ChatWidget />
    </div>
  );
}

export default App;
