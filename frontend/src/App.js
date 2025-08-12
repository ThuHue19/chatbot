import React, { useEffect } from 'react';
import ChatWidget from './ChatWidget';
import './App.css';

function App() {
  // Tạo hiệu ứng tuyết rơi
  useEffect(() => {
    const snowContainer = document.createElement('div');
    snowContainer.className = 'snow-container';
    document.body.appendChild(snowContainer);

    function createSnowflake() {
      const snowflake = document.createElement('div');
      snowflake.className = 'snowflake';
      snowflake.textContent = '❄';
      snowflake.style.left = Math.random() * window.innerWidth + 'px';
      snowflake.style.fontSize = Math.random() * 10 + 10 + 'px';
      snowflake.style.animationDuration = Math.random() * 3 + 2 + 's';
      snowflake.style.opacity = Math.random();
      snowContainer.appendChild(snowflake);

      setTimeout(() => {
        snowflake.remove();
      }, 5000);
    }

    const snowInterval = setInterval(createSnowflake, 200);
    return () => clearInterval(snowInterval);
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        ❄️ Chào mừng bạn đến với Chatbot ❄️
      </header>

      <ChatWidget />

      <footer className="App-footer">
        © 2025 Hoàng Minh Diệp & Nguyễn Thị Thu Huệ - Đại học Khoa học Tự nhiên
      </footer>
    </div>
  );
}

export default App;
