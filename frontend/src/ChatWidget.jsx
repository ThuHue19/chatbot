import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './ChatWidget.css';

const MAX_LINES = 3;

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [expandedMsgs, setExpandedMsgs] = useState(new Set());
  const [isIndependent, setIsIndependent] = useState(true);
  const [visible, setVisible] = useState(true); // nút ẩn hoàn toàn

  const toggleChat = () => setIsOpen(!isOpen);
  const closeChat = () => setVisible(false);

  const toggleExpand = (idx) => {
    setExpandedMsgs((prev) => {
      const newSet = new Set(prev);
      newSet.has(idx) ? newSet.delete(idx) : newSet.add(idx);
      return newSet;
    });
  };

  const sendQuestion = async () => {
    const question = input.trim();
    if (!question) return;

    setMessages((prev) => [...prev, { sender: 'user', text: question }]);
    setInput('');

    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          is_independent: isIndependent,
          force_no_cache: false,
        }),
      });

      const data = await res.json();
      let answer = '';

      if (data.results) {
        answer = data.results
          .map((result, idx) => {
            if (result && result.rows && result.rows.length > 0) {
              const header = result.columns.join(' | ');
              const rows = result.rows.map((row) => row.join(' | ')).join('\n');
              return `**Kết quả truy vấn #${idx + 1}:**\n${header}\n${rows}`;
            } else {
              return 'Tôi chưa có thông tin, bạn liên hệ tổng đài nhé.';
            }
          })
          .join('\n\n');
      } else {
        answer = data.answer || '❌ Không có câu trả lời phù hợp.';
      }

      setMessages((prev) => [...prev, { sender: 'bot', text: answer }]);
    } catch (err) {
      console.error('Lỗi khi gửi câu hỏi:', err);
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: '❌ Lỗi khi gửi câu hỏi. Vui lòng thử lại.' },
      ]);
    }
  };

  if (!visible) return null;

  return (
    <div className="chat-container">
      {!isOpen && (
        <button className="chat-toggle" onClick={toggleChat}>
          💬
        </button>
      )}

      {isOpen && (
        <div className="chat-box">
          <div className="chat-header">
            <span>🤖</span>
            <div>
              <button onClick={toggleChat} className="minimize-btn">–</button>
            </div>
          </div>

          <div className="chat-body">
            {messages.map((msg, idx) => {
              const isExpanded = expandedMsgs.has(idx);
              const isBot = msg.sender === 'bot';
              const lines = msg.text.split('\n');
              const shouldTruncate = isBot && lines.length > MAX_LINES;
              const displayText =
                shouldTruncate && !isExpanded
                  ? lines.slice(0, MAX_LINES).join('\n') + '\n...'
                  : msg.text;

              return (
                <div className={`message ${msg.sender}`} key={idx}>
                  <div className="avatar">
                    {msg.sender === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className={msg.sender === 'user' ? 'user-msg' : 'bot-msg'}>
                    <ReactMarkdown
                      components={{
                        a: ({ node, ...props }) => (
                          <a {...props} target="_blank" rel="noopener noreferrer" />
                        ),
                        ul: ({ children }) => <>{children}</>,
                        li: ({ children }) => <div style={{ marginLeft: '1em' }}>{children}</div>,
                      }}
                    >
                      {displayText}
                    </ReactMarkdown>

                    {shouldTruncate && (
                      <button
                        onClick={() => toggleExpand(idx)}
                        className="toggle-more-btn"
                      >
                        {isExpanded ? 'Ẩn bớt' : 'Hiển thị thêm'}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="chat-input">
            <input
              type="text"
              placeholder="Nhập câu hỏi..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendQuestion()}
            />
            <button onClick={sendQuestion} title="Gửi">
              <svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24" width="18" height="18">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>

          <div className="context-options">
            <label>
              <input
                type="radio"
                name="context"
                value="new"
                checked={isIndependent}
                onChange={() => setIsIndependent(true)}
              />
              Hỏi mới
            </label>
            <label>
              <input
                type="radio"
                name="context"
                value="continue"
                checked={!isIndependent}
                onChange={() => setIsIndependent(false)}
              />
              Tiếp tục
            </label>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWidget;
