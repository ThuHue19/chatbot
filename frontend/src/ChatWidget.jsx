import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './ChatWidget.css';
import '@fortawesome/fontawesome-free/css/all.min.css';
const MAX_LINES = 3;

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [expandedMsgs, setExpandedMsgs] = useState(new Set());
  const [isIndependent, setIsIndependent] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  const toggleTheme = () => {
        setIsDarkMode(!isDarkMode);
        document.body.classList.toggle('dark-mode', !isDarkMode);
      };

  // Quản lý popup thu âm
  const [isRecordingPopupOpen, setIsRecordingPopupOpen] = useState(false);
  const [isListening, setIsListening] = useState(false);

  const endOfMessagesRef = useRef(null);
  const recognitionRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn('Trình duyệt không hỗ trợ SpeechRecognition');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'vi-VN';
    recognition.interimResults = false;
    recognition.continuous = false;

    recognition.onstart = () => {
      setIsListening(true);
    };
    recognition.onend = () => {
      setIsListening(false);
      setIsRecordingPopupOpen(false);
      inputRef.current?.focus(); // Focus lại input sau khi thu âm
    };
    recognition.onerror = (event) => {
      console.error('Speech recognition error', event.error);
      setIsListening(false);
      setIsRecordingPopupOpen(false);
      inputRef.current?.focus();
    };
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
    };

    recognitionRef.current = recognition;
  }, []);

  useEffect(() => {
    setMessages([
      {
        sender: 'bot',
        text: '👋 Chào bạn, tôi có thể giúp gì cho bạn? Lưu ý câu hỏi càng chi tiết thì tôi sẽ trả lời càng đúng. Hãy bắt đầu cuộc hội thoại hôm nay của chúng ta nhé!'
      }
    ]);
  }, []);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleChat = () => setIsOpen(!isOpen);

  const toggleExpand = (idx) => {
    setExpandedMsgs((prev) => {
      const newSet = new Set(prev);
      newSet.has(idx) ? newSet.delete(idx) : newSet.add(idx);
      return newSet;
    });
  };

  // Bật popup thu âm và bắt đầu thu âm
  const openRecordingPopup = () => {
    if (recognitionRef.current && !isListening) {
      setIsRecordingPopupOpen(true);
      recognitionRef.current.start();
    }
  };

  const buttonStyle = {
    backgroundColor: 'var(--primary-color)',
    border: 'none',
    borderRadius: '50%',
    width: 40,
    height: 40,
    cursor: 'pointer',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const sendQuestion = async () => {
    const question = input.trim();
    if (!question) return;

    setMessages((prev) => [...prev, { sender: 'user', text: question }]);
    setInput('');

    const typingIndicator = {
      sender: 'bot',
      text: (
        <div className="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      )
    };
    setMessages((prev) => [...prev, typingIndicator]);
    setIsLoading(true);

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

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

      const data = await res.json();
      const answer = data.answer || 'Tôi chưa có thông tin, bạn liên hệ tổng đài nhé.';

      setMessages((prev) => {
        const updated = [...prev];
        updated.pop();
        return [...updated, { sender: 'bot', text: answer }];
      });
    } catch (err) {
      console.error('❌ Lỗi khi gửi câu hỏi:', err);
      setMessages((prev) => {
        const updated = [...prev];
        updated.pop();
        return [...updated, { sender: 'bot', text: 'Tôi chưa có thông tin, bạn liên hệ tổng đài nhé.' }];
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {!isOpen && (
        <button className="chat-toggle" onClick={toggleChat}>
  <i className="fa-solid fa-comments"></i>
</button>

      )}

      {isOpen && (
        <div className="chat-box">
          <div className="chat-header">
  {/* Logo MobiFone */}
  <img
    src="/assets/Logo Mobifone.png" // logo đã lưu trong public/assets
    alt="MobiFone"
    className="logo"
  />
{/* Nút chuyển sáng/tối */}
  <button 
    className="theme-toggle-btn" 
    onClick={toggleTheme}
    title={isDarkMode ? "Chế độ sáng" : "Chế độ tối"}
  >
    <i className={`fa-solid ${isDarkMode ? "fa-sun" : "fa-moon"}`}></i>
  </button>


  {/* Nút thu nhỏ */}
  <button onClick={toggleChat} className="minimize-btn">–</button>
</div>


          <div className="chat-body">
            {messages.map((msg, idx) => {
              const isExpanded = expandedMsgs.has(idx);
              const isBot = msg.sender === 'bot';
              const textContent = typeof msg.text === 'string' ? msg.text : '';
              const lines = textContent.split('\n');
              const shouldTruncate = isBot && lines.length > MAX_LINES;
              const displayText =
                shouldTruncate && !isExpanded
                  ? lines.slice(0, MAX_LINES).join('\n') + '\n...'
                  : textContent;

              return (
                <div className={`message ${msg.sender}`} key={idx}>
                  <div className="avatar">
  {msg.sender === 'user' ? (
    <i className="fa-regular fa-user"></i>
  ) : (
    <i className="fa-solid fa-robot"></i>
  )}
</div>

                  <div className={isBot ? 'bot-msg' : 'user-msg'}>
                    {typeof msg.text === 'string' ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: ({ node, ...props }) => (
                            <a {...props} target="_blank" rel="noopener noreferrer">
                              {props.children}
                            </a>
                          ),
                          ul: ({ children }) => <>{children}</>,
                          li: ({ children }) => <div style={{ marginLeft: '1em' }}>{children}</div>,
                        }}
                      >
                        {displayText}
                      </ReactMarkdown>
                    ) : (
                      msg.text
                    )}

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
            <div ref={endOfMessagesRef} />
          </div>

          <div className="chat-input" style={{ display: 'flex', alignItems: 'center' }}>
            <input
              type="text"
              placeholder="Nhập tin nhắn dưới 1000 ký tự nhé!"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  if (isRecordingPopupOpen) {
                    recognitionRef.current && recognitionRef.current.stop();
                    setIsRecordingPopupOpen(false);
                  } else {
                    sendQuestion();
                  }
                }
              }}
              style={{ flex: 1 }}
              disabled={isLoading}
              ref={inputRef}
            />
            <button onClick={sendQuestion} title="Gửi" style={{ ...buttonStyle, marginLeft: 8 }} disabled={isLoading}>
              <svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24" width="20" height="20">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>

            <button
              onClick={openRecordingPopup}
              title="Nói để nhập"
              style={buttonStyle}
              disabled={isLoading}
            >
              <svg
    xmlns="http://www.w3.org/2000/svg"
    version="1.1"
    width="18"
    height="18"
    viewBox="0 0 256 256"
    xmlSpace="preserve"
    fill="white"  // hoặc màu bạn muốn, có thể đổi fill thành 'white' hoặc 'black'
  >
    <g
      transform="translate(1.4066 1.4066) scale(2.81 2.81)"
      style={{
        stroke: 'none',
        strokeWidth: 0,
        strokeDasharray: 'none',
        strokeLinecap: 'butt',
        strokeLinejoin: 'miter',
        strokeMiterlimit: 10,
        fillRule: 'nonzero',
        opacity: 1,
      }}
    >
      <path d="M 45 70.968 c -16.013 0 -29.042 -13.028 -29.042 -29.042 c 0 -1.712 1.388 -3.099 3.099 -3.099 c 1.712 0 3.099 1.388 3.099 3.099 C 22.157 54.522 32.404 64.77 45 64.77 c 12.595 0 22.843 -10.248 22.843 -22.843 c 0 -1.712 1.387 -3.099 3.099 -3.099 s 3.099 1.388 3.099 3.099 C 74.042 57.94 61.013 70.968 45 70.968 z"/>
      <path d="M 45 60.738 L 45 60.738 c -10.285 0 -18.7 -8.415 -18.7 -18.7 V 18.7 C 26.3 8.415 34.715 0 45 0 h 0 c 10.285 0 18.7 8.415 18.7 18.7 v 23.337 C 63.7 52.322 55.285 60.738 45 60.738 z"/>
      <path d="M 45 89.213 c -1.712 0 -3.099 -1.387 -3.099 -3.099 V 68.655 c 0 -1.712 1.388 -3.099 3.099 -3.099 c 1.712 0 3.099 1.387 3.099 3.099 v 17.459 C 48.099 87.826 46.712 89.213 45 89.213 z"/>
      <path d="M 55.451 90 H 34.549 c -1.712 0 -3.099 -1.387 -3.099 -3.099 s 1.388 -3.099 3.099 -3.099 h 20.901 c 1.712 0 3.099 1.387 3.099 3.099 S 57.163 90 55.451 90 z"/>
    </g>
  </svg>
            </button>
          </div>

          {isRecordingPopupOpen && (
  <div className="recording-popup">
    <div className="recording-popup-content">
      <div className="recording-icon" style={{ marginBottom: 10 }}>
        {isListening ? (
          // Icon micro khi đang nghe (đổi thành icon bạn gửi, đổi màu fill thành đỏ)
          <svg
            xmlns="http://www.w3.org/2000/svg"
            version="1.1"
            width="40"
            height="40"
            viewBox="0 0 256 256"
            xmlSpace="preserve"
            fill="red"
          >
            <g
              transform="translate(1.4066 1.4066) scale(2.81 2.81)"
              style={{
                stroke: 'none',
                strokeWidth: 0,
                strokeDasharray: 'none',
                strokeLinecap: 'butt',
                strokeLinejoin: 'miter',
                strokeMiterlimit: 10,
                fillRule: 'nonzero',
                opacity: 1,
              }}
            >
              <path
                d="M 45 70.968 c -16.013 0 -29.042 -13.028 -29.042 -29.042 c 0 -1.712 1.388 -3.099 3.099 -3.099 c 1.712 0 3.099 1.388 3.099 3.099 C 22.157 54.522 32.404 64.77 45 64.77 c 12.595 0 22.843 -10.248 22.843 -22.843 c 0 -1.712 1.387 -3.099 3.099 -3.099 s 3.099 1.388 3.099 3.099 C 74.042 57.94 61.013 70.968 45 70.968 z"
                fill="red"
              />
              <path
                d="M 45 60.738 L 45 60.738 c -10.285 0 -18.7 -8.415 -18.7 -18.7 V 18.7 C 26.3 8.415 34.715 0 45 0 h 0 c 10.285 0 18.7 8.415 18.7 18.7 v 23.337 C 63.7 52.322 55.285 60.738 45 60.738 z"
                fill="red"
              />
              <path
                d="M 45 89.213 c -1.712 0 -3.099 -1.387 -3.099 -3.099 V 68.655 c 0 -1.712 1.388 -3.099 3.099 -3.099 c 1.712 0 3.099 1.387 3.099 3.099 v 17.459 C 48.099 87.826 46.712 89.213 45 89.213 z"
                fill="red"
              />
              <path
                d="M 55.451 90 H 34.549 c -1.712 0 -3.099 -1.387 -3.099 -3.099 s 1.388 -3.099 3.099 -3.099 h 20.901 c 1.712 0 3.099 1.387 3.099 3.099 S 57.163 90 55.451 90 z"
                fill="red"
              />
            </g>
          </svg>
        ) : (
          // Icon micro khi chưa nghe (màu đen, fill đen)
          <svg
            xmlns="http://www.w3.org/2000/svg"
            version="1.1"
            width="40"
            height="40"
            viewBox="0 0 256 256"
            xmlSpace="preserve"
            fill="black"
          >
            <g
              transform="translate(1.4066 1.4066) scale(2.81 2.81)"
              style={{
                stroke: 'none',
                strokeWidth: 0,
                strokeDasharray: 'none',
                strokeLinecap: 'butt',
                strokeLinejoin: 'miter',
                strokeMiterlimit: 10,
                fillRule: 'nonzero',
                opacity: 1,
              }}
            >
              <path d="M 45 70.968 c -16.013 0 -29.042 -13.028 -29.042 -29.042 c 0 -1.712 1.388 -3.099 3.099 -3.099 c 1.712 0 3.099 1.388 3.099 3.099 C 22.157 54.522 32.404 64.77 45 64.77 c 12.595 0 22.843 -10.248 22.843 -22.843 c 0 -1.712 1.387 -3.099 3.099 -3.099 s 3.099 1.388 3.099 3.099 C 74.042 57.94 61.013 70.968 45 70.968 z" />
              <path d="M 45 60.738 L 45 60.738 c -10.285 0 -18.7 -8.415 -18.7 -18.7 V 18.7 C 26.3 8.415 34.715 0 45 0 h 0 c 10.285 0 18.7 8.415 18.7 18.7 v 23.337 C 63.7 52.322 55.285 60.738 45 60.738 z" />
              <path d="M 45 89.213 c -1.712 0 -3.099 -1.387 -3.099 -3.099 V 68.655 c 0 -1.712 1.388 -3.099 3.099 -3.099 c 1.712 0 3.099 1.387 3.099 3.099 v 17.459 C 48.099 87.826 46.712 89.213 45 89.213 z" />
              <path d="M 55.451 90 H 34.549 c -1.712 0 -3.099 -1.387 -3.099 -3.099 s 1.388 -3.099 3.099 -3.099 h 20.901 c 1.712 0 3.099 1.387 3.099 3.099 S 57.163 90 55.451 90 z" />
            </g>
          </svg>
        )}
      </div>
      <div>
        {isListening ? 'Tôi đang nghe, bạn hãy nói...' : 'Bấm nút micro để bắt đầu'}
      </div>
    </div>
  </div>
)}


          <div className="context-options">
  <label className="context-label">
    <input
      type="radio"
      name="context"
      value="new"
      checked={isIndependent}
      onChange={() => setIsIndependent(true)}
    />
    <span className="context-text">Hỏi mới</span>
  </label>
  <label className="context-label">
    <input
      type="radio"
      name="context"
      value="continue"
      checked={!isIndependent}
      onChange={() => setIsIndependent(false)}
    />
    <span className="context-text">Tiếp tục</span>
  </label>
</div>

        </div>
      )}
    </div>
  );
};

export default ChatWidget;
