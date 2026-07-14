import { useState, useRef, useEffect, useCallback, type KeyboardEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useBackend } from '@/hooks/useBackend';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  suggestions?: string[];
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  date: string;
}

const AUTH_KEY = 'nyaya-mitra-auth-logged-in';
const USERNAME_KEY = 'nyaya-mitra-auth-username';
const SESSIONS_KEY = 'nyaya-mitra-chat-sessions-v2';

const DEFAULT_WELCOME_MESSAGES: ChatMessage[] = [
  {
    id: 'welcome',
    role: 'assistant',
    content: "Hello! I'm Nyaya Mitra, your legal companion. Tell me what's going on, and I'll help you understand your situation and what to do next.",
    timestamp: new Date().toISOString(),
    suggestions: ['I got a notice from a bank', 'Someone is harassing me online', 'My landlord is threatening to evict me'],
  }
];

const INITIAL_MOCK_SESSIONS: ChatSession[] = [
  {
    id: 'session-1',
    title: 'Bank_Notice_Help.doc',
    date: '2026-07-12',
    messages: [
      { id: '1', role: 'user', content: 'I received a notice from my bank regarding non-payment.', timestamp: '2026-07-12T10:00:00Z' },
      {
        id: '2',
        role: 'assistant',
        content: 'Do not panic. You should review the notice period and look for details of any outstanding dues. You can reply formally explaining your circumstances or requesting a repayment schedule.',
        timestamp: '2026-07-12T10:01:00Z'
      }
    ]
  },
  {
    id: 'session-2',
    title: 'Online_Safety_Log.doc',
    date: '2026-07-13',
    messages: [
      { id: '1', role: 'user', content: 'Someone is harassing me online on social media.', timestamp: '2026-07-13T14:30:00Z' },
      {
        id: '2',
        role: 'assistant',
        content: 'This violates cyber safety regulations. First, document all evidence by taking screenshots. Do not engage with the harasser. You can file a formal complaint with the National Cyber Crime portal.',
        timestamp: '2026-07-13T14:31:00Z'
      }
    ]
  }
];

export default function Chat() {
  const navigate = useNavigate();
  const { query } = useBackend();

  // Auth States
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(() => {
    return localStorage.getItem(AUTH_KEY) === 'true';
  });
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const [usernameInput, setUsernameInput] = useState('');
  const [passwordInput, setPasswordInput] = useState('');
  const [currentUser, setCurrentUser] = useState<string>(() => {
    return localStorage.getItem(USERNAME_KEY) || 'Guest User';
  });

  // Chat Sessions States
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    const stored = localStorage.getItem(SESSIONS_KEY);
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch {
        // ignore
      }
    }
    return INITIAL_MOCK_SESSIONS;
  });

  // Window coordinates and focus stack states
  const [folderPos, setFolderPos] = useState({ x: 80, y: 50 });
  const [chatPos, setChatPos] = useState({ x: 340, y: 30 });
  const [focusedWindow, setFocusedWindow] = useState<'folder' | 'chat'>('chat');
  const [isMobile, setIsMobile] = useState(false);

  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [historyFolderOpen, setHistoryFolderOpen] = useState(false);
  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [systemTime, setSystemTime] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  interface SpeechRecognitionInstance {
    lang: string;
    continuous: boolean;
    interimResults: boolean;
    start(): void;
    stop(): void;
    onresult: ((event: any) => void) | null;
    onerror: ((event: { error: string }) => void) | null;
    onend: (() => void) | null;
  }

  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);

  // Sync clock and check viewport responsiveness
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setSystemTime(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);

    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);

    return () => {
      clearInterval(interval);
      window.removeEventListener('resize', checkMobile);
    };
  }, []);

  // Save sessions to localStorage
  useEffect(() => {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  }, [sessions]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSessionId, sessions, isTyping]);

  const activeSession = sessions.find(s => s.id === activeSessionId);
  const messages = activeSession ? activeSession.messages : [];

  const handleAuthSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!usernameInput.trim() || !passwordInput.trim()) {
      setError('Please fill in all fields.');
      return;
    }
    setError(null);
    localStorage.setItem(AUTH_KEY, 'true');
    localStorage.setItem(USERNAME_KEY, usernameInput);
    setCurrentUser(usernameInput);
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    localStorage.removeItem(AUTH_KEY);
    localStorage.removeItem(USERNAME_KEY);
    setIsLoggedIn(false);
    setActiveSessionId(null);
    setHistoryFolderOpen(false);
  };

  const handleCreateNewChat = () => {
    const newId = 'session-' + Date.now();
    const newSession: ChatSession = {
      id: newId,
      title: `Case_Log_${sessions.length + 1}.doc`,
      date: new Date().toISOString().split('T')[0],
      messages: [...DEFAULT_WELCOME_MESSAGES]
    };
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newId);
    setFocusedWindow('chat');
  };

  const handleClearSession = () => {
    if (!activeSessionId) return;
    setSessions(prev => prev.map(s => {
      if (s.id === activeSessionId) {
        return { ...s, messages: [...DEFAULT_WELCOME_MESSAGES] };
      }
      return s;
    }));
    setError(null);
  };

  const handleCloseSession = () => {
    setActiveSessionId(null);
  };

  const extractSuggestions = useCallback((answer: string): string[] => {
    const lines = answer.split('\n').filter(l => l.trim().startsWith('-') || l.trim().match(/^\d+\./));
    const suggestions = lines
      .map(l => l.replace(/^[-*\d.]+/, '').trim().replace(/^["']|["']$/g, ''))
      .filter(l => l.length > 10 && l.length < 120)
      .slice(0, 3);

    if (suggestions.length < 3) {
      const defaults = ['Explain in simple words', 'What if I ignore this?', 'What does the law say?'];
      return [...suggestions, ...defaults].slice(0, 3);
    }
    return suggestions;
  }, []);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isTyping || !activeSessionId) return;

    setError(null);

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    // Update session messages
    setSessions(prev => prev.map(s => {
      if (s.id === activeSessionId) {
        return { ...s, messages: [...s.messages, userMsg] };
      }
      return s;
    }));
    setInput('');
    setIsTyping(true);

    try {
      const result = await query(text, 'standard');
      const suggestions = extractSuggestions(result.answer);

      const aiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.answer,
        timestamp: new Date().toISOString(),
        suggestions,
      };

      setSessions(prev => prev.map(s => {
        if (s.id === activeSessionId) {
          return { ...s, messages: [...s.messages, aiMsg] };
        }
        return s;
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Something went wrong. Please try again.';
      setError(errorMessage);
    } finally {
      setIsTyping(false);
    }
  }, [input, isTyping, query, activeSessionId, extractSuggestions]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    inputRef.current?.focus();
  };

  const startRecording = useCallback(() => {
    const SpeechRecognitionConstructor = (window as { SpeechRecognition?: new () => SpeechRecognitionInstance; webkitSpeechRecognition?: new () => SpeechRecognitionInstance }).SpeechRecognition
      || (window as { SpeechRecognition?: new () => SpeechRecognitionInstance; webkitSpeechRecognition?: new () => SpeechRecognitionInstance }).webkitSpeechRecognition;

    if (!SpeechRecognitionConstructor) {
      setError('Voice input is not supported in this browser.');
      return;
    }

    const recognition = new SpeechRecognitionConstructor();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onresult = (event: any) => {
      let transcript = '';
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setInput(prev => prev + transcript);
    };

    recognition.onerror = () => {
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
  }, []);

  const stopRecording = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setIsRecording(false);
  }, []);

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Draggable Window Handler
  const handleDragStart = (e: React.MouseEvent, windowType: 'folder' | 'chat') => {
    if (isMobile) return; // Disable dragging on mobile viewport

    setFocusedWindow(windowType);

    const target = e.target as HTMLElement;
    if (target.closest('button') || target.closest('input') || target.closest('textarea')) {
      return;
    }

    e.preventDefault();

    const startX = e.clientX;
    const startY = e.clientY;

    const initialPos = windowType === 'folder' ? folderPos : chatPos;
    const setPos = windowType === 'folder' ? setFolderPos : setChatPos;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      setPos({
        x: Math.max(0, initialPos.x + dx),
        y: Math.max(0, initialPos.y + dy)
      });
    };

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  return (
    <div className="min-h-screen retro-desktop-bg flex flex-col retro-font select-none text-black">
      {/* Top Menu Bar */}
      <header className="bg-[#eaeaea] border-b border-black h-8 flex items-center justify-between px-4 z-20 text-xs">
        <div className="flex items-center gap-4">
          <span className="font-bold cursor-default">Nyaya Mitra OS</span>
          <span className="cursor-default hover:bg-black hover:text-white px-2 py-0.5 rounded-none">File</span>
          <span className="cursor-default hover:bg-black hover:text-white px-2 py-0.5 rounded-none">Edit</span>
          {isLoggedIn && (
            <span className="cursor-pointer hover:bg-black hover:text-white px-2 py-0.5 rounded-none font-bold" onClick={handleLogout}>
              Shutdown ({currentUser})
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <span className="font-bold">{systemTime}</span>
        </div>
      </header>

      {/* Main Desktop Space */}
      <main className={`flex-1 p-4 md:p-8 relative flex flex-col lg:flex-row gap-6 items-center lg:items-start justify-start ${isMobile ? 'overflow-y-auto' : 'overflow-hidden'}`}>
        
        {/* Scenario 1: User needs to Login/Signup */}
        {!isLoggedIn ? (
          <div className="w-full max-w-sm retro-window mx-auto my-auto retro-shadow-large z-10">
            {/* Title Bar */}
            <div className="flex items-center justify-between px-2 py-1.5 border-b border-black bg-[#eaeaea]">
              <div className="w-4 h-4 border border-black bg-[#eaeaea] flex items-center justify-center">
                <span className="w-1.5 h-1.5 bg-black"></span>
              </div>
              <div className="flex-1 h-3 mx-2 retro-stripes"></div>
              <span className="font-bold px-1 text-xs uppercase">Security Access</span>
              <div className="flex-1 h-3 mx-2 retro-stripes"></div>
            </div>

            {/* Selector Tab Controls */}
            <div className="flex border-b border-black bg-gray-300">
              <button
                onClick={() => setAuthMode('login')}
                className={`flex-1 py-2 font-bold text-xs ${
                  authMode === 'login' ? 'bg-white text-black' : 'bg-gray-300 text-gray-700 hover:bg-gray-400'
                }`}
              >
                Sign In
              </button>
              <button
                onClick={() => setAuthMode('signup')}
                className={`flex-1 py-2 font-bold text-xs border-l border-black ${
                  authMode === 'signup' ? 'bg-white text-black' : 'bg-gray-300 text-gray-700 hover:bg-gray-400'
                }`}
              >
                Create Account
              </button>
            </div>

            <form onSubmit={handleAuthSubmit} className="p-5 space-y-4 bg-[#eaeaea]">
              {error && (
                <div className="text-xs text-red-800 font-bold border border-red-800 bg-red-100 p-2">
                  {error}
                </div>
              )}
              <div className="space-y-1">
                <label className="text-[10px] uppercase font-bold text-gray-600 block">Username</label>
                <input
                  type="text"
                  required
                  value={usernameInput}
                  onChange={(e) => setUsernameInput(e.target.value)}
                  className="w-full retro-input-sunken px-2 py-1.5 text-xs outline-none"
                />
              </div>

              <div className="space-y-1">
                <label className="text-[10px] uppercase font-bold text-gray-600 block">Password</label>
                <input
                  type="password"
                  required
                  value={passwordInput}
                  onChange={(e) => setPasswordInput(e.target.value)}
                  className="w-full retro-input-sunken px-2 py-1.5 text-xs outline-none"
                />
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  className="w-full retro-button py-2.5 text-xs font-bold text-black"
                >
                  {authMode === 'login' ? 'Proceed to OS' : 'Register Account'}
                </button>
              </div>
            </form>
          </div>
        ) : (
          <>
            {/* Desktop Icons Panel */}
            <div className="flex flex-row lg:flex-col gap-6 lg:gap-8 p-2 shrink-0 justify-center lg:justify-start items-center z-0">
              
              {/* Chat History Folder */}
              <div
                onClick={() => { setHistoryFolderOpen(true); setFocusedWindow('folder'); }}
                className={`group flex flex-col items-center cursor-pointer p-2 rounded border transition-all ${
                  historyFolderOpen
                    ? 'bg-white/20 border-white'
                    : 'border-transparent hover:bg-white/10 hover:border-black/20'
                } w-20 lg:w-24 text-center`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-yellow-100 fill-yellow-100/30"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z"/></svg>
                <span className="text-[9px] lg:text-[10px] font-bold text-white bg-black/60 px-1.5 py-0.5 mt-1 rounded leading-none break-all select-none">
                  Chat History
                </span>
              </div>

              {/* Action item to create new chat directly */}
              <div
                onClick={handleCreateNewChat}
                className="group flex flex-col items-center cursor-pointer p-2 hover:bg-white/10 rounded border border-transparent hover:border-black/20 w-20 lg:w-24 text-center transition-all"
              >
                <div className="relative mb-1">
                  <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-gray-200 fill-gray-200/20"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                  <div className="absolute inset-0 flex items-center justify-center pt-2 pl-1">
                    <span className="text-black font-extrabold text-xs lg:text-sm">+</span>
                  </div>
                </div>
                <span className="text-[9px] lg:text-[10px] font-bold text-white bg-black/60 px-1.5 py-0.5 mt-1 rounded leading-none break-all select-none">
                  New_Chat.doc
                </span>
              </div>
            </div>

            {/* Desktop Windows Overlay Container */}
            <div className={isMobile ? "w-full flex flex-col gap-6 z-10 pointer-events-auto" : "absolute inset-0 top-8 left-28 pointer-events-none"}>
              
              {/* Folder Window: Chat History Explorer */}
              {historyFolderOpen && (
                <div
                  onMouseDown={() => setFocusedWindow('folder')}
                  style={isMobile ? {} : {
                    position: 'absolute',
                    left: `${folderPos.x}px`,
                    top: `${folderPos.y}px`,
                    zIndex: focusedWindow === 'folder' ? 30 : 10,
                  }}
                  className={`w-full ${isMobile ? 'h-[40vh]' : 'lg:w-80 lg:h-[65vh]'} retro-window flex flex-col overflow-hidden retro-shadow-large pointer-events-auto`}
                >
                  {/* Draggable Title Bar */}
                  <div
                    onMouseDown={(e) => handleDragStart(e, 'folder')}
                    className={`flex items-center justify-between px-2 py-1.5 border-b border-black bg-[#eaeaea] select-none shrink-0 ${isMobile ? 'cursor-default' : 'cursor-move'}`}
                  >
                    <button
                      onClick={() => setHistoryFolderOpen(false)}
                      className="w-5 h-5 retro-button flex items-center justify-center text-xs font-bold"
                      title="Close Directory"
                    >
                      <span className="w-2.5 h-2.5 bg-black"></span>
                    </button>
                    <div className="flex-1 h-3 mx-2 retro-stripes"></div>
                    <span className="font-bold px-2 text-xs uppercase text-black select-none whitespace-nowrap">
                      Chat History
                    </span>
                    <div className="flex-1 h-3 mx-2 retro-stripes"></div>
                    <div className="w-5 h-5"></div>
                  </div>

                  {/* Directory Files Space */}
                  <div className="flex-1 bg-white p-4 overflow-y-auto grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-3 gap-4 items-start justify-items-center">
                    {sessions.map((sess) => (
                      <div
                        key={sess.id}
                        onClick={() => { setActiveSessionId(sess.id); setFocusedWindow('chat'); }}
                        className={`group flex flex-col items-center cursor-pointer p-2 rounded border transition-all ${
                          activeSessionId === sess.id
                            ? 'bg-gray-200 border-black'
                            : 'border-transparent hover:bg-gray-100 hover:border-black/20'
                        } w-20 text-center`}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-gray-600 fill-gray-100"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        <span className="text-[9px] font-bold text-black mt-1 leading-tight break-all select-none max-w-full block px-0.5">
                          {sess.title}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Chat Session Window */}
              {activeSessionId && (
                <div
                  onMouseDown={() => setFocusedWindow('chat')}
                  style={isMobile ? {} : {
                    position: 'absolute',
                    left: `${chatPos.x}px`,
                    top: `${chatPos.y}px`,
                    zIndex: focusedWindow === 'chat' ? 30 : 10,
                  }}
                  className={`w-full ${isMobile ? 'h-[65vh]' : 'max-w-2xl lg:h-[75vh]'} retro-window flex flex-col overflow-hidden retro-shadow-large pointer-events-auto`}
                >
                  
                  {/* Draggable Title Bar */}
                  <div
                    onMouseDown={(e) => handleDragStart(e, 'chat')}
                    className={`flex items-center justify-between px-2 py-1.5 border-b border-black bg-[#eaeaea] select-none shrink-0 ${isMobile ? 'cursor-default' : 'cursor-move'}`}
                  >
                    <button
                      onClick={handleCloseSession}
                      className="w-5 h-5 retro-button flex items-center justify-center text-xs font-bold"
                      title="Close File"
                    >
                      <span className="w-2.5 h-2.5 bg-black"></span>
                    </button>
                    
                    <div className="flex-1 h-3 mx-3 retro-stripes"></div>
                    
                    <span className="font-bold px-3 text-xs uppercase tracking-wider text-black select-none whitespace-nowrap">
                      {activeSession?.title}
                    </span>
                    
                    <div className="flex-1 h-3 mx-3 retro-stripes"></div>
                    
                    <div className="w-5 h-5"></div>
                  </div>

                  {/* Actions Header */}
                  <div className="bg-[#eaeaea] border-b border-black px-4 py-2 flex items-center justify-between shrink-0 text-xs">
                    <button
                      onClick={handleClearSession}
                      className="retro-button px-3 py-1 font-bold text-black flex items-center gap-1.5 text-[10px]"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                      Clear Log
                    </button>
                  </div>

                  {/* Messages */}
                  <div className="flex-1 overflow-y-auto p-4 md:p-5 space-y-5 bg-white select-text">
                    {messages.map((msg) => (
                      <div key={msg.id} className="space-y-2">
                        <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <div
                            className={`max-w-[90%] sm:max-w-[80%] border-1.5 border-black p-3.5 md:p-4 retro-shadow-small ${
                              msg.role === 'user' ? 'bg-white text-black' : 'bg-[#f6f6f3] text-black'
                            }`}
                          >
                            <div className="text-[9px] uppercase font-bold border-b border-black pb-1 mb-2.5 text-gray-500 tracking-wider">
                              {msg.role === 'user' ? 'User' : 'Nyaya Mitra GPT'} - {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                            <p className="text-xs sm:text-sm leading-relaxed whitespace-pre-line font-medium text-black">
                              {msg.content}
                            </p>
                          </div>
                        </div>

                        {/* Beveled suggestions */}
                        {msg.role === 'assistant' && msg.suggestions && msg.suggestions.length > 0 && (
                          <div className="flex flex-wrap gap-2 justify-start pl-1 py-1">
                            {msg.suggestions.map((s, i) => (
                              <button
                                key={i}
                                onClick={() => handleSuggestionClick(s)}
                                className="retro-button px-3.5 py-1.5 text-xs font-bold text-black"
                              >
                                {s}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}

                    {/* Blinking cursor typing */}
                    {isTyping && (
                      <div className="flex justify-start">
                        <div className="border-1.5 border-black bg-[#eaeaea] p-4 text-xs font-bold retro-shadow-small">
                          <span>Processing Query</span>
                          <span className="ml-1.5 inline-block w-2.5 h-3 bg-black animate-pulse"></span>
                        </div>
                      </div>
                    )}
                    
                    <div ref={messagesEndRef} />
                  </div>

                  {/* Input area */}
                  <div className="border-t border-black bg-[#eaeaea] p-4 shrink-0">
                    <div className="max-w-3xl mx-auto flex items-end gap-3">
                      {/* Voice button */}
                      <button
                        onClick={toggleRecording}
                        className={`w-10 h-10 md:w-11 md:h-11 retro-button flex items-center justify-center shrink-0 ${
                          isRecording ? 'bg-red-600 text-white' : ''
                        }`}
                        title="Voice Input"
                      >
                        {isRecording ? (
                          <span className="w-3.5 h-3.5 bg-white animate-ping rounded-full" />
                        ) : (
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                        )}
                      </button>

                      {/* Text box */}
                      <div className="flex-1 retro-input-sunken p-1.5 flex bg-white">
                        <textarea
                          ref={inputRef}
                          value={input}
                          onChange={(e) => setInput(e.target.value)}
                          onKeyDown={handleKeyDown}
                          placeholder="Type message... (Ctrl+Enter to send)"
                          rows={1}
                          disabled={isTyping}
                          className="w-full bg-transparent resize-none outline-none text-xs sm:text-sm text-black placeholder:text-gray-500 py-1 px-1.5 max-h-24 disabled:opacity-50"
                          style={{ minHeight: '34px' }}
                        />
                      </div>

                      {/* Send button */}
                      <button
                        onClick={handleSend}
                        disabled={!input.trim() || isTyping}
                        className="w-10 h-10 md:w-11 md:h-11 retro-button flex items-center justify-center shrink-0"
                        title="Send Message"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                      </button>
                    </div>
                    
                    <p className="text-[10px] text-gray-500 text-center mt-3 font-bold uppercase tracking-wide">
                      This is a general legal advisor assistant. Verify all answers independently.
                    </p>
                  </div>

                </div>
              )}

              {/* Desktop Window Empty State Placeholder */}
              {!historyFolderOpen && !activeSessionId && (
                <div className={`${isMobile ? 'py-8 w-full' : 'absolute inset-0 flex h-[75vh]'} flex flex-col items-center justify-center text-center select-none text-white/40 max-w-xl mx-auto`}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mb-2 opacity-40"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/></svg>
                  <p className="text-xs uppercase tracking-wider font-bold">Open Chat History folder or double-click New_Chat.doc to begin</p>
                </div>
              )}

            </div>
          </>
        )}
      </main>
    </div>
  );
}
