(function () {
	"use strict";

	// Configuration
	const config = {
		apiUrl: window.AI_CHATBOT_API_URL || "http://localhost:8000",
		domain: window.AI_CHATBOT_DOMAIN || window.location.hostname,
		position: window.AI_CHATBOT_POSITION || "bottom-right",
		theme: window.AI_CHATBOT_THEME || "modern",
		autoStart: window.AI_CHATBOT_AUTO_START !== false,
		welcomeDelay: 3000, // Delay before showing welcome message
		typingSpeed: 1000, // Base typing indicator duration
	};

	// Widget state
	let sessionId = localStorage.getItem("ai_chatbot_session") || null;
	let isOpen = false;
	let isMinimized = false;
	let isReady = false;
	let messageCount = 0;
	let lastMessageTime = Date.now();
	let conversationContext = {
		hasGreeted: false,
		userName: null,
		topics: [],
		sentiment: "neutral",
	};

	// Create widget HTML with improved design
	const widgetHTML = `
        <div id="ai-chatbot-container" class="ai-chatbot-container ai-chatbot-${config.position}">
            <div id="ai-chatbot-widget" class="ai-chatbot-widget ai-chatbot-hidden">
                <div class="ai-chatbot-header">
                    <div class="ai-chatbot-header-content">
                        <div class="ai-chatbot-status">
                            <span class="ai-chatbot-status-dot"></span>
                            <span class="ai-chatbot-status-text">AI Assistant</span>
                            <span class="ai-chatbot-status-detail">Ready to help</span>
                        </div>
                        <div class="ai-chatbot-header-actions">
                            <button class="ai-chatbot-minimize" onclick="AIChatbot.minimize()" title="Minimize">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M4 8H12" stroke="currentColor" stroke-width="2"/>
                                </svg>
                            </button>
                            <button class="ai-chatbot-close" onclick="AIChatbot.close()" title="Close">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M4 4L12 12M4 12L12 4" stroke="currentColor" stroke-width="2"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="ai-chatbot-messages" id="ai-chatbot-messages">
                    <div class="ai-chatbot-welcome" id="ai-chatbot-welcome">
                        <div class="ai-chatbot-avatar-large">
                            <svg width="40" height="40" viewBox="0 0 24 24" fill="none">
                                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <h3>Welcome! 👋</h3>
                        <p>I'm your AI assistant for ${config.domain}</p>
                        <div class="ai-chatbot-loading" id="ai-chatbot-loading">
                            <div class="ai-chatbot-spinner"></div>
                            <span>Checking knowledge base...</span>
                        </div>
                    </div>
                </div>
                
                <div class="ai-chatbot-suggestions" id="ai-chatbot-suggestions" style="display: none;">
                    <div class="ai-chatbot-suggestion" onclick="AIChatbot.sendSuggestion(this)">
                        What services do you offer?
                    </div>
                    <div class="ai-chatbot-suggestion" onclick="AIChatbot.sendSuggestion(this)">
                        How can I contact you?
                    </div>
                    <div class="ai-chatbot-suggestion" onclick="AIChatbot.sendSuggestion(this)">
                        Tell me more about this
                    </div>
                </div>
                
                <div class="ai-chatbot-input-container">
                    <div class="ai-chatbot-input-wrapper">
                        <textarea 
                            id="ai-chatbot-input" 
                            class="ai-chatbot-input"
                            placeholder="Type your message..."
                            disabled
                            rows="1"
                        ></textarea>
                        <button 
                            id="ai-chatbot-send" 
                            class="ai-chatbot-send"
                            onclick="AIChatbot.sendMessage()"
                            disabled
                            title="Send message"
                        >
                            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                                <path d="M2 10L18 2L10 18L8 11L2 10Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
                            </svg>
                        </button>
                    </div>
                    <div class="ai-chatbot-input-hint">Press Enter to send, Shift+Enter for new line</div>
                </div>
            </div>
            
            <button id="ai-chatbot-trigger" class="ai-chatbot-trigger" onclick="AIChatbot.toggle()">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="ai-chatbot-icon-chat">
                    <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C13.19 22 14.34 21.78 15.41 21.37L21 23L19.37 17.41C20.78 15.34 22 13.19 22 12C22 6.48 17.52 2 12 2ZM8 13C7.45 13 7 12.55 7 12C7 11.45 7.45 11 8 11C8.55 11 9 11.45 9 12C9 12.55 8.55 13 8 13ZM12 13C11.45 13 11 12.55 11 12C11 11.45 11.45 11 12 11C12.55 11 13 11.45 13 12C13 12.55 12.55 13 12 13ZM16 13C15.45 13 15 12.55 15 12C15 11.45 15.45 11 16 11C16.55 11 17 11.45 17 12C17 12.55 16.55 13 16 13Z" fill="currentColor"/>
                </svg>
                <span class="ai-chatbot-trigger-pulse"></span>
                <span class="ai-chatbot-badge" id="ai-chatbot-badge" style="display: none;">1</span>
            </button>
        </div>
    `;

	// Enhanced styles with better animations and natural feel
	const styles = `
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            
            .ai-chatbot-container {
                position: fixed;
                z-index: 999999;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .ai-chatbot-bottom-right {
                bottom: 20px;
                right: 20px;
            }
            
            .ai-chatbot-widget {
                width: 400px;
                height: 600px;
                background: #ffffff;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.15);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                transform-origin: bottom right;
            }
            
            .ai-chatbot-widget.ai-chatbot-hidden {
                opacity: 0;
                transform: scale(0.95) translateY(20px);
                pointer-events: none;
            }
            
            .ai-chatbot-widget.ai-chatbot-minimized {
                height: 64px;
            }
            
            .ai-chatbot-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 18px 20px;
                flex-shrink: 0;
            }
            
            .ai-chatbot-header-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .ai-chatbot-status {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .ai-chatbot-status-dot {
                width: 10px;
                height: 10px;
                background: #4ade80;
                border-radius: 50%;
                box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.3);
                animation: statusPulse 2s infinite;
            }
            
            @keyframes statusPulse {
                0% { 
                    transform: scale(1);
                    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.3);
                }
                50% { 
                    transform: scale(1.1);
                    box-shadow: 0 0 0 4px rgba(74, 222, 128, 0.1);
                }
                100% { 
                    transform: scale(1);
                    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.3);
                }
            }
            
            .ai-chatbot-status-text {
                font-weight: 600;
                font-size: 16px;
            }
            
            .ai-chatbot-status-detail {
                font-size: 12px;
                opacity: 0.9;
            }
            
            .ai-chatbot-header-actions {
                display: flex;
                gap: 8px;
            }
            
            .ai-chatbot-header-actions button {
                background: rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                border: none;
                width: 36px;
                height: 36px;
                border-radius: 10px;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .ai-chatbot-header-actions button:hover {
                background: rgba(255,255,255,0.3);
                transform: translateY(-1px);
            }
            
            .ai-chatbot-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fb;
                scroll-behavior: smooth;
            }
            
            .ai-chatbot-messages::-webkit-scrollbar {
                width: 6px;
            }
            
            .ai-chatbot-messages::-webkit-scrollbar-track {
                background: transparent;
            }
            
            .ai-chatbot-messages::-webkit-scrollbar-thumb {
                background: #e0e0e0;
                border-radius: 3px;
            }
            
            .ai-chatbot-welcome {
                text-align: center;
                padding: 60px 20px;
            }
            
            .ai-chatbot-avatar-large {
                width: 80px;
                height: 80px;
                margin: 0 auto 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                animation: float 3s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            
            .ai-chatbot-welcome h3 {
                margin: 0 0 8px 0;
                font-size: 24px;
                font-weight: 600;
                color: #1f2937;
            }
            
            .ai-chatbot-welcome p {
                margin: 0 0 24px 0;
                color: #6b7280;
                font-size: 16px;
            }
            
            .ai-chatbot-loading {
                display: inline-flex;
                align-items: center;
                gap: 12px;
                padding: 14px 24px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }
            
            .ai-chatbot-spinner {
                width: 20px;
                height: 20px;
                border: 2px solid #e5e7eb;
                border-top-color: #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .ai-chatbot-message {
                margin-bottom: 20px;
                display: flex;
                gap: 12px;
                animation: messageSlide 0.3s ease-out;
            }
            
            @keyframes messageSlide {
                from { 
                    opacity: 0;
                    transform: translateY(10px);
                }
                to { 
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .ai-chatbot-message-user {
                flex-direction: row-reverse;
            }
            
            .ai-chatbot-message-avatar {
                width: 36px;
                height: 36px;
                border-radius: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: 600;
                flex-shrink: 0;
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-avatar {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
            
            .ai-chatbot-message-content {
                max-width: 75%;
                padding: 14px 18px;
                background: white;
                border-radius: 18px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                position: relative;
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .ai-chatbot-message-text {
                margin: 0;
                font-size: 15px;
                line-height: 1.6;
                color: #374151;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-text {
                color: white;
            }
            
            .ai-chatbot-message-time {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 6px;
            }
            
            .ai-chatbot-message-sources {
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #e5e7eb;
            }
            
            .ai-chatbot-message-sources-title {
                font-size: 12px;
                color: #6b7280;
                margin-bottom: 6px;
                font-weight: 500;
            }
            
            .ai-chatbot-message-source {
                font-size: 13px;
                color: #667eea;
                text-decoration: none;
                display: inline-block;
                margin-right: 12px;
                margin-bottom: 4px;
                padding: 4px 8px;
                background: rgba(102, 126, 234, 0.1);
                border-radius: 6px;
                transition: all 0.2s;
            }
            
            .ai-chatbot-message-source:hover {
                background: rgba(102, 126, 234, 0.2);
                transform: translateY(-1px);
            }
            
            .ai-chatbot-suggestions {
                padding: 12px 20px;
                background: white;
                border-top: 1px solid #e5e7eb;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }
            
            .ai-chatbot-suggestion {
                padding: 8px 14px;
                background: #f3f4f6;
                border: 1px solid #e5e7eb;
                border-radius: 20px;
                font-size: 13px;
                color: #4b5563;
                cursor: pointer;
                transition: all 0.2s;
                white-space: nowrap;
            }
            
            .ai-chatbot-suggestion:hover {
                background: #667eea;
                color: white;
                border-color: #667eea;
                transform: translateY(-1px);
            }
            
            .ai-chatbot-input-container {
                padding: 20px;
                background: white;
                border-top: 1px solid #e5e7eb;
                flex-shrink: 0;
            }
            
            .ai-chatbot-input-wrapper {
                display: flex;
                gap: 12px;
                align-items: flex-end;
            }
            
            .ai-chatbot-input {
                flex: 1;
                padding: 14px 18px;
                border: 2px solid #e5e7eb;
                border-radius: 16px;
                font-size: 15px;
                font-family: inherit;
                outline: none;
                transition: all 0.2s;
                resize: none;
                max-height: 120px;
                line-height: 1.4;
            }
            
            .ai-chatbot-input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .ai-chatbot-input:disabled {
                background: #f9fafb;
                cursor: not-allowed;
            }
            
            .ai-chatbot-send {
                width: 48px;
                height: 48px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 14px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            .ai-chatbot-send:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            .ai-chatbot-send:active:not(:disabled) {
                transform: translateY(0);
            }
            
            .ai-chatbot-send:disabled {
                background: #9ca3af;
                cursor: not-allowed;
                box-shadow: none;
            }
            
            .ai-chatbot-input-hint {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 6px;
                text-align: center;
            }
            
            .ai-chatbot-trigger {
                width: 64px;
                height: 64px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 50%;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 6px 24px rgba(102, 126, 234, 0.4);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            .ai-chatbot-trigger:hover {
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.5);
            }
            
            .ai-chatbot-trigger-pulse {
                position: absolute;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, transparent 70%);
                border-radius: 50%;
                animation: triggerPulse 2s infinite;
            }
            
            @keyframes triggerPulse {
                0% {
                    transform: scale(0.8);
                    opacity: 0;
                }
                50% {
                    opacity: 0.3;
                }
                100% {
                    transform: scale(1.5);
                    opacity: 0;
                }
            }
            
            .ai-chatbot-badge {
                position: absolute;
                top: -4px;
                right: -4px;
                background: #ef4444;
                color: white;
                font-size: 12px;
                font-weight: 600;
                min-width: 22px;
                height: 22px;
                border-radius: 11px;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0 6px;
                box-shadow: 0 2px 8px rgba(239, 68, 68, 0.4);
                animation: badgeBounce 0.5s ease-out;
            }
            
            @keyframes badgeBounce {
                0% { transform: scale(0); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
            }
            
            .ai-chatbot-typing {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 8px 12px;
            }
            
            .ai-chatbot-typing span {
                width: 8px;
                height: 8px;
                background: #667eea;
                border-radius: 50%;
                animation: typing 1.4s ease-in-out infinite;
            }
            
            .ai-chatbot-typing span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .ai-chatbot-typing span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                30% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            @media (max-width: 480px) {
                .ai-chatbot-widget {
                    width: 100vw;
                    height: 100vh;
                    border-radius: 0;
                    max-height: 100vh;
                }
                
                .ai-chatbot-container {
                    bottom: 0;
                    right: 0;
                }
                
                .ai-chatbot-trigger {
                    bottom: 20px;
                    right: 20px;
                }
                
                .ai-chatbot-suggestions {
                    flex-wrap: nowrap;
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                }
                
                .ai-chatbot-suggestion {
                    flex-shrink: 0;
                }
            }
        </style>
    `;

	// Inject HTML and styles
	document.head.insertAdjacentHTML("beforeend", styles);
	document.body.insertAdjacentHTML("beforeend", widgetHTML);

	// API class with improved error handling
	class ChatbotAPI {
		constructor(apiUrl) {
			this.apiUrl = apiUrl;
			this.retryCount = 3;
			this.retryDelay = 1000;
		}

		async request(url, options = {}) {
			for (let i = 0; i < this.retryCount; i++) {
				try {
					const response = await fetch(url, {
						...options,
						headers: {
							"Content-Type": "application/json",
							...options.headers,
						},
					});

					if (!response.ok) {
						const error = await response.json().catch(() => ({ detail: "Unknown error" }));
						throw new Error(error.detail || `HTTP ${response.status}`);
					}

					return response.json();
				} catch (error) {
					if (i === this.retryCount - 1) throw error;
					await new Promise((resolve) => setTimeout(resolve, this.retryDelay * (i + 1)));
				}
			}
		}

		async checkDomainReady(domain) {
			try {
				await this.request(`${this.apiUrl}/api/chat`, {
					method: "POST",
					body: JSON.stringify({
						question: "test",
						domain: domain,
						session_id: "test-session",
					}),
				});
				return true;
			} catch (error) {
				return false;
			}
		}

		async sendMessage(question, sessionId, domain) {
			return this.request(`${this.apiUrl}/api/chat`, {
				method: "POST",
				body: JSON.stringify({
					question,
					session_id: sessionId,
					domain: domain,
					require_reasoning: true,
				}),
			});
		}
	}

	// Initialize API
	const api = new ChatbotAPI(config.apiUrl);

	// Main chatbot class with enhanced functionality
	class AIChatbot {
		constructor() {
			this.widget = document.getElementById("ai-chatbot-widget");
			this.trigger = document.getElementById("ai-chatbot-trigger");
			this.messages = document.getElementById("ai-chatbot-messages");
			this.input = document.getElementById("ai-chatbot-input");
			this.sendButton = document.getElementById("ai-chatbot-send");
			this.badge = document.getElementById("ai-chatbot-badge");
			this.suggestions = document.getElementById("ai-chatbot-suggestions");
			this.domain = config.domain;

			this.init();
		}

		async init() {
			console.log("🤖 Initializing AI Chatbot for:", this.domain);

			// Setup event listeners
			this.setupEventListeners();

			// Check domain status
			await this.checkDomainStatus();

			// Setup auto-resize for textarea
			this.setupAutoResize();
		}

		setupEventListeners() {
			// Input handling
			this.input.addEventListener("keypress", (e) => {
				if (e.key === "Enter" && !e.shiftKey) {
					e.preventDefault();
					this.sendMessage();
				}
			});

			this.input.addEventListener("input", () => {
				this.adjustTextareaHeight();
				this.updateSendButtonState();
			});

			// Focus input when clicking messages area
			this.messages.addEventListener("click", (e) => {
				if (e.target === this.messages && isReady) {
					this.input.focus();
				}
			});
		}

		setupAutoResize() {
			const resize = () => {
				this.input.style.height = "auto";
				this.input.style.height = Math.min(this.input.scrollHeight, 120) + "px";
			};
			this.input.addEventListener("input", resize);
		}

		adjustTextareaHeight() {
			this.input.style.height = "auto";
			this.input.style.height = Math.min(this.input.scrollHeight, 120) + "px";
		}

		updateSendButtonState() {
			const hasText = this.input.value.trim().length > 0;
			this.sendButton.disabled = !hasText || !isReady;
		}

		async checkDomainStatus() {
			try {
				const statusElement = document.querySelector(".ai-chatbot-status-detail");
				statusElement.textContent = "Connecting...";

				const ready = await api.checkDomainReady(this.domain);

				if (ready) {
					console.log("✅ Domain ready for chat");
					this.onReady();
				} else {
					console.log("⚠️ Domain not analyzed");
					this.showError("This website hasn't been analyzed yet. Please set it up first.");
				}
			} catch (error) {
				console.error("❌ Failed to check domain:", error);
				this.showError("Connection failed. Please try again.");
			}
		}

		onReady() {
			isReady = true;
			this.input.disabled = false;
			this.input.placeholder = "Type your message...";

			const statusElement = document.querySelector(".ai-chatbot-status-detail");
			statusElement.textContent = "Online";

			// Clear loading state
			const loading = document.getElementById("ai-chatbot-loading");
			if (loading) loading.style.display = "none";

			// Show welcome message after a delay
			setTimeout(() => {
				if (!conversationContext.hasGreeted) {
					this.showWelcomeMessage();
				}
			}, config.welcomeDelay);

			// Show badge if closed
			if (!isOpen) {
				setTimeout(() => this.showBadge(), config.welcomeDelay + 1000);
			}
		}

		showWelcomeMessage() {
			const welcome = document.getElementById("ai-chatbot-welcome");
			if (welcome) welcome.style.display = "none";

			this.addMessage("bot", `Hi there! 👋 I'm your AI assistant for ${this.domain}. I've studied everything about this website and I'm here to help you find what you need. What would you like to know?`);

			// Show suggestions
			this.showSuggestions(["What services do you offer?", "How can I contact you?", "Tell me about your company", "What are your hours?"]);

			conversationContext.hasGreeted = true;
		}

		showSuggestions(suggestions) {
			this.suggestions.innerHTML = "";
			suggestions.forEach((text) => {
				const suggestion = document.createElement("div");
				suggestion.className = "ai-chatbot-suggestion";
				suggestion.textContent = text;
				suggestion.onclick = () => this.sendSuggestion(suggestion);
				this.suggestions.appendChild(suggestion);
			});
			this.suggestions.style.display = "flex";
		}

		hideSuggestions() {
			this.suggestions.style.display = "none";
		}

		sendSuggestion(element) {
			const text = element.textContent;
			this.input.value = text;
			this.adjustTextareaHeight();
			this.sendMessage();
		}

		showError(message) {
			const loading = document.getElementById("ai-chatbot-loading");
			if (loading) {
				loading.innerHTML = `<span style="color: #ef4444;">${message}</span>`;
			}
		}

		toggle() {
			if (isOpen) {
				this.close();
			} else {
				this.open();
			}
		}

		open() {
			isOpen = true;
			this.widget.classList.remove("ai-chatbot-hidden");
			this.trigger.style.display = "none";
			this.hideBadge();

			if (isReady) {
				setTimeout(() => this.input.focus(), 300);
			}

			// Track opening
			this.trackEvent("chat_opened");
		}

		close() {
			isOpen = false;
			this.widget.classList.add("ai-chatbot-hidden");
			this.trigger.style.display = "flex";

			// Track closing
			this.trackEvent("chat_closed", { message_count: messageCount });
		}

		minimize() {
			isMinimized = !isMinimized;
			this.widget.classList.toggle("ai-chatbot-minimized");
		}

		showBadge() {
			this.badge.style.display = "flex";
			this.badge.textContent = "1";
		}

		hideBadge() {
			this.badge.style.display = "none";
		}

		async sendMessage() {
			const message = this.input.value.trim();

			if (!message || !isReady) return;

			// Clear input immediately
			this.input.value = "";
			this.adjustTextareaHeight();
			this.updateSendButtonState();
			this.hideSuggestions();

			// Add user message
			this.addMessage("user", message);

			// Update context
			messageCount++;
			lastMessageTime = Date.now();

			// Show typing indicator with dynamic duration
			const typingDuration = this.calculateTypingDuration(message);
			const typingId = this.showTyping();

			try {
				// Send to API
				const response = await api.sendMessage(message, sessionId, this.domain);

				// Store session ID
				if (response.session_id) {
					sessionId = response.session_id;
					localStorage.setItem("ai_chatbot_session", sessionId);
				}

				// Simulate natural typing delay
				const remainingDelay = typingDuration - (Date.now() - lastMessageTime);
				if (remainingDelay > 0) {
					await new Promise((resolve) => setTimeout(resolve, remainingDelay));
				}

				// Remove typing indicator
				this.removeTyping(typingId);

				// Add bot response
				this.addMessage("bot", response.answer, response.sources);

				// Update conversation context
				this.updateConversationContext(message, response);

				// Show relevant suggestions based on response
				this.generateContextualSuggestions(response);
			} catch (error) {
				console.error("Failed to send message:", error);
				this.removeTyping(typingId);
				this.addMessage("bot", "I apologize, but I'm having trouble connecting right now. Please try again in a moment, or check your internet connection.");
			}
		}

		calculateTypingDuration(message) {
			// Simulate natural typing speed based on message length
			const wordsPerMinute = 300;
			const words = message.split(" ").length;
			const baseTime = (words / wordsPerMinute) * 60 * 1000;
			return Math.max(config.typingSpeed, Math.min(baseTime, 3000));
		}

		updateConversationContext(userMessage, response) {
			// Analyze sentiment
			const positiveSentiments = ["thanks", "great", "awesome", "perfect", "excellent", "good"];
			const negativeSentiments = ["bad", "wrong", "incorrect", "unhappy", "disappointed"];

			const messageLower = userMessage.toLowerCase();
			if (positiveSentiments.some((word) => messageLower.includes(word))) {
				conversationContext.sentiment = "positive";
			} else if (negativeSentiments.some((word) => messageLower.includes(word))) {
				conversationContext.sentiment = "negative";
			}

			// Extract topics
			if (response.query_type) {
				conversationContext.topics.push(response.query_type);
			}
		}

		generateContextualSuggestions(response) {
			const suggestions = [];

			// Based on query type
			if (response.query_type === "simple") {
				suggestions.push("Tell me more about this");
				suggestions.push("What else should I know?");
			} else if (response.query_type === "complex") {
				suggestions.push("Can you simplify that?");
				suggestions.push("Show me an example");
			}

			// Based on sentiment
			if (conversationContext.sentiment === "negative") {
				suggestions.push("Contact support");
				suggestions.push("Report an issue");
			}

			// Always include a general option
			suggestions.push("Ask something else");

			if (suggestions.length > 0) {
				setTimeout(() => this.showSuggestions(suggestions), 500);
			}
		}

		addMessage(sender, text, sources = []) {
			const messageDiv = document.createElement("div");
			messageDiv.className = `ai-chatbot-message ai-chatbot-message-${sender}`;

			const avatar = document.createElement("div");
			avatar.className = "ai-chatbot-message-avatar";
			avatar.textContent = sender === "user" ? "You" : "AI";

			const content = document.createElement("div");
			content.className = "ai-chatbot-message-content";

			const textP = document.createElement("p");
			textP.className = "ai-chatbot-message-text";
			textP.textContent = text;
			content.appendChild(textP);

			// Add timestamp
			const time = document.createElement("div");
			time.className = "ai-chatbot-message-time";
			time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
			content.appendChild(time);

			// Add sources if available
			if (sources && sources.length > 0) {
				const sourcesDiv = document.createElement("div");
				sourcesDiv.className = "ai-chatbot-message-sources";

				const sourcesTitle = document.createElement("div");
				sourcesTitle.className = "ai-chatbot-message-sources-title";
				sourcesTitle.textContent = "📎 Sources:";
				sourcesDiv.appendChild(sourcesTitle);

				sources.forEach((source) => {
					if (source && source.url) {
						const sourceLink = document.createElement("a");
						sourceLink.className = "ai-chatbot-message-source";
						sourceLink.href = source.url;
						sourceLink.textContent = source.title || "View source";
						sourceLink.target = "_blank";
						sourceLink.rel = "noopener noreferrer";
						sourcesDiv.appendChild(sourceLink);
					}
				});

				content.appendChild(sourcesDiv);
			}

			messageDiv.appendChild(avatar);
			messageDiv.appendChild(content);

			this.messages.appendChild(messageDiv);

			// Smooth scroll to bottom
			requestAnimationFrame(() => {
				this.messages.scrollTo({
					top: this.messages.scrollHeight,
					behavior: "smooth",
				});
			});
		}

		showTyping() {
			const typingId = `typing-${Date.now()}`;
			const typingDiv = document.createElement("div");
			typingDiv.id = typingId;
			typingDiv.className = "ai-chatbot-message ai-chatbot-message-bot";
			typingDiv.innerHTML = `
                <div class="ai-chatbot-message-avatar">AI</div>
                <div class="ai-chatbot-message-content">
                    <div class="ai-chatbot-typing">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;

			this.messages.appendChild(typingDiv);
			this.messages.scrollTop = this.messages.scrollHeight;

			return typingId;
		}

		removeTyping(typingId) {
			const typingDiv = document.getElementById(typingId);
			if (typingDiv) {
				typingDiv.style.opacity = "0";
				setTimeout(() => typingDiv.remove(), 200);
			}
		}

		trackEvent(eventName, data = {}) {
			// Analytics tracking placeholder
			if (window.gtag) {
				window.gtag("event", eventName, {
					event_category: "chatbot",
					...data,
				});
			}
		}
	}

	// Initialize chatbot
	const chatbot = new AIChatbot();

	// Expose API for external control
	window.AIChatbot = {
		open: () => chatbot.open(),
		close: () => chatbot.close(),
		toggle: () => chatbot.toggle(),
		minimize: () => chatbot.minimize(),
		sendMessage: () => chatbot.sendMessage(),
		sendSuggestion: (element) => chatbot.sendSuggestion(element),
	};

	// Auto-open on mobile after delay
	if (window.innerWidth < 768 && config.autoStart) {
		setTimeout(() => {
			if (!isOpen) {
				chatbot.open();
			}
		}, 5000);
	}
})();
