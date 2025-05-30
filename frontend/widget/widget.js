(function() {
    'use strict';
    
    // Configuration
    const config = {
        apiUrl: window.AI_CHATBOT_API_URL || 'http://localhost:8000',
        position: window.AI_CHATBOT_POSITION || 'bottom-right',
        theme: window.AI_CHATBOT_THEME || 'modern',
        autoStart: window.AI_CHATBOT_AUTO_START !== false
    };
    
    // Widget state
    let sessionId = localStorage.getItem('ai_chatbot_session') || null;
    let isOpen = false;
    let isMinimized = false;
    let crawlStatus = 'pending';
    
    // Create widget HTML
    const widgetHTML = `
        <div id="ai-chatbot-container" class="ai-chatbot-container ai-chatbot-${config.position}">
            <div id="ai-chatbot-widget" class="ai-chatbot-widget ai-chatbot-hidden">
                <div class="ai-chatbot-header">
                    <div class="ai-chatbot-header-content">
                        <div class="ai-chatbot-status">
                            <span class="ai-chatbot-status-dot"></span>
                            <span class="ai-chatbot-status-text">AI Assistant</span>
                        </div>
                        <div class="ai-chatbot-header-actions">
                            <button class="ai-chatbot-minimize" onclick="AIChatbot.minimize()">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M4 8H12" stroke="currentColor" stroke-width="2"/>
                                </svg>
                            </button>
                            <button class="ai-chatbot-close" onclick="AIChatbot.close()">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                                    <path d="M4 4L12 12M4 12L12 4" stroke="currentColor" stroke-width="2"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="ai-chatbot-messages" id="ai-chatbot-messages">
                    <div class="ai-chatbot-welcome">
                        <h3>Welcome! 👋</h3>
                        <p>I'm learning about this website to help answer your questions.</p>
                        <div class="ai-chatbot-loading" id="ai-chatbot-loading">
                            <div class="ai-chatbot-spinner"></div>
                            <span>Analyzing website content...</span>
                        </div>
                    </div>
                </div>
                
                <div class="ai-chatbot-input-container">
                    <div class="ai-chatbot-input-wrapper">
                        <input 
                            type="text" 
                            id="ai-chatbot-input" 
                            class="ai-chatbot-input"
                            placeholder="Ask me anything about this website..."
                            disabled
                        />
                        <button 
                            id="ai-chatbot-send" 
                            class="ai-chatbot-send"
                            onclick="AIChatbot.sendMessage()"
                            disabled
                        >
                            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                                <path d="M2 10L18 2L10 18L8 11L2 10Z" stroke="currentColor" stroke-width="2"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
            
            <button id="ai-chatbot-trigger" class="ai-chatbot-trigger" onclick="AIChatbot.toggle()">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="ai-chatbot-icon-chat">
                    <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C13.19 22 14.34 21.78 15.41 21.37L21 23L19.37 17.41C20.78 15.34 22 13.19 22 12C22 6.48 17.52 2 12 2ZM8 13C7.45 13 7 12.55 7 12C7 11.45 7.45 11 8 11C8.55 11 9 11.45 9 12C9 12.55 8.55 13 8 13ZM12 13C11.45 13 11 12.55 11 12C11 11.45 11.45 11 12 11C12.55 11 13 11.45 13 12C13 12.55 12.55 13 12 13ZM16 13C15.45 13 15 12.55 15 12C15 11.45 15.45 11 16 11C16.55 11 17 11.45 17 12C17 12.55 16.55 13 16 13Z" fill="currentColor"/>
                </svg>
                <span class="ai-chatbot-badge" id="ai-chatbot-badge" style="display: none;">1</span>
            </button>
        </div>
    `;
    
    // Create styles
    const styles = `
        <style>
            .ai-chatbot-container {
                position: fixed;
                z-index: 999999;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .ai-chatbot-bottom-right {
                bottom: 20px;
                right: 20px;
            }
            
            .ai-chatbot-widget {
                width: 380px;
                height: 600px;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 5px 40px rgba(0,0,0,0.16);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                transition: all 0.3s ease;
                transform-origin: bottom right;
            }
            
            .ai-chatbot-widget.ai-chatbot-hidden {
                opacity: 0;
                transform: scale(0.9) translateY(20px);
                pointer-events: none;
            }
            
            .ai-chatbot-widget.ai-chatbot-minimized {
                height: 60px;
            }
            
            .ai-chatbot-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px 20px;
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
                gap: 8px;
            }
            
            .ai-chatbot-status-dot {
                width: 8px;
                height: 8px;
                background: #4ade80;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(1.1); }
                100% { opacity: 1; transform: scale(1); }
            }
            
            .ai-chatbot-header-actions {
                display: flex;
                gap: 8px;
            }
            
            .ai-chatbot-header-actions button {
                background: rgba(255,255,255,0.2);
                border: none;
                width: 32px;
                height: 32px;
                border-radius: 8px;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .ai-chatbot-header-actions button:hover {
                background: rgba(255,255,255,0.3);
            }
            
            .ai-chatbot-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f9fafb;
            }
            
            .ai-chatbot-welcome {
                text-align: center;
                padding: 40px 20px;
            }
            
            .ai-chatbot-welcome h3 {
                margin: 0 0 8px 0;
                font-size: 20px;
                color: #111827;
            }
            
            .ai-chatbot-welcome p {
                margin: 0 0 24px 0;
                color: #6b7280;
                font-size: 14px;
            }
            
            .ai-chatbot-loading {
                display: inline-flex;
                align-items: center;
                gap: 12px;
                padding: 12px 20px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
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
                margin-bottom: 16px;
                display: flex;
                gap: 12px;
                animation: fadeIn 0.3s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .ai-chatbot-message-user {
                flex-direction: row-reverse;
            }
            
            .ai-chatbot-message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 8px;
                background: #667eea;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: 600;
                flex-shrink: 0;
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-avatar {
                background: #10b981;
            }
            
            .ai-chatbot-message-content {
                max-width: 70%;
                padding: 12px 16px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            
            .ai-chatbot-message-user .ai-chatbot-message-content {
                background: #667eea;
                color: white;
            }
            
            .ai-chatbot-message-text {
                margin: 0;
                font-size: 14px;
                line-height: 1.5;
            }
            
            .ai-chatbot-message-sources {
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #e5e7eb;
            }
            
            .ai-chatbot-message-sources-title {
                font-size: 12px;
                color: #6b7280;
                margin-bottom: 4px;
            }
            
            .ai-chatbot-message-source {
                font-size: 12px;
                color: #667eea;
                text-decoration: none;
                display: block;
                margin-bottom: 2px;
            }
            
            .ai-chatbot-message-source:hover {
                text-decoration: underline;
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
            }
            
            .ai-chatbot-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                font-size: 14px;
                outline: none;
                transition: all 0.2s;
            }
            
            .ai-chatbot-input:focus {
                border-color: #667eea;
            }
            
            .ai-chatbot-input:disabled {
                background: #f3f4f6;
                cursor: not-allowed;
            }
            
            .ai-chatbot-send {
                width: 44px;
                height: 44px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .ai-chatbot-send:hover:not(:disabled) {
                background: #5a67d8;
                transform: scale(1.05);
            }
            
            .ai-chatbot-send:disabled {
                background: #9ca3af;
                cursor: not-allowed;
            }
            
            .ai-chatbot-trigger {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 50%;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 24px rgba(0,0,0,0.2);
                transition: all 0.3s;
                position: relative;
            }
            
            .ai-chatbot-trigger:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 32px rgba(0,0,0,0.3);
            }
            
            .ai-chatbot-badge {
                position: absolute;
                top: -4px;
                right: -4px;
                background: #ef4444;
                color: white;
                font-size: 12px;
                font-weight: 600;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: badgePop 0.3s ease;
            }
            
            @keyframes badgePop {
                0% { transform: scale(0); }
                50% { transform: scale(1.2); }
                100% { transform: scale(1); }
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
            }
        </style>
    `;
    
    // Inject HTML and styles
    document.head.insertAdjacentHTML('beforeend', styles);
    document.body.insertAdjacentHTML('beforeend', widgetHTML);
    
    // API class
    class ChatbotAPI {
        constructor(apiUrl) {
            this.apiUrl = apiUrl;
        }
        
        async startCrawl(domain) {
            const response = await fetch(`${this.apiUrl}/api/crawl`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ domain, max_pages: 100 })
            });
            
            if (!response.ok) {
                throw new Error('Failed to start crawl');
            }
            
            return response.json();
        }
        
        async getCrawlStatus(jobId) {
            const response = await fetch(`${this.apiUrl}/api/crawl/${jobId}`);
            
            if (!response.ok) {
                throw new Error('Failed to get crawl status');
            }
            
            return response.json();
        }
        
        async sendMessage(question, sessionId) {
            const response = await fetch(`${this.apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question,
                    session_id: sessionId,
                    require_reasoning: true
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to send message');
            }
            
            return response.json();
        }
    }
    
    // Initialize API
    const api = new ChatbotAPI(config.apiUrl);
    
    // Main chatbot class
    class AIChatbot {
        constructor() {
            this.widget = document.getElementById('ai-chatbot-widget');
            this.trigger = document.getElementById('ai-chatbot-trigger');
            this.messages = document.getElementById('ai-chatbot-messages');
            this.input = document.getElementById('ai-chatbot-input');
            this.sendButton = document.getElementById('ai-chatbot-send');
            this.badge = document.getElementById('ai-chatbot-badge');
            this.crawlJobId = null;
            this.ready = false;
            
            this.init();
        }
        
        async init() {
            // Add event listeners
            this.input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            
            // Auto-start crawling if configured
            if (config.autoStart) {
                await this.startCrawling();
            }
        }
        
        async startCrawling() {
            try {
                const domain = window.location.hostname;
                const result = await api.startCrawl(domain);
                
                this.crawlJobId = result.job_id;
                
                if (result.status === 'already_crawling') {
                    // Already ready
                    this.onCrawlComplete();
                } else {
                    // Monitor progress
                    this.monitorCrawlProgress();
                }
            } catch (error) {
                console.error('Failed to start crawling:', error);
                this.showError('Failed to initialize. Please refresh the page.');
            }
        }
        
        async monitorCrawlProgress() {
            const checkProgress = async () => {
                try {
                    const status = await api.getCrawlStatus(this.crawlJobId);
                    
                    if (status.status === 'completed') {
                        this.onCrawlComplete();
                    } else if (status.status === 'failed') {
                        this.showError('Failed to analyze website. Please try again.');
                    } else {
                        // Update progress
                        const progress = status.progress || 0;
                        this.updateProgress(progress);
                        
                        // Check again
                        setTimeout(checkProgress, 2000);
                    }
                } catch (error) {
                    console.error('Failed to check progress:', error);
                    setTimeout(checkProgress, 5000);
                }
            };
            
            checkProgress();
        }
        
        updateProgress(progress) {
            const loading = document.getElementById('ai-chatbot-loading');
            if (loading) {
                loading.querySelector('span').textContent = 
                    `Analyzing website content... ${progress}%`;
            }
        }
        
        onCrawlComplete() {
            this.ready = true;
            this.input.disabled = false;
            this.sendButton.disabled = false;
            
            // Clear welcome message
            this.messages.innerHTML = '';
            
            // Show ready message
            this.addMessage('bot', 'Hello! I\'ve analyzed this website and I\'m ready to help. What would you like to know?');
            
            // Show notification badge
            if (!isOpen) {
                this.showBadge();
            }
        }
        
        showError(message) {
            const loading = document.getElementById('ai-chatbot-loading');
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
            this.widget.classList.remove('ai-chatbot-hidden');
            this.trigger.style.display = 'none';
            this.hideBadge();
            
            // Focus input if ready
            if (this.ready) {
                this.input.focus();
            }
            
            // Start crawling if not started
            if (!this.crawlJobId && config.autoStart) {
                this.startCrawling();
            }
        }
        
        close() {
            isOpen = false;
            this.widget.classList.add('ai-chatbot-hidden');
            this.trigger.style.display = 'flex';
        }
        
        minimize() {
            isMinimized = !isMinimized;
            this.widget.classList.toggle('ai-chatbot-minimized');
        }
        
        showBadge() {
            this.badge.style.display = 'flex';
        }
        
        hideBadge() {
            this.badge.style.display = 'none';
        }
        
        async sendMessage() {
            const message = this.input.value.trim();
            
            if (!message || !this.ready) return;
            
            // Clear input
            this.input.value = '';
            
            // Add user message
            this.addMessage('user', message);
            
            // Show typing indicator
            const typingId = this.showTyping();
            
            try {
                // Send to API
                const response = await api.sendMessage(message, sessionId);
                
                // Store session ID
                if (response.session_id) {
                    sessionId = response.session_id;
                    localStorage.setItem('ai_chatbot_session', sessionId);
                }
                
                // Remove typing indicator
                this.removeTyping(typingId);
                
                // Add bot response
                this.addMessage('bot', response.answer, response.sources);
                
            } catch (error) {
                console.error('Failed to send message:', error);
                this.removeTyping(typingId);
                this.addMessage('bot', 'Sorry, I encountered an error. Please try again.');
            }
        }
        
        addMessage(sender, text, sources = []) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `ai-chatbot-message ai-chatbot-message-${sender}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'ai-chatbot-message-avatar';
            avatar.textContent = sender === 'user' ? 'U' : 'AI';
            
            const content = document.createElement('div');
            content.className = 'ai-chatbot-message-content';
            
            const textP = document.createElement('p');
            textP.className = 'ai-chatbot-message-text';
            textP.textContent = text;
            content.appendChild(textP);
            
            // Add sources if available
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'ai-chatbot-message-sources';
                
                const sourcesTitle = document.createElement('div');
                sourcesTitle.className = 'ai-chatbot-message-sources-title';
                sourcesTitle.textContent = 'Sources:';
                sourcesDiv.appendChild(sourcesTitle);
                
                sources.forEach(source => {
                    const sourceLink = document.createElement('a');
                    sourceLink.className = 'ai-chatbot-message-source';
                    sourceLink.href = source.url;
                    sourceLink.textContent = source.title || source.url;
                    sourceLink.target = '_blank';
                    sourcesDiv.appendChild(sourceLink);
                });
                
                content.appendChild(sourcesDiv);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            
            this.messages.appendChild(messageDiv);
            
            // Scroll to bottom
            this.messages.scrollTop = this.messages.scrollHeight;
        }
        
        showTyping() {
            const typingId = `typing-${Date.now()}`;
            const typingDiv = document.createElement('div');
            typingDiv.id = typingId;
            typingDiv.className = 'ai-chatbot-message ai-chatbot-message-bot';
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
                typingDiv.remove();
            }
        }
    }
    
    // Initialize chatbot
    const chatbot = new AIChatbot();
    
    // Expose API for external use
    window.AIChatbot = {
        open: () => chatbot.open(),
        close: () => chatbot.close(),
        toggle: () => chatbot.toggle(),
        minimize: () => chatbot.minimize(),
        sendMessage: () => chatbot.sendMessage(),
        startCrawling: () => chatbot.startCrawling()
    };
    
    // Add typing indicator styles
    const typingStyles = `
        <style>
            .ai-chatbot-typing {
                display: inline-flex;
                align-items: center;
                gap: 4px;
            }
            
            .ai-chatbot-typing span {
                width: 8px;
                height: 8px;
                background: #9ca3af;
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
                    transform: translateY(0);
                    opacity: 0.7;
                }
                30% {
                    transform: translateY(-10px);
                    opacity: 1;
                }
            }
        </style>
    `;
    
    document.head.insertAdjacentHTML('beforeend', typingStyles);
})();