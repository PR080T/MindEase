# 🌟 MindEase - AI Mental Health Chatbot

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [🔧 Technical Details](#-technical-details)
- [🛠️ Installation & Setup](#️-installation--setup)
- [🚀 Usage Guide](#-usage-guide)
- [🎨 UI/UX Design](#-uiux-design)
- [🤖 AI Models](#-ai-models)
- [📊 Dataset](#-dataset)
- [🔒 Security & Privacy](#-security--privacy)
- [🆘 Crisis Resources](#-crisis-resources)
- [🌐 Deployment](#-deployment)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview

**MindEase** is a sophisticated AI-powered mental health chatbot designed to provide compassionate, empathetic support through real-time conversations. Built with Streamlit and powered by multiple large language models, it offers a safe, judgment-free space for mental health discussions.

### 🎪 **Key Highlights:**
- **Multi-AI Architecture**: 5 advanced AI models working together
- **90+ Mental Health Topics**: Comprehensive coverage of mental health scenarios
- **Indian Crisis Support**: Localized emergency resources
- **Privacy-First**: No data storage, secure conversations
- **Professional Design**: Modern, calming UI with therapeutic colors
- **Real-time Support**: Instant responses with personalized advice

## ✨ Features

### 🤖 **AI Capabilities**
- **5 AI Models**: Mistral AI, LLaMA 3.3 Turbo, DeepSeek R1, LLaMA Vision, Cohere Command
- **Dynamic Model Switching**: Change AI models mid-conversation
- **Contextual Understanding**: Maintains conversation context
- **Emotional Intelligence**: Recognizes and responds to emotional cues
- **Personalized Responses**: Uses user names for personalized interaction

### 💬 **Conversation Features**
- **Intent Recognition**: 90+ mental health scenarios covered
- **Crisis Detection**: Automatic crisis resource provision
- **Session Persistence**: Chat history maintained during session
- **Typing Indicators**: Real-time response generation feedback
- **Message Timestamps**: Track conversation flow

### 🎨 **User Interface**
- **Modern Design**: Professional gradient color scheme
- **Responsive Layout**: Works on all devices (desktop, tablet, mobile)
- **Animated Elements**: Smooth transitions and hover effects
- **Dark Theme**: Eye-friendly dark mode design
- **Accessibility**: Screen reader friendly, keyboard navigation

### 🔒 **Security & Privacy**
- **No Data Storage**: Conversations not saved permanently
- **API Key Security**: Secure environment variable handling
- **HTTPS Support**: Encrypted communication
- **Privacy-First**: No personal data collection

## 🏗️ Architecture

### **System Architecture**
```
┌─────────────────────────────────────────────────────────────┐
│                     MindEase Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)                                       │
│  ├── Chat Interface                                         │
│  ├── Model Selection                                        │
│  ├── Settings Panel                                         │
│  └── Crisis Resources                                       │
├─────────────────────────────────────────────────────────────┤
│  Backend (Python)                                           │
│  ├── Session Management                                     │
│  ├── API Integration                                        │
│  ├── Response Processing                                    │
│  └── Dataset Integration                                    │
├─────────────────────────────────────────────────────────────┤
│  AI Models (External APIs)                                  │
│  ├── Together AI (Mistral, LLaMA, DeepSeek)                │
│  └── Cohere (Command Model)                                 │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── Mental Health Dataset (JSON)                          │
│  └── Environment Variables                                  │
└─────────────────────────────────────────────────────────────┘
```

### **Data Flow**
1. **User Input** → Streamlit Interface
2. **Input Processing** → Context Addition from Dataset
3. **AI Request** → Selected Model via API
4. **Response Generation** → AI Model Processing
5. **Response Enhancement** → Personalization & Formatting
6. **Display** → Formatted Response in Chat Interface

## 📁 Project Structure

```
Mental_Health_Chatbot/
├── 📄 MentalHealthChatbot.py          # Main application file
├── 📊 MentalHealthChatbotDataset.json # AI training dataset
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # This comprehensive guide
├── 🚀 DEPLOYMENT_GUIDE.md            # Deployment instructions
├── 🔧 runtime.txt                     # Python version specification
├── 🌐 .env.example                    # Environment variables template
├── 🚫 .gitignore                      # Git ignore patterns
├── 📁 .streamlit/                     # Streamlit configuration
│   ├── config.toml                    # Streamlit settings
│   └── secrets.toml                   # API keys (local development)
└── 📁 .vscode/                        # VS Code settings
    └── settings.json                  # Editor configuration
```

## 🔧 Technical Details

### **Core Technologies**
- **Framework**: Streamlit 1.28.0+
- **Language**: Python 3.8+
- **AI Integration**: LangChain
- **APIs**: Together AI, Cohere
- **UI**: HTML/CSS with Streamlit components
- **Deployment**: Streamlit Cloud

### **Key Python Libraries**
```python
streamlit>=1.28.0          # Web framework
langchain-together>=0.1.0  # Together AI integration
langchain-community>=0.0.0 # LangChain community tools
python-dotenv>=1.0.0       # Environment variable management
```

### **File-by-File Breakdown**

#### 🔹 **MentalHealthChatbot.py** (Main Application)
**Lines 1-22**: **Environment Setup**
- Imports essential libraries
- Loads environment variables
- Configures API keys with fallback mechanisms

**Lines 24-50**: **Dataset Loading Function**
- `load_mental_health_data()`: Loads and processes mental health dataset
- Converts intent-based structure to keyword-advice mapping
- Handles file not found and JSON parsing errors

**Lines 52-63**: **System Prompt Configuration**
- Defines AI behavior and response guidelines
- Ensures empathetic, supportive tone
- Provides clear instructions for crisis situations

**Lines 65-91**: **AI Model Initialization**
- `initialize_models()`: Sets up multiple AI models
- Handles API key validation
- Provides error handling for model failures

**Lines 96-143**: **Response Generation**
- `get_response()`: Main AI interaction function
- Adds contextual information from dataset
- Personalizes responses with user names
- Handles API errors and response formatting

**Lines 145-323**: **UI Styling (CSS)**
- Professional gradient color scheme
- Modern chat bubble design
- Responsive layout for all devices
- Smooth animations and transitions

**Lines 325-327**: **Page Header**
- App title and subtitle
- Professional branding

**Lines 329-343**: **Session State Management**
- Chat history persistence
- User preferences storage
- Input state management

**Lines 345-420**: **Sidebar Configuration**
- Model selection interface
- User settings panel
- Crisis resources display
- Application controls

**Lines 422-520**: **Main Chat Interface**
- Message display system
- User input handling
- Send/Clear button functionality
- Real-time message updates

#### 🔹 **MentalHealthChatbotDataset.json** (AI Training Data)
**Structure**:
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello! How can I help you today?"]
    }
  ]
}
```

**Content Coverage**:
- 90+ mental health scenarios
- Depression, anxiety, stress management
- Crisis situations and self-harm
- Positive affirmations and coping strategies
- Relationship and family issues

#### 🔹 **requirements.txt** (Dependencies)
```
streamlit>=1.28.0
langchain-together>=0.1.0
langchain-community>=0.0.0
python-dotenv>=1.0.0
```

#### 🔹 **.streamlit/config.toml** (Streamlit Configuration)
```toml
[theme]
base = "dark"
primaryColor = "#667eea"
backgroundColor = "#0f0f23"
secondaryBackgroundColor = "#1a1a2e"
textColor = "#ffffff"

[server]
enableXsrfProtection = false
enableCORS = false
```

## 🛠️ Installation & Setup

### **Prerequisites**
- **Python 3.8+** (Recommended: Python 3.9-3.11)
- **Git** (for cloning repository)
- **API Keys** (Together AI and/or Cohere)
- **Text Editor/IDE** (VS Code, PyCharm, etc.)

### **Step 1: Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/mindease-mental-health-chatbot.git
cd mindease-mental-health-chatbot
```

### **Step 2: Create Virtual Environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: API Key Setup**
1. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Get API Keys**:
   - **Together AI**: Visit [together.ai](https://together.ai) → Sign up → Get API key
   - **Cohere**: Visit [cohere.com](https://cohere.com) → Sign up → Get API key

3. **Add keys to .env file**:
   ```
   TOGETHER_API_KEY=your_together_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
   ```

### **Step 5: Run Application**
```bash
streamlit run MentalHealthChatbot.py
```

### **Step 6: Open in Browser**
- **Local URL**: http://localhost:8501
- **Network URL**: http://YOUR_IP:8501

## 🚀 Usage Guide

### **Starting a Conversation**
1. **Launch the app**: Run `streamlit run MentalHealthChatbot.py`
2. **Choose AI model**: Select from the sidebar dropdown
3. **Enter your name** (optional): For personalized responses
4. **Start chatting**: Type your message and click "Send"

### **AI Model Selection**
- **Mistral AI**: Fast, efficient for general conversations
- **LLaMA 3.3 Turbo**: Advanced reasoning, empathetic responses
- **DeepSeek R1**: Deep understanding, thoughtful advice
- **LLaMA Vision**: Multimodal capabilities (future feature)
- **Cohere Command**: Professional-grade conversational AI

### **Feature Usage**
- **Clear Chat**: Reset conversation history
- **Crisis Resources**: Access emergency helplines
- **Model Switching**: Change AI models mid-conversation
- **Personalization**: Add your name for personalized responses

### **Best Practices**
- **Be specific**: Describe your feelings in detail
- **Use context**: Reference previous parts of conversation
- **Try different models**: Each AI has unique strengths
- **Take breaks**: Don't rely solely on AI for mental health

## 🎨 UI/UX Design

### **Color Scheme**
- **Primary**: `#4f46e5` (Professional Indigo)
- **Secondary**: `#7c3aed` (Deep Purple)
- **Accent**: `#667eea` (Soft Blue-Purple)
- **Background**: `#0f0f23` (Dark Navy)
- **Text**: `#ffffff` (White)
- **Warning**: `#ff6b6b` (Soft Red)

### **Typography**
- **Headers**: Inter, sans-serif, 800 weight
- **Body**: Default system fonts
- **UI Elements**: Uppercase, letter-spacing for buttons

### **Layout Design**
- **Responsive**: Works on all screen sizes
- **Chat Bubbles**: Rounded corners, shadows
- **Animations**: Smooth transitions, hover effects
- **Accessibility**: High contrast, keyboard navigation

### **Animation Effects**
- **Slide-in messages**: New messages animate from left
- **Button hover**: Scale and color transitions
- **Pulse effect**: Crisis warning animation
- **Loading states**: Smooth loading indicators

## 🤖 AI Models

### **Together AI Models**

#### **1. Mistral AI (mistralai/Mistral-7B-Instruct-v0.3)**
- **Purpose**: General conversation, quick responses
- **Strengths**: Fast inference, good for basic support
- **Use Cases**: Initial conversations, general advice
- **Response Style**: Concise, practical

#### **2. LLaMA 3.3 Turbo (meta-llama/Llama-3.3-70B-Instruct-Turbo-Free)**
- **Purpose**: Advanced reasoning, empathetic responses
- **Strengths**: Deep understanding, emotional intelligence
- **Use Cases**: Complex mental health discussions
- **Response Style**: Detailed, empathetic

#### **3. DeepSeek R1 (deepseek-ai/deepseek-r1-distill-llama-70b-free)**
- **Purpose**: Deep analysis, thoughtful advice
- **Strengths**: Analytical thinking, problem-solving
- **Use Cases**: Strategic guidance, coping strategies
- **Response Style**: Analytical, solution-focused

#### **4. LLaMA Vision (meta-llama/Llama-Vision-Free)**
- **Purpose**: Multimodal capabilities (future feature)
- **Strengths**: Text and image processing
- **Use Cases**: Visual therapy aids, mood tracking
- **Response Style**: Multimodal responses

### **Cohere Model**

#### **5. Cohere Command (command-xlarge)**
- **Purpose**: Professional-grade conversations
- **Strengths**: Consistent quality, reliable responses
- **Use Cases**: Formal counseling-style interactions
- **Response Style**: Professional, structured

### **Model Configuration**
```python
# Model initialization with error handling
models = {
    "Mistral AI": Together(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        together_api_key=TOGETHER_API_KEY,
        max_tokens=1024
    ),
    # ... other models
}
```

## 📊 Dataset

### **Dataset Structure**
The `MentalHealthChatbotDataset.json` file contains structured mental health conversation data:

```json
{
  "intents": [
    {
      "tag": "depression",
      "patterns": [
        "I feel sad all the time",
        "I'm always down",
        "I feel hopeless"
      ],
      "responses": [
        "I understand you're going through a difficult time...",
        "Your feelings are valid, and you're not alone..."
      ]
    }
  ]
}
```

### **Content Categories**
1. **Emotional Support**: Depression, anxiety, loneliness
2. **Crisis Situations**: Self-harm, suicidal thoughts
3. **Coping Strategies**: Stress management, relaxation
4. **Relationships**: Family, friends, romantic relationships
5. **Life Challenges**: Work stress, academic pressure
6. **Positive Reinforcement**: Encouragement, motivation

### **Dataset Usage**
- **Keyword Matching**: Identifies relevant context
- **Response Enhancement**: Adds specific mental health advice
- **Context Injection**: Enriches AI prompts with relevant information

## 🔒 Security & Privacy

### **Data Protection**
- **No Permanent Storage**: Conversations not saved to disk
- **Session-Based**: Data cleared when session ends
- **API Security**: Keys stored in environment variables
- **HTTPS**: Encrypted communication in production

### **Privacy Features**
- **Anonymous Usage**: No personal data collection
- **Local Processing**: Minimal data sent to external APIs
- **No Logging**: Conversations not logged or monitored
- **User Control**: Users can clear chat history anytime

### **Security Best Practices**
- **API Key Protection**: Never commit keys to version control
- **Environment Variables**: Use `.env` for sensitive data
- **Regular Updates**: Keep dependencies updated
- **Error Handling**: Graceful failure without data exposure

## 🆘 Crisis Resources

### **Indian Mental Health Helplines**
- **iCall Mental Health Helpline**: 9152987821 (24/7)
- **AASRA Suicide Prevention**: +91 98204 66726
- **Vandrevala Foundation**: 1800 233 3330
- **WhatsApp Support**: +91 9999 666 555
- **Emergency Services**: 112

### **Crisis Detection**
The app automatically provides crisis resources when detecting:
- Suicidal ideation keywords
- Self-harm mentions
- Extreme despair expressions
- Crisis-related language patterns

### **Crisis Response Protocol**
1. **Immediate Resources**: Display crisis hotlines
2. **Gentle Messaging**: Encourage professional help
3. **Safety Planning**: Provide immediate coping strategies
4. **Follow-up**: Maintain supportive conversation

## 🌐 Deployment

For comprehensive deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

### **Quick Deploy Options**
1. **Streamlit Cloud**: Free, easy deployment
2. **Heroku**: Scalable cloud platform
3. **Railway**: Modern deployment platform
4. **Render**: Simple web services

### **Environment Variables for Deployment**
```
TOGETHER_API_KEY=your_together_api_key
COHERE_API_KEY=your_cohere_api_key
```

## 🐛 Troubleshooting

### **Common Issues**

#### **1. "No AI models available"**
- **Cause**: Missing or invalid API keys
- **Solution**: Check `.env` file, verify API keys
- **Command**: `python -c "import os; print(os.getenv('TOGETHER_API_KEY'))"`

#### **2. "Dataset file not found"**
- **Cause**: Missing `MentalHealthChatbotDataset.json`
- **Solution**: Ensure file exists in project directory
- **Command**: `ls MentalHealthChatbotDataset.json`

#### **3. "Module not found" errors**
- **Cause**: Missing dependencies
- **Solution**: Install requirements
- **Command**: `pip install -r requirements.txt`

#### **4. "Port already in use"**
- **Cause**: Another Streamlit app running
- **Solution**: Use different port
- **Command**: `streamlit run MentalHealthChatbot.py --server.port 8502`

#### **5. API rate limits**
- **Cause**: Too many requests to AI APIs
- **Solution**: Implement request throttling, upgrade API plan
- **Monitoring**: Check API usage in provider dashboards

### **Debug Mode**
Enable debug mode for detailed error information:
```bash
streamlit run MentalHealthChatbot.py --logger.level=debug
```

### **Log Analysis**
Check Streamlit logs for errors:
```bash
# View recent logs
streamlit run MentalHealthChatbot.py --server.enableXsrfProtection=false
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### **Development Setup**
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes**: Implement your feature
4. **Test thoroughly**: Ensure all functionality works
5. **Submit PR**: Create pull request with description

### **Contribution Guidelines**
- **Code Style**: Follow PEP 8 standards
- **Documentation**: Update README for new features
- **Testing**: Test all changes thoroughly
- **Commit Messages**: Use clear, descriptive messages

### **Areas for Contribution**
- **New AI Models**: Add support for additional models
- **UI Improvements**: Enhance user interface
- **Feature Additions**: Add new mental health features
- **Localization**: Add support for multiple languages
- **Documentation**: Improve guides and tutorials

## 📄 License

This project is licensed under the MIT License. See the full license below:

```
MIT License

Copyright (c) 2024 MindEase

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ⚠️ **Important Disclaimer**

**MindEase is a supportive tool and NOT a replacement for professional mental health care.**

- This chatbot provides general support and information
- It cannot diagnose mental health conditions
- It should not replace therapy or professional counseling
- In case of mental health crisis, contact emergency services immediately
- Always consult qualified mental health professionals for serious issues

### **When to Seek Professional Help**
- Persistent feelings of sadness or hopelessness
- Thoughts of self-harm or suicide
- Inability to function in daily life
- Substance abuse issues
- Trauma or abuse situations

**Emergency Numbers**: 
- **India**: 112 (Emergency Services)
- **USA**: 988 (Suicide & Crisis Lifeline)
- **UK**: 116 123 (Samaritans)

---

## 🎯 **Future Roadmap**

### **Upcoming Features**
- **Voice Integration**: Speech-to-text and text-to-speech
- **Mood Tracking**: Visual mood history and patterns
- **Multilingual Support**: Hindi, Tamil, and other Indian languages
- **Therapist Matching**: Connect with professional therapists
- **Group Support**: Anonymous peer support groups
- **Mobile App**: Native iOS and Android applications

### **Technical Improvements**
- **Performance**: Faster response times
- **AI Enhancement**: Better context understanding
- **Scalability**: Support for more concurrent users
- **Analytics**: Usage insights and improvement metrics

---

**Made with ❤️ for mental health awareness**

*"Every conversation matters. Every person deserves support. Together, we can make mental health care accessible to everyone."*

---

**Support**: For technical issues, create an issue on GitHub  
**Contact**: For general inquiries, reach out via GitHub discussions  
**Community**: Join our community for updates and discussions

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Active Development
