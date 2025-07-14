# 🌟 MindEase - AI Mental Health Chatbot

A compassionate AI-powered mental health support chatbot built with Streamlit and multiple language models.

## ✨ Features

- **Multiple AI Models**: Mistral AI, LLaMA 3.3 Turbo, DeepSeek R1, LLaMA Vision, Cohere Command, and OpenAI GPT models
- **90+ Mental Health Topics**: Comprehensive coverage of mental health scenarios
- **Real-time Support**: Instant responses with personalized advice
- **Privacy-First**: No data storage, secure conversations
- **Crisis Resources**: Emergency helplines and support information
- **Modern UI**: Professional dark theme with smooth animations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys from [Together AI](https://together.ai) and/or [Cohere](https://cohere.com) and/or [OpenAI](https://openai.com)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Mental_Health_Chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # TOGETHER_API_KEY=your_together_api_key
   # COHERE_API_KEY=your_cohere_api_key
   # OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the application**
   ```bash
   streamlit run MentalHealthChatbot.py
   ```

5. **Open in browser**: http://localhost:8501

## 📁 Project Structure

```
Mental_Health_Chatbot/
├── MentalHealthChatbot.py          # Main application
├── MentalHealthChatbotDataset.json # AI training dataset
├── requirements.txt                # Dependencies
├── runtime.txt                     # Python version
├── Procfile                        # Heroku deployment config
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── .streamlit/                     # Streamlit configuration
├── .vscode/                        # VS Code settings
└── README.md                       # This file
```

## 🤖 AI Models

- **Mistral AI**: Fast, efficient responses
- **LLaMA 3.3 Turbo**: Advanced reasoning and empathy
- **DeepSeek R1**: Deep analysis and problem-solving
- **LLaMA Vision**: Multimodal capabilities
- **Cohere Command**: Professional-grade conversations
- **OpenAI GPT**: Industry-leading language models

## 🔒 Privacy & Security

- No conversation data is stored permanently
- API keys are securely managed through environment variables
- All communications are encrypted (HTTPS)
- No personal data collection

## 🆘 Crisis Support

The chatbot includes crisis detection and provides emergency resources including:
- National suicide prevention hotlines
- Mental health crisis centers
- Emergency services information

## 📄 License

This project is open source. Please use responsibly and ensure proper mental health resources are available to users.

---

**⚠️ Important**: This chatbot is for support purposes only and is not a replacement for professional mental health care. If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately.
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
