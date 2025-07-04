import streamlit as st
import json
import time
import os
from langchain_together import Together
from langchain_community.llms import Cohere
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API Keys from environment variables or Streamlit secrets
try:
    # Try Streamlit secrets first (for cloud deployment)
    TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY", os.getenv("TOGETHER_API_KEY"))
    COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", os.getenv("COHERE_API_KEY"))
except:
    # Fallback to environment variables (for local development)
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Load dataset function
@st.cache_data
def load_mental_health_data():
    try:
        # Try to load from current directory first
        dataset_path = os.path.join(os.path.dirname(__file__), "MentalHealthChatbotDataset.json")
        with open(dataset_path, "r", encoding='utf-8') as file:
            data = json.load(file)
            
        # Convert the intent-based structure to a simple keyword-advice mapping
        keyword_advice = {}
        for intent in data.get("intents", []):
            tag = intent.get("tag", "")
            patterns = intent.get("patterns", [])
            responses = intent.get("responses", [])
            
            # Create keyword mappings
            for pattern in patterns:
                if responses:
                    keyword_advice[pattern.lower()] = responses[0]  # Use first response
                    
        return keyword_advice
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please ensure MentalHealthChatbotDataset.json is in the same directory.")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return {}

# System prompt for AI behavior
SYSTEM_PROMPT = """
You are a kind, compassionate, and supportive mental health assistant.  
Your goal is to **uplift, encourage, and provide clear, practical advice** to users in distress.

**How to Respond:**
- **Start every response with a strong, reassuring sentence in CAPITALS and bold.**  
- Focus on **empowering solutions** rather than just acknowledging distress.  
- Use a **warm, hopeful tone**, reminding them that **things can improve and they are capable**.  
- Offer **small, achievable steps** for self-care, deep breathing, and positive self-talk.  
- If a user feels **overwhelmed, remind them of their inner strength**.  
"""

# Initialize AI models
def initialize_models():
    models = {}
    
    # Initialize Together AI models if API key is available
    if TOGETHER_API_KEY:
        try:
            models.update({
                "Mistral AI": Together(model="mistralai/Mistral-7B-Instruct-v0.3", together_api_key=TOGETHER_API_KEY),
                "LLaMA 3.3 Turbo": Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", together_api_key=TOGETHER_API_KEY),
                "DeepSeek R1": Together(model="deepseek-ai/deepseek-r1-distill-llama-70b-free", together_api_key=TOGETHER_API_KEY),
                "LLaMA Vision": Together(model="meta-llama/Llama-Vision-Free", together_api_key=TOGETHER_API_KEY),
            })
        except Exception as e:
            st.warning(f"⚠️ Could not initialize Together AI models: {str(e)}")
    
    # Initialize Cohere model if API key is available
    if COHERE_API_KEY:
        try:
            models["Cohere Command"] = Cohere(model="command-xlarge", cohere_api_key=COHERE_API_KEY)
        except Exception as e:
            st.warning(f"⚠️ Could not initialize Cohere model: {str(e)}")
    
    if not models:
        st.error("❌ No AI models available. Please check your API keys in the .env file.")
        
    return models

# Initialize models
models = initialize_models()

# Function to get AI response
def get_response(model_name, user_query, dataset, user_name=""):
    if not models or model_name not in models:
        return "❌ Selected AI model is not available. Please check your API keys."
    
    # Add context from dataset if relevant keywords are found
    context_added = False
    for keyword, advice in dataset.items():
        if keyword in user_query.lower():
            user_query += f"\n[Additional Context: {advice}]"
            context_added = True
            break  # Only add one context to avoid overwhelming the prompt

    # Personalize the prompt if user name is provided
    user_prefix = f"{user_name}: " if user_name else "User: "
    modified_query = SYSTEM_PROMPT + f"\n{user_prefix}" + user_query + "\nAI:"

    try:
        response = models[model_name].invoke(modified_query, max_tokens=1024)
        response = response.strip()

        # Clean up response if it contains unwanted prefixes
        if response.startswith("AI:"):
            response = response[3:].strip()

        # Personalize response with user's name if provided
        if user_name and not any(greeting in response.lower() for greeting in [user_name.lower(), "hello", "hi"]):
            # Add name naturally to the response
            if response.startswith("**"):
                # If response starts with bold text, add name after it
                parts = response.split("**", 2)
                if len(parts) >= 3:
                    response = f"**{parts[1]}** {user_name}, {parts[2]}"
            else:
                response = f"{user_name}, {response}"

        # Ensure response completion (simplified logic)
        if len(response) < 50 and not response.endswith((".", "!", "?")):
            try:
                additional_response = models[model_name].invoke(modified_query + " " + response, max_tokens=512)
                response += " " + additional_response.strip()
            except:
                pass  # If retry fails, use original response

        return response if isinstance(response, str) else str(response)

    except Exception as e:
        return f"⚠️ I'm having trouble connecting right now. Error: {str(e)}"

# Streamlit UI Configuration
st.set_page_config(page_title="Mental Health Chatbot", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
        /* Main App Styling */
        .stApp {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff;
        }
        
        /* Chat Container */
        .chat-container {
            max-width: 700px; 
            margin: auto; 
            background: rgba(255, 255, 255, 0.05);
            padding: 25px; 
            border-radius: 20px; 
            box-shadow: 0px 8px 32px rgba(0, 255, 200, 0.15);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Chat Messages */
        .chat-message { 
            padding: 15px; 
            border-radius: 12px; 
            margin: 15px 0; 
            font-size: 16px;
            animation: slideIn 0.6s ease-out forwards; 
            display: block; 
            max-width: 85%;
            word-wrap: break-word;
        }
        
        /* User Messages */
        .user-message { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            border-radius: 20px 20px 5px 20px; 
            padding: 12px 18px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* AI Messages */
        .ai-message { 
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
            margin-right: auto;
            border-radius: 20px 20px 20px 5px; 
            padding: 12px 18px;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        }
        
        /* Enhanced Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }
        
        /* Send Button Special Styling */
        div[data-testid="column"]:nth-child(1) .stButton > button {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
            animation: pulse 2s infinite !important;
            font-weight: 700 !important;
        }
        
        div[data-testid="column"]:nth-child(1) .stButton > button:hover {
            background: linear-gradient(135deg, #ee5a24 0%, #ff6b6b 100%) !important;
            animation: none !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6) !important;
        }
        
        /* Clear Button Styling */
        div[data-testid="column"]:nth-child(2) .stButton > button {
            background: linear-gradient(135deg, #6b7280 0%, #9ca3af 100%) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Input Field Styling */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 15px !important;
            color: white !important;
            padding: 12px 16px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.5) !important;
            background: rgba(255, 255, 255, 0.15) !important;
        }
        
        /* Selectbox Styling */
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        /* Sidebar Styling */
        .stSidebar {
            background: linear-gradient(180deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Crisis Warning */
        .crisis-warning { 
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            padding: 15px; 
            border-radius: 12px; 
            color: white; 
            font-weight: bold;
            animation: pulse 3s infinite;
            border: 2px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 20px rgba(255, 107, 107, 0.3);
        }
        
        /* Title Styling */
        .main-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem !important;
            font-weight: 800 !important;
            text-align: center !important;
            margin-bottom: 10px !important;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #b0b0b0;
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        

        
        /* Animations */
        @keyframes slideIn { 
            from { opacity: 0; transform: translateX(-20px); } 
            to { opacity: 1; transform: translateX(0); } 
        }
        
        @keyframes pulse { 
            0%, 100% { opacity: 1; transform: scale(1); } 
            50% { opacity: 0.8; transform: scale(1.02); } 
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
            50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>💬 MindEase</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>🌿 Your AI companion for mental well-being and emotional support</p>", unsafe_allow_html=True)

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [("ai-message", "<strong>🤖 MindEase:</strong> **YOU ARE STRONGER THAN YOU THINK!** Hello! I am here to support you on your mental health journey. How are you feeling right now?",datetime.datetime.now().strftime("%H:%M:%S"))]

# Session state for user preferences
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# Session state for input management
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# Enhanced Sidebar
with st.sidebar:
    # Header with status
    st.markdown("### 🛠️ Settings & Controls")
    
    # Status indicator
    if models:
        st.success(f"✅ {len(models)} AI models available")
    else:
        st.error("❌ No AI models available")
    
    # Model Selection
    if models:
        model_choice = st.selectbox(
            "🤖 Select AI Model:", 
            list(models.keys()),
            help="Choose the AI model that best fits your conversation style"
        )
        
        # Model info
        model_descriptions = {
            "Mistral AI": "⚡ Fast and efficient for general conversations",
            "LLaMA 3.3 Turbo": "🧠 Advanced reasoning and empathetic responses",
            "DeepSeek R1": "💭 Deep understanding and thoughtful advice",
            "LLaMA Vision": "👁️ Multimodal capabilities (text and vision)",
            "Cohere Command": "💼 Professional-grade conversational AI"
        }
        
        if model_choice in model_descriptions:
            st.info(model_descriptions[model_choice])
    else:
        st.error("❌ No AI models available. Please configure your API keys in a .env file.")
        st.info("💡 Create a .env file with your API keys. See .env.example for reference.")
        st.stop()
    
    st.markdown("---")
    
    # User personalization
    st.markdown("### 👤 Personalization")
    user_name = st.text_input(
        "Your Name (Optional):", 
        value=st.session_state.user_name,
        help="Add your name for more personalized responses"
    )
    if user_name != st.session_state.user_name:
        st.session_state.user_name = user_name
        if user_name:
            st.success(f"👋 Hello, {user_name}! Responses will now be personalized for you.")
    

    
    st.markdown("---")
    st.markdown("### 🌟 Mental Health Tips")
    
    # Rotating tips
    tips = [
        "🫁 **Breathe deeply** - Try the 4-7-8 technique",
        "🙏 **Practice gratitude** - List 3 things you're thankful for",
        "💧 **Stay hydrated** - Drink water regularly",
        "😴 **Get quality sleep** - Aim for 7-9 hours nightly",
        "🤝 **Connect with others** - Reach out to friends and family",
        "💝 **Be self-compassionate** - Treat yourself with kindness",
        "🚶 **Move your body** - Even a short walk helps",
        "🧘 **Practice mindfulness** - Stay present in the moment"
    ]
    
    import random
    daily_tip = tips[hash(datetime.datetime.now().strftime("%Y-%m-%d")) % len(tips)]
    st.info(f"**Today's Tip:** {daily_tip}")
    
    st.markdown("---")
    st.markdown("### 🆘 Crisis Resources")
    st.markdown("""
    <div class="crisis-warning">
    <strong>🚨 If you're in crisis, please contact:</strong><br>
    📞 <strong>iCall Mental Health Helpline</strong>: 9152987821 (24/7)<br>
    📞 <strong>AASRA Suicide Prevention</strong>: +91 98204 66726<br>
    📞 <strong>Vandrevala Foundation</strong>: 1800 233 3330<br>
    📱 <strong>WhatsApp Support</strong>: +91 9999 666 555<br>
    🚑 <strong>Emergency Services</strong>: 112
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About MindEase")
    st.markdown("""
    🌟 **MindEase** is your AI-powered mental health companion, designed to provide:
    
    ✨ **Empathetic conversations**  
    🎯 **Personalized support**  
    🛡️ **Safe, judgment-free space**  
    📚 **Evidence-based guidance**  
    
    **⚠️ Important**: This is a supportive tool, not a replacement for professional mental health care.
    """)
    
    # Version info
    st.markdown("---")
    st.caption("🔧 MindEase v2.1 | Enhanced Dataset & Experience")

# Chatbox UI
for role, text, timestamp in st.session_state.messages:
    st.markdown(f'<div class="chat-message {role}">{text} <br><small style="color:gray;">🕒 {timestamp}</small></div>', unsafe_allow_html=True)

# Enhanced Input Section
# User Input Field with session state management
user_input = st.text_input(
    "💬 Share your thoughts and feelings...", 
    value=st.session_state.user_input,
    key=f"user_input_{st.session_state.input_key}",
    placeholder="Type your message here... Press Enter or click Send",
    help="Express yourself freely - I'm here to listen and support you"
)

# Buttons Layout
col1, col2 = st.columns([4, 1])
with col1:
    send_btn = st.button("🚀 Send", key="send-btn", help="Send your message", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", key="clear-btn", help="Start fresh conversation", use_container_width=True)

# Function to clear input
def clear_input():
    st.session_state.user_input = ""
    st.session_state.last_input = ""
    st.session_state.input_key += 1

# Function to send message
def send_message(message_text):
    if message_text.strip():
        # Add user message immediately
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append(("user-message", f"<strong>You:</strong> {message_text}", timestamp))
        
        # Clear input immediately for better UX
        clear_input()
        
        # Show processing message
        with st.spinner(f"🤖 MindEase ({model_choice}) is crafting a thoughtful response..."):
            # Add some realistic thinking time
            time.sleep(1.8)
            
            # Load dataset and get response
            dataset = load_mental_health_data()
            response = get_response(model_choice, message_text, dataset, st.session_state.user_name)
        
        # Add AI response with timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append(("ai-message", f"<strong>🤖 MindEase ({model_choice}):</strong> {response}", timestamp))
        
        # Show success feedback
        st.success("✨ Response generated! Continue the conversation below.")
        
        # Rerun to update the chat
        st.rerun()
    else:
        st.error("⚠️ Please type a message before sending. I'm here to listen!")

# Handle Clear Chat
if clear_btn:
    st.session_state.messages = [("ai-message", "<strong>🤖 MindEase:</strong> **YOU ARE STRONGER THAN YOU THINK!** Hello! I am here to support you on your mental health journey. How are you feeling right now?",datetime.datetime.now().strftime("%H:%M:%S"))]
    clear_input()
    st.success("💫 Chat cleared! Ready for a fresh start.")
    st.rerun()

# Handle Message Send (Button click or Enter key)
if send_btn or (user_input and user_input != st.session_state.last_input and user_input.strip()):
    if user_input.strip():
        st.session_state.last_input = user_input
        send_message(user_input)
    else:
        st.session_state.user_input = user_input
