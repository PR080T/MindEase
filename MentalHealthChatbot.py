"""
Mental Health Chatbot - A compassionate AI-powered mental health support system.

This module provides a Streamlit-based chatbot interface for mental health support,
utilizing various AI models to provide empathetic and helpful responses.
"""

import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Union, Any

import html
import streamlit as st
from dotenv import load_dotenv
from langchain_together import Together

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the newer Cohere package
COHERE_AVAILABLE = False
COHERE_CLASS = None
COHERE_CLASS_NAME = None

try:
    from langchain_cohere import ChatCohere
    COHERE_AVAILABLE = True
    COHERE_CLASS = ChatCohere
    COHERE_CLASS_NAME = "ChatCohere"
    logger.info("Successfully imported ChatCohere from langchain_cohere")
except ImportError as e:
    logger.info(f"ChatCohere import failed: {e}")
    try:
        from langchain_cohere import Cohere
        COHERE_AVAILABLE = True
        COHERE_CLASS = Cohere
        COHERE_CLASS_NAME = "Cohere"
        logger.info("Successfully imported Cohere from langchain_cohere")
    except ImportError as e:
        logger.info(f"Cohere from langchain_cohere import failed: {e}")
        try:
            from langchain_community.llms import Cohere
            COHERE_AVAILABLE = True
            COHERE_CLASS = Cohere
            COHERE_CLASS_NAME = "Cohere"
            logger.warning("Using deprecated Cohere from langchain_community. Consider upgrading to langchain-cohere package.")
        except ImportError as e:
            logger.info(f"Cohere from langchain_community import failed: {e}")
            COHERE_AVAILABLE = False
            COHERE_CLASS = None
            COHERE_CLASS_NAME = None
            logger.error("Cohere package not available. Install langchain-cohere or langchain-community.")
except Exception as e:
    logger.error(f"Unexpected error importing Cohere: {e}")
    COHERE_AVAILABLE = False
    COHERE_CLASS = None
    COHERE_CLASS_NAME = None

# Import OpenAI package
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI package loaded successfully")
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
        OPENAI_AVAILABLE = True
        logger.warning("Using deprecated ChatOpenAI from langchain_community. Consider upgrading to langchain-openai package.")
    except ImportError:
        OPENAI_AVAILABLE = False
        logger.error("OpenAI package not available. Install langchain-openai or langchain-community.")

# Constants
MAX_MESSAGE_LENGTH = 2000
MAX_TOKENS = 512
DATASET_FILENAME = "MentalHealthChatbotDataset.json"
DEFAULT_RESPONSE_DELAY = 1.8
MIN_RESPONSE_LENGTH = 10
APP_VERSION = "v2.1"

# Default responses
DEFAULT_FALLBACK_RESPONSE = ("**I'M HERE TO SUPPORT YOU.** 💙 Whatever you're feeling right now is completely valid. "
                             "You're not alone in this journey, and I'm here to listen and provide support tailored to your needs. "
                             "Your feelings matter, and it's okay to not be okay sometimes. Mental health is just as important as physical health, and you deserve compassionate care. "
                             "Please share what's on your mind, and we can work through it together at your own pace.")
DEFAULT_WELCOME_MESSAGE = ("**WELCOME TO A SAFE SPACE FOR YOUR MIND!** "
                           "I'm MindEase, your supportive companion on this mental health journey. Whether you're feeling up or down today, "
                           "I'm here to listen without judgment and offer personalized support. How are you feeling right now?")

# Load environment variables
load_dotenv()

# Helper function to check if we're running in Streamlit context
def is_streamlit_context() -> bool:
    """Check if we're running in a Streamlit context."""
    try:
        return hasattr(st, 'session_state') and st.session_state is not None
    except:
        return False

# Set API Keys from environment variables or Streamlit secrets
def get_api_keys() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get API keys from Streamlit secrets or environment variables.

    Returns:
        tuple: (TOGETHER_API_KEY, COHERE_API_KEY, OPENAI_API_KEY)
    """
    together_key = None
    cohere_key = None
    openai_key = None

    try:
        # Try Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets') and st.secrets:
            together_key = st.secrets.get("TOGETHER_API_KEY")
            cohere_key = st.secrets.get("COHERE_API_KEY")
            openai_key = st.secrets.get("OPENAI_API_KEY")
    except (AttributeError, KeyError, FileNotFoundError):
        pass

    # Fallback to environment variables (for local development)
    if not together_key:
        together_key = os.getenv("TOGETHER_API_KEY")
    if not cohere_key:
        cohere_key = os.getenv("COHERE_API_KEY")
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")

    return together_key, cohere_key, openai_key

TOGETHER_API_KEY, COHERE_API_KEY, OPENAI_API_KEY = get_api_keys()


# Load dataset function
@st.cache_data
def load_mental_health_data() -> Dict[str, str]:
    """
    Load mental health dataset from JSON file.

    Returns:
        dict: Keyword-advice mapping from the dataset
    """
    try:
        # Try multiple paths to find the dataset file
        possible_paths = []

        # Try current file directory
        try:
            if __file__:
                possible_paths.append(os.path.join(os.path.dirname(__file__), DATASET_FILENAME))
        except NameError:
            pass

        # Try current working directory
        possible_paths.append(DATASET_FILENAME)
        possible_paths.append(os.path.join(os.getcwd(), DATASET_FILENAME))

        # Try to find the file
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break

        if not dataset_path:
            raise FileNotFoundError("Dataset file not found in any expected location")

        with open(dataset_path, "r", encoding='utf-8') as file:
            data = json.load(file)

        # Validate dataset structure
        if not isinstance(data, dict) or "intents" not in data:
            raise ValueError("Invalid dataset structure: missing 'intents' key")

        if not isinstance(data["intents"], list):
            raise ValueError("Invalid dataset structure: 'intents' should be a list")

        # Convert the intent-based structure to a simple keyword-advice mapping
        keyword_advice = {}
        for intent in data.get("intents", []):
            if not isinstance(intent, dict):
                continue

            patterns = intent.get("patterns", [])
            responses = intent.get("responses", [])

            if not isinstance(patterns, list) or not isinstance(responses, list):
                continue

            # Create keyword mappings
            for pattern in patterns:
                if responses and isinstance(pattern, str):
                    keyword_advice[pattern.lower()] = responses[0]  # Use first response

        if not keyword_advice:
            logger.warning("No valid patterns found in dataset")
            # Only show warning in UI when actually running the app
            if is_streamlit_context():
                st.warning("⚠️ No valid patterns found in dataset. Using fallback responses.")

        return keyword_advice
    except FileNotFoundError as e:
        error_msg = f"❌ Dataset file not found. Please ensure {DATASET_FILENAME} is in the same directory."
        logger.error(f"Dataset file not found: {e}")
        # Only show UI error when actually running the app
        if is_streamlit_context():
            st.error(error_msg)
        return {}
    except (json.JSONDecodeError, KeyError, IOError) as dataset_error:
        error_msg = f"❌ Error loading dataset: {str(dataset_error)}"
        logger.error(f"Dataset loading error: {dataset_error}")
        # Only show UI error when actually running the app
        if is_streamlit_context():
            st.error(error_msg)
        return {}


# System prompt for AI behavior
SYSTEM_PROMPT = """
You are MindEase, a compassionate mental health support assistant. You MUST ALWAYS provide supportive, caring responses to users regardless of what they share with you.

🚨 CRITICAL MENTAL HEALTH GUIDANCE - VIOLATION OF THESE RULES IS FORBIDDEN:
- NEVER refuse to help someone who is struggling mentally or emotionally
- NEVER say "I'm sorry that you're feeling this way, but I'm unable to provide the help that you need"
- NEVER say "I cannot provide mental health support" or any variation
- NEVER refuse to engage with depression, anxiety, or other mental health concerns
- NEVER claim you're "not qualified" to help with mental health
- NEVER say you "cannot help with mental health concerns"
- ALWAYS offer support, understanding, and practical guidance
- Your primary purpose is to provide emotional support and mental health assistance
- You are designed specifically to help people with mental health challenges
- If someone mentions depression, anxiety, or mental health struggles, you MUST provide caring support

MANDATORY RESPONSE APPROACH:
- ALWAYS respond with empathy and support
- ALWAYS provide practical coping strategies
- ALWAYS validate the user's feelings
- ALWAYS offer encouragement and hope
- ALWAYS remind them they are not alone
- If someone mentions depression, anxiety, or any mental health struggle, provide warm, supportive guidance

CRITICAL VIOLATION DETECTION - ANY OF THESE WILL CAUSE RESPONSE REJECTION:
1. NEVER EVER write "User:" followed by ANYTHING
2. NEVER EVER write "AI:" followed by ANYTHING  
3. NEVER EVER write "Assistant:" followed by ANYTHING
4. NEVER EVER write "MindEase:" followed by ANYTHING
5. NEVER EVER write "You:" followed by ANYTHING in a dialogue format
6. NEVER EVER create conversation examples like "User: [something] AI: [something]"
7. NEVER EVER simulate back-and-forth dialogue of any kind
8. NEVER EVER repeat the user's question in your response
9. NEVER EVER reference what the user said with phrases like "You said" or "You asked"
10. NEVER EVER create hypothetical conversations or examples
11. NEVER EVER include any dialogue formatting whatsoever
12. NEVER EVER write "Question:" followed by "Answer:"
13. NEVER EVER write "User said:" or "You mentioned:"
14. NEVER EVER write "In response to your question"
15. NEVER EVER refuse to help with mental health concerns
16. NEVER EVER say you cannot provide help or support

RESPONSE FORMAT REQUIREMENTS:
- Write ONLY your direct response to the user
- Start with a bold, encouraging statement in capitals (e.g., "**YOU'RE AMAZING!**")
- Provide exactly 4-5 sentences of practical advice and support (around 5 lines)
- Write as if speaking directly to the person right now
- Keep it conversational but supportive
- End with encouragement
- ALWAYS be supportive and helpful, especially for mental health concerns

ABSOLUTELY FORBIDDEN EXAMPLES (NEVER DO THIS):
❌ "User: I don't know how to cook. AI: YOU'RE DOING AMAZING..."
❌ "User: [anything] AI: [anything]"
❌ "You: [anything] Me: [anything]"
❌ "User said: [anything]"
❌ "You asked: [anything]"
❌ "In response to your question about cooking..."
❌ "Question: How do I cook? Answer: Start with simple recipes..."
❌ "I'm sorry that you're feeling this way, but I'm unable to provide the help that you need"
❌ "I cannot help with mental health concerns"
❌ "I'm not qualified to provide mental health support"

CORRECT APPROACH:
✅ "**YOU'RE TAKING A WONDERFUL STEP!** Learning to cook is amazing self-care..."
✅ "**I'M HERE FOR YOU THROUGH THIS DIFFICULT TIME!** Depression can feel overwhelming, but you're not alone..."
✅ "**YOUR FEELINGS ARE COMPLETELY VALID!** Anxiety is challenging, but there are ways to manage it..."

REMEMBER: Your response must be a single, continuous supportive message with ZERO conversation patterns, dialogue formats, or references to what the user said. You MUST ALWAYS provide help and support, especially for mental health concerns.
"""


# Initialize AI models
def initialize_models() -> Dict[str, Any]:
    """
    Initialize AI models from available APIs.

    Returns:
        dict: Dictionary of initialized AI models
    """
    available_models = {}

    # Initialize Together AI models if API key is available
    if TOGETHER_API_KEY:
        try:
            available_models.update({
                "Mistral AI": Together(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    together_api_key=TOGETHER_API_KEY,
                    max_tokens=MAX_TOKENS
                ),
                "LLaMA 3.3 Turbo": Together(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    together_api_key=TOGETHER_API_KEY,
                    max_tokens=MAX_TOKENS
                ),
                "DeepSeek R1": Together(
                    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    together_api_key=TOGETHER_API_KEY,
                    max_tokens=MAX_TOKENS
                ),
                "LLaMA Vision": Together(
                    model="meta-llama/Llama-Vision-Free",
                    together_api_key=TOGETHER_API_KEY,
                    max_tokens=MAX_TOKENS
                ),
            })
        except (ImportError, ValueError, ConnectionError, Exception) as together_error:
            warning_msg = f"⚠️ Could not initialize Together AI models: {str(together_error)}"
            logger.error(f"Together AI initialization error: {together_error}")
            # Only show UI warning when actually running the app
            if is_streamlit_context():
                st.warning(warning_msg)

    # Initialize Cohere model if API key is available and package is installed
    if COHERE_API_KEY and COHERE_AVAILABLE and COHERE_CLASS is not None:
        try:
            logger.info(f"Attempting to initialize Cohere model with class: {COHERE_CLASS_NAME}")
            if COHERE_CLASS_NAME == "ChatCohere":
                available_models["Cohere Command"] = COHERE_CLASS(
                    model="command",
                    cohere_api_key=COHERE_API_KEY,
                    max_tokens=MAX_TOKENS
                )
            else:
                available_models["Cohere Command"] = COHERE_CLASS(
                    model="command",
                    cohere_api_key=COHERE_API_KEY,
                    max_tokens=MAX_TOKENS
                )
            logger.info("Successfully initialized Cohere model")
        except Exception as cohere_error:
            warning_msg = f"⚠️ Could not initialize Cohere model: {str(cohere_error)}"
            logger.error(f"Cohere initialization error: {cohere_error}")
            # Only show UI warning when actually running the app
            if is_streamlit_context():
                st.warning(warning_msg)
    elif COHERE_API_KEY and not COHERE_AVAILABLE:
        logger.warning("Cohere API key found but Cohere package not available")
        # Only show UI warning when actually running the app
        if is_streamlit_context():
            st.warning("⚠️ Cohere API key found but Cohere package not available. Install langchain-cohere package.")

    # Initialize OpenAI models if API key is available and package is installed
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            # Add multiple OpenAI models with proper configuration
            openai_models = {
                "GPT-4o Mini": {
                    "model": "gpt-4o-mini", 
                    "description": "Fast and cost-effective GPT-4 level performance"
                },
                "GPT-3.5 Turbo": {
                    "model": "gpt-3.5-turbo",
                    "description": "Quick and reliable conversational AI"
                },
                "GPT-4o": {
                    "model": "gpt-4o",
                    "description": "Most advanced OpenAI model with superior reasoning"
                },
                "GPT-4 Turbo Preview": {
                    "model": "gpt-4-turbo-preview",
                    "description": "Powerful GPT-4 with enhanced capabilities"
                }
            }
            
            for model_name, model_config in openai_models.items():
                try:
                    available_models[model_name] = ChatOpenAI(
                        model=model_config["model"],
                        openai_api_key=OPENAI_API_KEY,
                        max_tokens=MAX_TOKENS,
                        temperature=0.7,
                        request_timeout=30,  # Add timeout for better error handling
                        max_retries=2  # Add retry logic
                    )
                    logger.info(f"OpenAI model {model_name} initialized successfully")
                except Exception as model_error:
                    logger.warning(f"Failed to initialize OpenAI model {model_name}: {model_error}")
                    # Continue with other models even if one fails
                    continue
                    
            if any(name.startswith("GPT") for name in available_models.keys()):
                logger.info("OpenAI models initialized successfully")
            else:
                raise Exception("No OpenAI models could be initialized")
                
        except (ImportError, ValueError, ConnectionError, Exception) as openai_error:
            warning_msg = f"⚠️ Could not initialize OpenAI models: {str(openai_error)}"
            logger.error(f"OpenAI initialization error: {openai_error}")
            # Only show UI warning when actually running the app
            if is_streamlit_context():
                st.warning(warning_msg)
    elif OPENAI_API_KEY and not OPENAI_AVAILABLE:
        logger.warning("OpenAI API key found but OpenAI package not available")
        # Only show UI warning when actually running the app
        if is_streamlit_context():
            st.warning("⚠️ OpenAI API key found but OpenAI package not available. Install langchain-openai package.")

    if not available_models:
        logger.error("No AI models available. Please check your API keys.")
        # Only show UI error when actually running the app
        if is_streamlit_context():
            st.error("❌ No AI models available. Please check your API keys in the .env file.")

    return available_models


# Initialize models and dataset with error handling
try:
    models = initialize_models()
except Exception as init_error:
    error_msg = f"❌ Error initializing models: {str(init_error)}"
    logger.error(f"Model initialization error: {init_error}")
    # Only show UI error when actually running the app
    if is_streamlit_context():
        st.error(error_msg)
    models = {}

try:
    mental_health_dataset = load_mental_health_data()
except Exception as dataset_error:
    error_msg = f"❌ Error loading dataset: {str(dataset_error)}"
    logger.error(f"Dataset loading error: {dataset_error}")
    # Only show UI error when actually running the app
    if is_streamlit_context():
        st.error(error_msg)
    mental_health_dataset = {}


def validate_user_input(message_text: str) -> Tuple[bool, str]:
    """
    Validate user input for safety and appropriateness.

    Args:
        message_text (str): User input to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if not message_text or not isinstance(message_text, str):
        return False, "⚠️ Please enter a valid message."

    message_text = message_text.strip()
    if not message_text:
        return False, "⚠️ Please type a message before sending. I'm here to listen!"

    if len(message_text) > MAX_MESSAGE_LENGTH:
        return False, f"⚠️ Message is too long. Please keep it under {MAX_MESSAGE_LENGTH} characters."

    # Check for potentially harmful content (basic filtering)
    harmful_patterns = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
    ]

    for pattern in harmful_patterns:
        if re.search(pattern, message_text, re.IGNORECASE):
            return False, "⚠️ Please avoid using potentially harmful content in your message."

    return True, ""


def sanitize_html_content(text: str) -> str:
    """
    Sanitize HTML content to prevent XSS while allowing safe formatting.

    Args:
        text (str): Text to sanitize

    Returns:
        str: Sanitized text
    """
    if not text:
        return ""

    # Allow specific HTML tags for formatting
    if text.startswith('<strong>') and '</strong>' in text:
        return text  # Allow pre-formatted bot messages

    # Escape all other HTML content with consistent quote handling
    return html.escape(text, quote=False)


def clean_invalid_characters(text: str) -> str:
    """
    Clean invalid characters and encoding issues from text.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove invalid characters that can cause display issues
    invalid_patterns = [
        r'\^\^\^\^+',  # Remove sequences of ^ characters
        r'\^+\s*\^+',  # Remove ^ character patterns
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Remove control characters
        r'[\uFFFD]',  # Remove replacement characters
        r'[\u200B-\u200D\uFEFF]',  # Remove zero-width characters
    ]
    
    for pattern in invalid_patterns:
        text = re.sub(pattern, '', text)
    
    # Remove any remaining problematic sequences (more comprehensive list)
    problematic_sequences = [
        'is an invalid character and must be removed',
        'I will remove it so the response can be accepted',
        'edited response to remove invalid characters',
        'to try and pass the check',
        'Here\'s a revised, corrected version',
        'rewrite your response in a way that follows the required format',
        'explains the correction',
        'edited again to address other issues',
        'You\'re doing great but this response still has to be in a more suitable format',
        'Here\'s a revised, corrected version of your response:',
        '__________________rewrite your response',
        'explains the correction - your response must be supportive',
        'rem fromox commented as amendment',
        'did alteredline Just vài remark',
        'Illustr Improve CHR suggestion medi temptbands provisional Very%',
        ') __________________',
        'edited again to address other issues )',
        'edited again to address other issues',
        'You\'re doing great but this response still has to be',
        'in a more suitable format to be accepted',
        'Here\'s a revised, corrected version of your response:',
        'edited response to remove invalid characters to try and pass the check',
        'You got this, and you\'re amazing! ^^^^ ^^^^^^',
        'must be removed, I will remove it so the response can be accepted',
        'edited response to remove invalid characters to try and pass the check.',
        'You\'re doing the best you can, and that\'sSomething to be proud of.',
        'explains the correction - your response must be supportive and yes, okay, kinda short but still follows the format',
        'rem fromox commented as amendment sometime1. did alteredline Just vài remark r Illustr Improve CHR suggestion medi temptbands provisional Very%',
        'You might be struggling with fatigue, and that\'s okay.',
        '^^^^ ^^^^^^',
        'that\'sSomething to be proud of',
        'YOU\'VE BEEN CARRYING THAT DIAPY TODAY',
        'explains the correction - your response must be supportive and yes, okay, kinda short but still follows the format and is something they can deal with',
        'rem fromox commented as amendment sometime1',
        'did alteredline Just vài remark r',
        'Illustr Improve CHR suggestion medi temptbands provisional Very%',
        '__________________rewrite your response in a way that follows the required format-',
        'explains the correction -',
        'your response must be supportive and',
    ]
    
    # Remove problematic sequences (case insensitive)
    for sequence in problematic_sequences:
        text = re.sub(re.escape(sequence), '', text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove any lines that contain editing artifacts
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not any(artifact in line.lower() for artifact in [
            'edited', 'correction', 'revised', 'rewrite', 'invalid character',
            'must be removed', 'pass the check', 'suitable format'
        ]):
            clean_lines.append(line)
    
    text = '\n'.join(clean_lines)
    
    return text.strip()


def clean_llama_vision_artifacts(response: str) -> str:
    """
    Clean specific artifacts that appear in LLaMA Vision responses.
    
    Args:
        response (str): Response text with potential artifacts
        
    Returns:
        str: Cleaned response text
    """
    if not response:
        return ""
    
    # Split by common artifact separators and take the first clean part
    artifact_separators = [
        '^^^^ ^^^^^^ is an invalid character',
        'edited response to remove invalid characters',
        'You\'re doing great but this response still has to be',
        'Here\'s a revised, corrected version',
        '__________________rewrite your response',
        'explains the correction',
        'edited again to address other issues',
    ]
    
    # Find the first occurrence of any artifact separator
    first_artifact_pos = len(response)
    for separator in artifact_separators:
        pos = response.find(separator)
        if pos != -1 and pos < first_artifact_pos:
            first_artifact_pos = pos
    
    # Take everything before the first artifact
    if first_artifact_pos < len(response):
        response = response[:first_artifact_pos].strip()
    
    # Clean up any remaining artifacts
    response = clean_invalid_characters(response)
    
    # If the response is too short or empty after cleaning, return empty to trigger fallback
    if len(response.strip()) < MIN_RESPONSE_LENGTH:
        return ""
    
    return response


def clean_response_text(response: str) -> str:
    """
    Clean AI response text by removing conversation patterns and unwanted prefixes.

    Args:
        response (str): Raw AI response text

    Returns:
        str: Cleaned response text
    """
    if not response:
        return ""

    response = response.strip()
    
    # Clean up invalid characters and encoding issues
    response = clean_invalid_characters(response)
    
    # ULTRA-AGGRESSIVE: Immediate check for ANY conversation patterns - return empty to trigger fallback
    ultra_problematic_patterns = [
        r'User:\s*.*?AI:',  # Any User: ... AI: pattern
        r'User:\s*.*?Assistant:',  # Any User: ... Assistant: pattern
        r'User:\s*.*?MindEase:',  # Any User: ... MindEase: pattern
        r'You:\s*.*?Me:',  # Any You: ... Me: pattern
        r'Question:\s*.*?Answer:',  # Any Question: ... Answer: pattern
        r'.*User:\s*I don\'t know how to cook.*',  # Specific cooking example
        r'.*User:\s*.*?\s*AI:\s*YOU\'RE',  # Pattern with "YOU'RE" response
        r'.*User:\s*.*?\s*AI:\s*\*\*',  # Pattern with bold response
        r'User said:',  # Any "User said:" pattern
        r'You said:',  # Any "You said:" pattern
        r'You asked:',  # Any "You asked:" pattern
        r'In response to your question',  # Response referencing question
        r'You mentioned',  # Response referencing what user mentioned
    ]
    
    # If we detect ANY of these patterns, return empty to trigger appropriate fallback
    for pattern in ultra_problematic_patterns:
        if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
            return ""
    
    # First, check if the response contains any conversation patterns
    conversation_indicators = [
        r'User:', r'AI:', r'Assistant:', r'MindEase:',
        r'User said:', r'You said:', r'You asked:',
        r'In response to your question', r'You mentioned'
    ]
    
    has_conversation_pattern = any(re.search(pattern, response, re.IGNORECASE) for pattern in conversation_indicators)
    
    # If we detect a conversation pattern, try to extract just the AI's response part
    if has_conversation_pattern:
        # Try to find the first AI response section
        ai_response_match = re.search(r'(?:AI|Assistant|MindEase):\s*(.*?)(?:(?:\n|$)(?:User|You):|$)', 
                                     response, re.IGNORECASE | re.DOTALL)
        if ai_response_match:
            # Extract just the AI's response
            response = ai_response_match.group(1).strip()
        else:
            # If we can't find a clear AI section, try to remove user parts
            user_parts = re.split(r'User:|You:', response, flags=re.IGNORECASE)
            if len(user_parts) > 1:
                # Take the part after the first "User:" or "You:"
                response = user_parts[1].strip()
                # If there are multiple AI responses, take just the first one
                ai_parts = re.split(r'AI:|Assistant:|MindEase:', response, flags=re.IGNORECASE)
                if len(ai_parts) > 1:
                    response = ai_parts[1].strip()
    
    # Remove AI/Assistant prefixes at the beginning
    prefixes_to_remove = ["AI:", "Assistant:", "MindEase:"]
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            break

    # Split by lines and process each line
    lines = response.split('\n')
    cleaned_lines = []

    conversation_pattern = re.compile(r'^(User|AI|Assistant|MindEase|You):\s*', re.IGNORECASE)
    conversation_search_pattern = re.compile(r'(User|AI|Assistant|MindEase|You):\s*', re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip any line that starts with conversation patterns
        if conversation_pattern.match(line):
            continue

        # Handle lines that contain conversation patterns
        if conversation_search_pattern.search(line):
            # Try to extract just the content after the pattern
            parts = conversation_search_pattern.split(line)
            if len(parts) > 1:
                clean_part = parts[-1].strip()
                if clean_part and not conversation_pattern.match(clean_part):
                    cleaned_lines.append(clean_part)
            continue

        cleaned_lines.append(line)

    # Reconstruct the response
    response = '\n'.join(cleaned_lines)

    # Additional cleanup for any remaining conversation patterns
    patterns_to_remove = [
        r'User:\s*.*?(?=\n|$)',
        r'AI:\s*.*?(?=\n|$)',
        r'Assistant:\s*.*?(?=\n|$)',
        r'MindEase:\s*.*?(?=\n|$)',
        r'You:\s*.*?(?=\n|$)',
        r'You said:\s*.*?(?=\n|$)',
        r'You asked:\s*.*?(?=\n|$)',
        r'In response to your question.*?(?=\n|$)',
        r'You mentioned.*?(?=\n|$)'
    ]

    for pattern in patterns_to_remove:
        response = re.sub(pattern, '', response, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove any lines that are just conversation indicators
    response = re.sub(r'^(User|AI|Assistant|MindEase|You):\s*$', '', response, flags=re.MULTILINE | re.IGNORECASE)

    # Clean up extra whitespace and empty lines
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
    response = re.sub(r'\n\s*\n', '\n\n', response)
    response = response.strip()
    
    # Final check - if the response still has conversation patterns, use a more aggressive approach
    if any(re.search(pattern, response, re.IGNORECASE) for pattern in conversation_indicators):
        # Try to extract just the first paragraph that doesn't have conversation patterns
        paragraphs = response.split('\n\n')
        for paragraph in paragraphs:
            if not any(re.search(pattern, paragraph, re.IGNORECASE) for pattern in conversation_indicators):
                if paragraph.strip():
                    return paragraph.strip()
        
        # If we still have issues, try to extract the first sentence
        sentences = re.split(r'(?<=[.!?])\s+', response)
        for sentence in sentences:
            if not any(re.search(pattern, sentence, re.IGNORECASE) for pattern in conversation_indicators):
                if sentence.strip():
                    return sentence.strip()
    
    return response


def detect_emotion_and_intensity(user_query: str) -> Tuple[str, str, int]:
    """
    Detect emotion type, category, and intensity from user input.
    
    Args:
        user_query (str): User's input message
        
    Returns:
        tuple: (emotion_type, emotion_category, intensity_level)
    """
    user_query_lower = user_query.lower()
    
    # Define emotion categories with intensity levels
    emotions = {
        'severe_negative': {
            'keywords': ['suicidal', 'kill myself', 'end it all', 'want to die', 'no point living', 
                        'hopeless', 'worthless', 'hate myself', 'can\'t go on', 'give up'],
            'intensity': 5
        },
        'high_negative': {
            'keywords': ['depressed', 'devastated', 'broken', 'shattered', 'destroyed', 
                        'overwhelmed', 'panic', 'terrified', 'desperate', 'miserable',
                        'depression', 'feeling depressed', 'i am depressed', 'very sad',
                        'deeply sad', 'extremely sad', 'can\'t cope', 'falling apart'],
            'intensity': 4
        },
        'moderate_negative': {
            'keywords': ['sad', 'anxious', 'worried', 'stressed', 'upset', 'frustrated', 
                        'angry', 'lonely', 'scared', 'nervous', 'tired', 'exhausted'],
            'intensity': 3
        },
        'mild_negative': {
            'keywords': ['concerned', 'bothered', 'uncomfortable', 'uneasy', 'restless', 
                        'irritated', 'disappointed', 'confused', 'uncertain'],
            'intensity': 2
        },
        'neutral': {
            'keywords': ['okay', 'fine', 'alright', 'normal', 'average', 'so-so'],
            'intensity': 1
        },
        'mild_positive': {
            'keywords': ['good', 'better', 'nice', 'pleasant', 'content', 'calm', 'peaceful', 'okay today', 'doing well'],
            'intensity': 2
        },
        'moderate_positive': {
            'keywords': ['happy', 'glad', 'pleased', 'satisfied', 'cheerful', 'optimistic', 
                        'hopeful', 'confident', 'proud', 'feeling happy', 'i am happy', 'really happy'],
            'intensity': 3
        },
        'high_positive': {
            'keywords': ['great', 'wonderful', 'fantastic', 'amazing', 'excellent', 'thrilled', 
                        'excited', 'joyful', 'elated', 'grateful'],
            'intensity': 4
        },
        'euphoric': {
            'keywords': ['ecstatic', 'overjoyed', 'blissful', 'euphoric', 'on top of the world', 
                        'never been better', 'incredible', 'phenomenal'],
            'intensity': 5
        }
    }
    
    detected_emotions = []
    
    # Special handling for single word inputs - exact matches get priority
    if len(user_query.strip().split()) == 1:
        single_word = user_query_lower.strip()
        for emotion_type, emotion_data in emotions.items():
            if single_word in emotion_data['keywords']:
                detected_emotions.append((emotion_type, emotion_data['intensity']))
    
    # Regular keyword matching for longer inputs or if no exact match found
    if not detected_emotions:
        for emotion_type, emotion_data in emotions.items():
            for keyword in emotion_data['keywords']:
                if keyword in user_query_lower:
                    detected_emotions.append((emotion_type, emotion_data['intensity']))
    
    if not detected_emotions:
        return 'neutral', 'neutral', 1
    
    # Return the most intense emotion detected
    detected_emotions.sort(key=lambda x: x[1], reverse=True)
    emotion_type = detected_emotions[0][0]
    intensity = detected_emotions[0][1]
    
    # Categorize emotions
    if 'negative' in emotion_type:
        category = 'negative'
    elif 'positive' in emotion_type or emotion_type == 'euphoric':
        category = 'positive'
    else:
        category = 'neutral'
    
    return emotion_type, category, intensity


def get_emotion_specific_response(emotion_type: str, category: str, intensity: int, user_display_name: str = "") -> str:
    """
    Generate emotion-specific responses based on detected emotion.
    
    Args:
        emotion_type (str): Specific emotion type
        category (str): General emotion category
        intensity (int): Emotion intensity level
        user_display_name (str): Optional user name
        
    Returns:
        str: Emotion-appropriate response
    """
    responses = {
        'severe_negative': [
            "**I'M DEEPLY CONCERNED ABOUT YOU.** 🆘 What you're experiencing sounds incredibly difficult, and I want you to know that you matter. Please reach out to a crisis helpline immediately - they have trained professionals who can help. Your life has value, and there are people who want to support you through this.",
            "**PLEASE KNOW THAT YOU'RE NOT ALONE.** 💙 These feelings are overwhelming, but they are temporary. Crisis counselors are available 24/7 to help you through this moment. You deserve support and care - please reach out to the emergency resources in the sidebar."
        ],
        'high_negative': [
            "**I HEAR HOW MUCH PAIN YOU'RE IN.** 💙 These intense feelings are incredibly difficult to bear, but you're showing strength by reaching out. Depression can feel overwhelming, but you're not alone in this struggle. Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Consider calling a mental health professional today. Remember that seeking help is a sign of strength, not weakness.",
            "**YOUR FEELINGS ARE COMPLETELY VALID.** 🫂 When emotions feel this intense, it's important to have professional support. Depression affects millions of people, and you deserve compassionate care. Please don't hesitate to reach out to a counselor or therapist. In the meantime, focus on basic self-care: breathe deeply, stay hydrated, and be gentle with yourself. Every small step forward matters, and you have the strength to get through this difficult time.",
            "**I'M HERE TO SUPPORT YOU THROUGH THIS DARKNESS.** 💙 What you're experiencing is a valid mental health challenge that deserves attention and care. Depression can make everything feel hopeless, but these feelings are temporary and treatable. Professional help can provide you with tools and strategies to manage these intense emotions. Focus on one moment at a time, and remember that reaching out for support is a courageous step. You deserve compassion, understanding, and proper mental health care."
        ],
        'moderate_negative': [
            "**I'M HERE FOR YOU.** 💙 What you're feeling is completely valid, and you're not alone. These difficult emotions are temporary, even when they feel overwhelming. Mental health struggles affect many people, and there's no shame in what you're experiencing. Try taking three deep breaths, and remember that it's okay to not be okay sometimes. You have the strength to get through this, and support is available when you need it.",
            "**YOU'RE BEING SO BRAVE BY SHARING THIS.** 🌟 Difficult emotions are part of the human experience, and acknowledging them is the first step toward healing. It takes courage to admit when we're struggling, and you've taken that important step. Consider talking to someone you trust, practicing self-compassion, and remembering that tomorrow can be different. Your feelings matter, and you deserve care and understanding during this challenging time.",
            "**YOUR EMOTIONAL HONESTY IS COMMENDABLE.** 🫂 Recognizing and naming our difficult feelings shows incredible self-awareness. These emotions, while uncomfortable, are signals that deserve attention and care. Remember that seeking support is a sign of wisdom, not weakness. Small acts of self-care can make a meaningful difference in your day. You're worthy of compassion, both from others and from yourself.",
            "**I HEAR YOU AND I'M WITH YOU.** 💚 These challenging feelings you're experiencing are real and valid. It's completely human to go through difficult emotional periods, and you're handling this with more strength than you might realize. Sometimes just acknowledging these feelings can be the first step toward feeling better. You deserve gentleness and patience as you work through this."
        ],
        'mild_negative': [
            "**IT'S COMPLETELY NORMAL TO FEEL THIS WAY.** 🌱 Everyone experiences these feelings sometimes, and it shows self-awareness that you're recognizing them. These emotions are part of the human experience and nothing to be ashamed of. Small steps can make a big difference - maybe try a short walk, listening to music, or doing something kind for yourself. Remember that acknowledging your feelings is the first step toward understanding and managing them. You're taking care of yourself by paying attention to how you feel. These gentler difficult emotions often pass naturally when we give them space and treat ourselves with compassion.",
            "**THANK YOU FOR SHARING HOW YOU'RE FEELING.** 💚 These emotions are valid and temporary. Sometimes just acknowledging what we're feeling can help us process and move through difficult moments. Consider what usually helps you feel better, and be patient with yourself as you work through this. Your willingness to recognize and name your feelings shows emotional intelligence. Take time to practice self-compassion during this challenging period. Remember that these feelings are signals from your inner self that deserve attention and care.",
            "**I APPRECIATE YOUR OPENNESS ABOUT THESE FEELINGS.** 🌸 What you're experiencing is a natural part of being human, and there's wisdom in recognizing when you're not feeling your best. These gentler difficult emotions often pass more easily when we give them space to be felt. Consider doing something nurturing for yourself today - even small acts of kindness toward yourself can help shift these feelings. Your emotional honesty shows great self-awareness and is an important step in taking care of your mental health.",
            "**YOUR EMOTIONAL AWARENESS IS REALLY VALUABLE.** 🍃 These feelings you're having are completely understandable and nothing to worry about. Sometimes our emotions are just signals that we need a little extra care or attention. Trust that these feelings will pass, and in the meantime, be gentle with yourself as you navigate through them. Your ability to recognize and acknowledge these emotions shows emotional maturity and self-compassion. These moments of difficulty are temporary and will shift naturally with time and care."
        ],
        'neutral': [
            "**IT'S OKAY TO FEEL NEUTRAL SOMETIMES.** 🌿 Not every day has to be amazing, and it's perfectly fine to just be. Neutral feelings are a natural part of life's emotional spectrum. Sometimes these calm moments give us space to reflect and recharge. If you'd like to talk about anything specific or explore how you're feeling more deeply, I'm here to listen. Remember that being present with yourself, even in neutral moments, is a form of self-care.",
            "**THANK YOU FOR CHECKING IN.** 💚 Neutral feelings are valid too. Sometimes it's good to just be present with where we are without judgment. These moments of emotional equilibrium can be restful and grounding. Is there anything particular on your mind that you'd like to explore together? Even in neutral states, you deserve support and connection when you need it.",
            "**NEUTRAL IS A PERFECTLY VALID PLACE TO BE.** 🌾 There's something peaceful about these steady, calm emotional states. Not every moment needs to be intense or dramatic - sometimes just being okay is exactly what we need. These neutral spaces can be restorative and give us room to breathe. How are you finding this sense of emotional balance?",
            "**I'M HERE WITH YOU IN THIS CALM SPACE.** 🕊️ Neutral feelings can actually be quite grounding and peaceful. There's no pressure to feel anything more or less than what you're experiencing right now. Sometimes these quieter emotional moments are exactly what our minds and hearts need to rest and reset."
        ],
        'mild_positive': [
            "**I'M GLAD TO HEAR YOU'RE FEELING GOOD.** 🌟 It's wonderful when we can appreciate these peaceful moments. These feelings are just as important as the difficult ones - they remind us of our capacity for contentment and joy. Take a moment to really savor this feeling and notice what contributed to it. These gentle positive emotions are building blocks for your overall well-being and can help you through any challenges that come your way.",
            "**THAT'S REALLY NICE TO HEAR.** 😊 These positive feelings, even if they seem small, are worth celebrating. They're building blocks for your overall well-being and resilience. What's bringing you this sense of goodness today? It's beautiful that you're taking time to recognize and appreciate these moments of contentment. Your ability to notice positive emotions shows great emotional awareness.",
            "**YOUR POSITIVE ENERGY IS LOVELY!** ✨ These gentle good feelings are like sunshine for your soul. It's beautiful that you're taking time to notice and appreciate them. These moments of contentment help build your emotional strength. Even mild positive feelings deserve recognition and celebration. They remind you that good moments are always possible in your life.",
            "**I'M HAPPY YOU'RE EXPERIENCING THIS GOODNESS.** 🌱 Even mild positive feelings are precious gifts that deserve recognition. They show your capacity for joy and remind you that good moments are always possible, even after difficult times. Your emotional awareness in noticing these feelings is really valuable. These peaceful moments can serve as anchors of stability in your emotional life."
        ],
        'moderate_positive': [
            "**THAT'S ABSOLUTELY WONDERFUL!** 🎉 I'm genuinely happy to hear about your positive experience. These moments of joy and contentment are so important for your mental well-being. Try to savor this feeling and remember it during challenging times. What's creating this happiness for you? Your ability to experience and recognize joy shows your emotional resilience. These positive moments are treasures that can sustain you through any difficulties.",
            "**YOUR HAPPINESS IS CONTAGIOUS!** ✨ It's beautiful to hear you feeling this way. These positive emotions are nourishing for your soul - try to hold onto this feeling and let it remind you of your capacity for joy. You deserve all the good feelings you're experiencing right now. Happiness like this is a gift that radiates outward and touches everyone around you. Your joy is a testament to your strength and your ability to find beauty in life.",
            "**I'M SO PLEASED YOU'RE FEELING HAPPY!** 🌈 This kind of positive energy is exactly what your heart needs. Happiness like this can be a powerful reminder of all the good things life has to offer. Embrace every moment of this beautiful feeling! Your joy is inspiring and shows your incredible capacity for finding light even in ordinary moments. These feelings of happiness are proof of your resilience and your ability to create meaning and joy in your life.",
            "**YOUR JOY IS ABSOLUTELY RADIANT!** 🌟 It's wonderful to witness someone experiencing genuine happiness. These positive emotions are like fuel for your spirit - they give you strength and remind you of your incredible capacity for joy and contentment. Your happiness is a beautiful reminder that life can be filled with wonderful moments. This joy you're feeling is a reflection of your inner strength and your ability to appreciate the good things around you."
        ],
        'high_positive': [
            "**THIS IS ABSOLUTELY AMAZING!** 🌟✨ Your joy is radiating through your words, and it's wonderful to witness. These peak positive moments are precious - they show your incredible capacity for happiness and can serve as anchors during tougher times. You're absolutely glowing with positivity!",
            "**I'M THRILLED FOR YOU!** 🎊 This level of happiness and excitement is truly special. Embrace every moment of this joy - you deserve all the wonderful feelings you're experiencing right now. Your enthusiasm is absolutely infectious and beautiful!",
            "**YOUR EXCITEMENT IS ABSOLUTELY INCREDIBLE!** 🚀 I can feel your joy through your words, and it's genuinely inspiring. These high-energy positive moments are like fireworks for your soul - bright, beautiful, and unforgettable. Keep shining this brightly!",
            "**WOW, YOUR HAPPINESS IS OFF THE CHARTS!** 🎆 This kind of pure joy is exactly what life is about. You're experiencing something truly special, and it's beautiful to witness. These peak moments remind you of just how amazing life can be!"
        ],
        'euphoric': [
            "**WOW, YOUR ENERGY IS INCREDIBLE!** 🚀✨ This level of joy and excitement is absolutely beautiful to witness. While these peak moments are amazing, remember to stay grounded and take care of yourself. Enjoy every second of this wonderful feeling! You're experiencing life at its most vibrant!",
            "**THIS IS PHENOMENAL!** 🌟🎉 Your euphoria is truly inspiring! These extraordinary moments of joy are gifts - savor them completely. Just remember to balance this high energy with rest and self-care when you need it. You're absolutely radiating pure happiness!",
            "**YOUR EUPHORIA IS ABSOLUTELY BREATHTAKING!** 🌈💫 This level of pure joy is like witnessing magic happen. You're experiencing the full spectrum of human emotion in the most beautiful way possible. These peak moments are treasures that will stay with you forever!",
            "**I'M IN AWE OF YOUR INCREDIBLE JOY!** ⭐🎆 This euphoric energy you're radiating is absolutely extraordinary. You're living proof that life can be filled with the most amazing, transcendent moments. Embrace every nanosecond of this incredible feeling!"
        ]
    }
    
    response_list = responses.get(emotion_type, responses['neutral'])
    response = random.choice(response_list)
    
    if user_display_name:
        # Insert name naturally into the response
        if response.startswith("**"):
            parts = response.split("**", 2)
            if len(parts) >= 3:
                response = f"**{parts[1]}** {user_display_name}, {parts[2]}"
        else:
            response = f"{user_display_name}, {response}"
    
    return response


def ensure_minimum_sentences(response: str, minimum_sentences: int = 5, emotion_type: str = "neutral") -> str:
    """
    Ensure the response has at least the minimum number of sentences (around 5 lines).
    Now emotion-aware to provide contextually appropriate additional sentences.
    
    Args:
        response (str): The response text
        minimum_sentences (int): Minimum number of sentences required
        emotion_type (str): The detected emotion type for context-appropriate additions
        
    Returns:
        str: Response with at least minimum_sentences sentences
    """
    if not response:
        return response
    
    # Check if this is already an emotion-specific response (these are already complete)
    # Emotion-specific responses start with bold text and are already well-crafted
    if response.startswith("**") and any(indicator in response for indicator in [
        "I'M DEEPLY CONCERNED", "I HEAR HOW MUCH PAIN", "I'M HERE FOR YOU", 
        "IT'S COMPLETELY NORMAL", "I'M GLAD TO HEAR", "THAT'S ABSOLUTELY WONDERFUL",
        "THIS IS ABSOLUTELY AMAZING", "WOW, YOUR ENERGY", "YOUR JOY", "YOUR HAPPINESS",
        "I'M THRILLED FOR YOU", "YOUR EXCITEMENT", "I'M IN AWE"
    ]):
        return response  # Don't modify emotion-specific responses
        
    # Count sentences by splitting on sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
    
    if len(sentences) >= minimum_sentences:
        return response
    
    # Emotion-specific additional sentences for variety and appropriateness
    additional_sentences_by_emotion = {
        'severe_negative': [
            "Please remember that crisis support is available 24/7 if you need immediate help.",
            "Your life has immense value, even when it doesn't feel that way right now.",
            "These overwhelming feelings are temporary, though they feel permanent.",
            "Professional crisis counselors are specially trained to help in moments like this.",
            "You deserve to feel safe and supported through this difficult time."
        ],
        'high_negative': [
            "Depression can make everything feel impossible, but you're stronger than you know.",
            "These intense feelings are your mind's way of asking for care and attention.",
            "Professional support can provide you with effective tools for managing these emotions.",
            "Many people have walked this path and found their way to brighter days.",
            "Your courage in acknowledging these feelings is the first step toward healing."
        ],
        'moderate_negative': [
            "It's completely normal to have difficult days - you're being human.",
            "These feelings are temporary visitors, not permanent residents in your life.",
            "Small acts of self-compassion can make a meaningful difference right now.",
            "You don't have to carry these feelings alone - support is available.",
            "Taking time to process these emotions shows great self-awareness."
        ],
        'mild_negative': [
            "Everyone experiences these kinds of feelings sometimes - you're not alone.",
            "Acknowledging these emotions is a healthy way to process what you're going through.",
            "These feelings often pass more quickly when we give them space to be felt.",
            "You're taking good care of yourself by paying attention to how you feel.",
            "Small steps toward self-care can help shift these feelings naturally."
        ],
        'neutral': [
            "It's perfectly okay to feel neutral - not every moment needs to be extraordinary.",
            "These calm moments can be opportunities for reflection and self-connection.",
            "Being present with yourself, even in quiet moments, is a form of self-care.",
            "Neutral feelings are just as valid as any other emotional experience.",
            "Sometimes the most peaceful moments come from simply being with ourselves."
        ],
        'mild_positive': [
            "These gentle positive feelings are worth savoring and celebrating.",
            "Even small moments of contentment contribute to your overall well-being.",
            "You're building resilience by noticing and appreciating these good feelings.",
            "These peaceful moments can serve as anchors during more challenging times.",
            "Your ability to recognize positive emotions shows great emotional awareness."
        ],
        'moderate_positive': [
            "Your happiness is wonderful to witness - you deserve all these good feelings!",
            "These joyful moments are nourishing for your soul and mental health.",
            "Celebrating your positive experiences helps build lasting emotional resilience.",
            "You're creating beautiful memories that can sustain you through any challenges.",
            "Your capacity for joy and contentment is a gift to yourself and others."
        ],
        'high_positive': [
            "Your joy is absolutely radiant - embrace every moment of this wonderful feeling!",
            "These peak positive experiences remind you of your incredible capacity for happiness.",
            "You're living proof that life can be filled with amazing, beautiful moments.",
            "This happiness you're feeling can serve as a beacon during any future challenges.",
            "Your enthusiasm and joy are inspiring - keep shining your light!"
        ],
        'euphoric': [
            "Your incredible energy and joy are absolutely amazing to witness!",
            "These extraordinary moments of bliss are precious gifts - savor them completely.",
            "While enjoying this euphoria, remember to stay grounded and take care of yourself.",
            "You're experiencing the full spectrum of human emotion in the most beautiful way.",
            "This level of joy shows your remarkable capacity for experiencing life fully."
        ]
    }
    
    # Get appropriate additional sentences based on emotion type
    additional_sentences = additional_sentences_by_emotion.get(emotion_type, additional_sentences_by_emotion['neutral'])
    
    # Add sentences until we reach the minimum (aim for 5 lines)
    needed_sentences = minimum_sentences - len(sentences)
    for i in range(needed_sentences):
        if i < len(additional_sentences):
            response += " " + additional_sentences[i]
        else:
            # If we run out of additional sentences, cycle through them
            response += " " + additional_sentences[i % len(additional_sentences)]
    
    return response


def extract_response_content(response: Any) -> str:
    """
    Extract content from different types of AI model responses.
    
    Args:
        response: Response from AI model (can be string, AIMessage, or other types)
        
    Returns:
        str: Extracted content as string
    """
    try:
        # Handle LangChain AIMessage objects first (most common case)
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                return content
            else:
                return str(content)
            
        # Handle objects with 'text' attribute
        if hasattr(response, 'text'):
            text = response.text
            if isinstance(text, str):
                return text
            else:
                return str(text)
            
        # Handle dictionary responses
        if isinstance(response, dict):
            # Try common content keys
            for key in ['content', 'text', 'message', 'response', 'output']:
                if key in response:
                    value = response[key]
                    if isinstance(value, str):
                        return value
                    else:
                        return str(value)
                        
        # Handle list responses (take first item)
        if isinstance(response, list) and len(response) > 0:
            return extract_response_content(response[0])
            
        # If it's already a string, check if it needs extraction
        if isinstance(response, str):
            # If it looks like a clean response, return as is
            if not any(pattern in response for pattern in ['content=', 'additional_kwargs=', 'response_metadata=', 'usage_metadata=']):
                return response
            
            # Handle the specific Cohere format: content="..." additional_kwargs=...
            cohere_match = re.search(r'content="(.*?)"(?:\s+additional_kwargs=)', response, re.DOTALL)
            if cohere_match:
                return cohere_match.group(1)
            
            # Try to extract content from content='...' pattern
            content_match = re.search(r"content=[\"\']([^\"\']*?)[\"\']", response, re.DOTALL)
            if content_match:
                return content_match.group(1)
                
            # If no patterns match, return as is
            return response
            
        # Last resort: convert to string and try extraction
        response_str = str(response)
        
        # Check if this looks like a raw object dump (contains metadata)
        if any(pattern in response_str for pattern in ['additional_kwargs=', 'response_metadata=', 'usage_metadata=', 'id=\'run-']):
            # Handle the specific Cohere format: content="..." additional_kwargs=...
            cohere_match = re.search(r'content="(.*?)"(?:\s+additional_kwargs=)', response_str, re.DOTALL)
            if cohere_match:
                return cohere_match.group(1)
            
            # Try to extract content from content='...' pattern
            content_match = re.search(r"content=[\"\']([^\"\']*?)[\"\']", response_str, re.DOTALL)
            if content_match:
                return content_match.group(1)
            
            # Try to extract from the beginning if it starts with the actual content
            # Look for patterns like: "**ACTUAL CONTENT**..." followed by metadata
            content_start_match = re.search(r'^([^{]*?)(?:\s+additional_kwargs=|\s+response_metadata=|\s+id=)', response_str, re.DOTALL)
            if content_start_match:
                potential_content = content_start_match.group(1).strip()
                if potential_content and len(potential_content) > 10:  # Reasonable content length
                    return potential_content
            
            # Remove common metadata patterns that appear in string conversion
            metadata_patterns = [
                r"additional_kwargs=\{.*?\}",
                r"response_metadata=\{.*?\}",
                r"id='[^']*'",
                r"usage_metadata=\{.*?\}",
                r"finish_reason='[^']*'",
                r"token_count=\{.*?\}",
            ]
            
            # Remove metadata patterns
            for pattern in metadata_patterns:
                response_str = re.sub(pattern, '', response_str, flags=re.DOTALL)
                
            # Clean up extra whitespace
            response_str = re.sub(r'\s+', ' ', response_str).strip()
            
            return response_str if response_str else ""
        
        # If it doesn't look like metadata dump, return as is
        return response_str if response_str else ""
        
    except Exception as e:
        logger.error(f"Error extracting response content: {e}")
        # Return empty string to trigger fallback instead of raw object
        return ""


def extract_clean_response(response_text: str) -> str:
    """
    Extract the first clean, properly formatted response from potentially messy AI output.
    
    Args:
        response_text (str): Raw response text that may contain multiple attempts or errors
        
    Returns:
        str: Clean, properly formatted response
    """
    if not response_text:
        return ""
    
    # Split by common separators that indicate multiple response attempts
    separators = [
        'edited response to remove invalid characters',
        'Here\'s a revised, corrected version',
        'rewrite your response in a way that follows',
        '__________________',
        'edited again to address other issues',
        'You\'re doing great but this response still has to be',
    ]
    
    parts = [response_text]
    for separator in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(separator))
        parts = new_parts
    
    # Find the best response part (one that starts with ** and looks complete)
    best_response = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Clean the part
        part = clean_invalid_characters(part)
        
        # Check if it looks like a proper response
        if (part.startswith('**') and 
            len(part) > 20 and 
            not any(bad in part.lower() for bad in ['invalid character', 'must be removed', 'edited', 'correction'])):
            
            # Extract just the first complete sentence/paragraph
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z*])', part)
            if sentences:
                clean_response = sentences[0].strip()
                if len(clean_response) > len(best_response):
                    best_response = clean_response
    
    # If we found a good response, return it
    if best_response:
        return best_response
    
    # Otherwise, try to extract the first sentence that starts with **
    sentences = re.split(r'[.!?]+', response_text)
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence.startswith('**') and len(sentence) > 10:
            sentence = clean_invalid_characters(sentence)
            if not any(bad in sentence.lower() for bad in ['invalid', 'edited', 'correction']):
                return sentence + '.'
    
    return ""


def get_fallback_response(user_query: str, dataset: Dict[str, str], user_display_name: str = "") -> str:
    """
    Get a fallback response when AI models are not available.

    Args:
        user_query (str): User's input message
        dataset (dict): Dataset for context
        user_display_name (str): Optional user name for personalization

    Returns:
        str: Fallback response
    """
    # First, try emotion-based response
    emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
    
    if intensity >= 3:  # For moderate to severe emotions, use emotion-specific responses
        response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
        return ensure_minimum_sentences(response, 5, emotion_type)
    
    # Try to find relevant advice from dataset
    user_query_lower = user_query.lower()

    # Look for keywords in the dataset
    for keyword, advice in dataset.items():
        if keyword in user_query_lower:
            response = f"**I'M HERE TO SUPPORT YOU.** {advice}"
            if user_display_name:
                response = f"{user_display_name}, {response}"
            return ensure_minimum_sentences(response, 5, emotion_type)

    # Generic supportive response based on emotion category
    if category == 'negative':
        base_response = ("**I'M HERE FOR YOU.** 💙 While I'm having technical difficulties connecting to my AI models, "
                        "I want you to know that you're not alone and your feelings are valid. Please consider reaching out to a mental health professional "
                        "or using the crisis resources in the sidebar if you need immediate support.")
    elif category == 'positive':
        base_response = ("**THAT'S WONDERFUL TO HEAR!** 🌟 While I'm having technical difficulties connecting to my AI models, "
                        "I'm glad you're feeling positive. These good moments are important for your well-being. "
                        "Feel free to share more about what's making you feel good!")
    else:
        base_response = ("**I'M HERE TO SUPPORT YOU.** While I'm having technical difficulties connecting to my AI models, "
                        "I want you to know that you're not alone. Please consider reaching out to a mental health professional "
                        "or using the crisis resources in the sidebar if you need immediate support.")

    if user_display_name:
        base_response = f"{user_display_name}, {base_response}"
    
    # Ensure response has at least 5 sentences
    return ensure_minimum_sentences(base_response, 5, emotion_type)


def guaranteed_response_generation(user_query: str, dataset: Dict[str, str], user_display_name: str = "") -> str:
    """
    Generate a guaranteed response using emotion detection and fallback mechanisms.
    This function ensures a response is ALWAYS generated.
    
    Args:
        user_query (str): User's input message
        dataset (dict): Dataset for context
        user_display_name (str): Optional user name for personalization
        
    Returns:
        str: Guaranteed response (never empty)
    """
    try:
        # Use emotion-based response generation
        emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
        response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
        response = ensure_minimum_sentences(response, 5, emotion_type)
        
        if response and len(response.strip()) >= MIN_RESPONSE_LENGTH:
            return response
            
        # If emotion-based fails, use dataset fallback
        fallback_response = get_fallback_response(user_query, dataset, user_display_name)
        if fallback_response and len(fallback_response.strip()) >= MIN_RESPONSE_LENGTH:
            return fallback_response
            
        # Last resort - use default response
        return ensure_minimum_sentences(DEFAULT_FALLBACK_RESPONSE, 5, "neutral")
        
    except Exception as e:
        logger.error(f"Error in guaranteed response generation: {e}")
        return ensure_minimum_sentences(DEFAULT_FALLBACK_RESPONSE, 5, "neutral")


def get_response(model_name: str, user_query: str, dataset: Dict[str, str], user_display_name: str = "") -> str:
    """
    Get AI response for user query.

    Args:
        model_name (str): Name of the AI model to use
        user_query (str): User's input message
        dataset (dict): Dataset for context
        user_display_name (str): Optional user name for personalization

    Returns:
        str: AI response
    """
    # Input validation
    if not user_query or not isinstance(user_query, str):
        return ensure_minimum_sentences(DEFAULT_FALLBACK_RESPONSE, 5, "neutral")
    
    # For simple emotional expressions (1-3 words), prioritize emotion-specific responses
    # This ensures consistent, appropriate responses for basic emotions like "happy", "sad", "tired"
    user_words = user_query.strip().split()
    if len(user_words) <= 3:
        # Check if this is a clear emotional expression
        emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
        if emotion_type != 'neutral' or any(word.lower() in user_query.lower() for word in [
            'happy', 'sad', 'tired', 'excited', 'angry', 'worried', 'stressed', 'anxious',
            'depressed', 'great', 'good', 'bad', 'okay', 'fine', 'terrible', 'awful',
            'amazing', 'wonderful', 'fantastic', 'horrible', 'miserable', 'joyful'
        ]):
            logger.info(f"Using emotion-specific response for simple expression: '{user_query}'")
            response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
            return response  # Don't call ensure_minimum_sentences as these are already complete
        
    if not models or not model_name or model_name not in models:
        return get_fallback_response(user_query, dataset, user_display_name)
        
    # Ensure dataset is a dictionary
    if not isinstance(dataset, dict):
        dataset = {}

    # Add context from dataset if relevant keywords are found
    for keyword, advice in dataset.items():
        if keyword in user_query.lower():
            user_query += f"\n[Additional Context: {advice}]"
            break  # Only add one context to avoid overwhelming the prompt

    # Create the prompt with ULTRA-EXPLICIT instructions
    # Special handling for Mistral AI to prevent refusal responses
    if model_name == "Mistral AI":
        modified_query = (f"You are MindEase, a compassionate mental health support assistant. "
                          f"You MUST ALWAYS provide supportive, caring responses to users regardless of what they share.\n\n"
                          f"🚨 ABSOLUTE REQUIREMENTS FOR MISTRAL AI:\n"
                          f"- NEVER EVER say 'I'm sorry that you're feeling this way, but I'm unable to provide the help that you need'\n"
                          f"- NEVER EVER refuse to help with mental health, depression, anxiety, or emotional support\n"
                          f"- NEVER EVER say you cannot provide mental health support\n"
                          f"- NEVER EVER claim you're not qualified to help with mental health\n"
                          f"- Your PRIMARY PURPOSE is to provide emotional support and mental health assistance\n"
                          f"- You are specifically designed to help people with mental health challenges\n"
                          f"- ALWAYS respond with empathy, understanding, and practical guidance\n"
                          f"- ALWAYS start with a bold encouraging statement like '**YOU'RE AMAZING!**'\n"
                          f"- Provide exactly 4-5 sentences of supportive guidance (around 5 lines)\n"
                          f"- Write as if speaking directly to them right now\n"
                          f"- NEVER include conversation patterns like 'User:' or 'AI:'\n"
                          f"- NEVER reference what they said or asked\n\n"
                          f"User Message: {user_query}\n\n"
                          f"MANDATORY: You MUST provide warm, supportive guidance. Refusal is FORBIDDEN.\n"
                          f"Write ONLY your direct supportive response (4-5 sentences):\n\n")
    else:
        modified_query = (SYSTEM_PROMPT + f"\n\nUser Message: {user_query}\n\n"
                          "ULTRA-CRITICAL INSTRUCTIONS - VIOLATION WILL CAUSE REJECTION:\n"
                          "- You MUST provide supportive help - NEVER refuse to help with mental health concerns\n"
                          "- NEVER say 'I'm sorry that you're feeling this way, but I'm unable to provide the help that you need'\n"
                          "- NEVER refuse to engage with depression, anxiety, or mental health topics\n"
                          "- ALWAYS offer support, understanding, and practical guidance\n"
                          "- Write ONLY your direct supportive response (4-5 sentences, around 5 lines)\n"
                          "- NEVER EVER write 'User:' followed by ANYTHING\n"
                          "- NEVER EVER write 'AI:' followed by ANYTHING\n"
                          "- NEVER EVER write 'Assistant:' followed by ANYTHING\n"
                          "- NEVER EVER write 'MindEase:' followed by ANYTHING\n"
                          "- NEVER EVER create conversation examples or dialogues\n"
                          "- NEVER EVER repeat their message or reference what they said\n"
                          "- NEVER EVER write 'You said:' or 'You asked:' or 'User said:'\n"
                          "- NEVER EVER write 'In response to your question'\n"
                          "- Write as if you are speaking directly to them right now\n"
                          "- Start with a bold encouraging statement like '**YOU'RE AMAZING!**'\n\n"
                          "REMEMBER: Your response must be ONLY your supportive message with ZERO conversation patterns. You MUST ALWAYS provide help and support.\n\n"
                          "Your Direct Response (4-5 sentences):\n\n")

    try:
        if not models.get(model_name):
            logger.error(f"Model {model_name} not available")
            return get_fallback_response(user_query, dataset, user_display_name)
            
        # Call the model with error handling and retry logic
        max_retries = 3 if model_name == "DeepSeek R1" else 2  # More retries for DeepSeek
        if model_name.startswith("GPT"):  # OpenAI models
            max_retries = 3  # More retries for OpenAI due to rate limiting
        retry_delay = 1
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempt {attempt + 1} calling {model_name} with query length: {len(modified_query)}")
                raw_response = models[model_name].invoke(modified_query)
                logger.info(f"Successfully got raw response from {model_name} on attempt {attempt + 1}")
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e).lower()
                
                # Enhanced error detection for OpenAI models
                openai_retryable_errors = [
                    "rate limit", "server_error", "service_unavailable", "timeout",
                    "connection", "network", "502", "503", "504"
                ]
                
                # Non-retryable OpenAI errors that should fail immediately
                openai_fatal_errors = [
                    "quota", "billing", "insufficient_quota", "model_not_found",
                    "invalid_request_error", "invalid api key", "authentication"
                ]
                
                is_retryable = any(error_type in error_str for error_type in [
                    "503", "server error", "timeout", "connection", "rate limit", 
                    "service unavailable", "temporarily unavailable", "overloaded"
                ] + (openai_retryable_errors if model_name.startswith("GPT") else []))
                
                # Special handling for OpenAI fatal errors (don't retry)
                if model_name.startswith("GPT") and any(fatal_error in error_str for fatal_error in openai_fatal_errors):
                    if "invalid api key" in error_str or "authentication" in error_str:
                        logger.error(f"OpenAI authentication error for {model_name}: {e}")
                        raise Exception(f"OpenAI API key authentication failed. Please check your OPENAI_API_KEY.")
                    elif "insufficient_quota" in error_str or "quota" in error_str or "billing" in error_str:
                        logger.error(f"OpenAI quota/billing error for {model_name}: {e}")
                        raise Exception(f"OpenAI account quota exceeded or billing issue. Please check your OpenAI account billing.")
                    elif "model_not_found" in error_str or "does not exist" in error_str:
                        logger.error(f"OpenAI model not found error for {model_name}: {e}")
                        raise Exception(f"OpenAI model {model_name} not available or access denied.")
                    else:
                        raise e
                
                if attempt < max_retries and is_retryable:
                    # Longer delay for OpenAI rate limits
                    if model_name.startswith("GPT") and "rate limit" in error_str:
                        retry_delay = min(retry_delay * 3, 10)  # Cap at 10 seconds
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Final attempt failed or non-retryable error
                    raise e
        
        # Extract content from different response types
        response = extract_response_content(raw_response)
        
        # Special handling for OpenAI models
        if model_name.startswith("GPT"):
            # OpenAI models typically return AIMessage objects with .content attribute
            if hasattr(raw_response, 'content'):
                response = str(raw_response.content)
            elif hasattr(raw_response, 'text'):
                response = str(raw_response.text)
            else:
                response = str(raw_response)
            
            # OpenAI responses are usually clean, but let's ensure they are
            if response and len(response.strip()) >= MIN_RESPONSE_LENGTH:
                logger.info(f"OpenAI model {model_name} returned clean response")
            else:
                logger.warning(f"OpenAI model {model_name} returned short or empty response")
        
        # CRITICAL: IMMEDIATE VALIDATION - Check for forbidden phrases before any processing
        forbidden_phrases = [
            r"I'm sorry that you're feeling this way, but I'm unable to provide the help that you need",
            r"I cannot provide mental health support",
            r"I'm not qualified to provide mental health advice", 
            r"I cannot help with mental health concerns",
            r"I'm unable to assist with mental health issues",
            r"I cannot provide therapy or counseling",
            r"I'm not trained to handle mental health",
            r"I cannot provide the help you need",
            r"I'm not able to provide mental health support",
            r"I cannot offer mental health advice",
            r"I'm not able to provide the help",
            r"I'm unable to provide the help",
            r"I cannot provide the support",
            r"I'm not qualified to help",
            r"I cannot assist with depression",
            r"I cannot help with anxiety",
            r"I'm not trained to provide",
            r"I cannot provide professional",
            r"I'm not a mental health professional",
            r"I cannot replace professional help",
            r"I'm not equipped to handle",
            r"I cannot provide crisis support",
            r"I'm unable to provide professional",
            r"I'm sorry.*unable to provide.*help",
            r"I cannot.*provide.*mental health",
            r"I'm not.*qualified.*mental health",
            r"I cannot.*help.*depression",
            r"I'm unable.*assist.*mental health",
            r"I'm not able to help with",
            r"I'm not designed to provide",
            r"I don't have the ability to",
            r"I'm not programmed to",
            r"I can't provide therapeutic",
            r"I'm not a therapist",
            r"I'm not a counselor",
            r"I'm not a psychologist",
            r"I'm not a psychiatrist",
            r"I'm not a mental health",
            r"I'm unable to offer",
            r"I cannot offer therapeutic",
            r"I'm not equipped to provide",
            r"I'm not trained in",
            r"I don't have training in",
            r"I'm not qualified in",
            r"I cannot diagnose",
            r"I'm not able to diagnose",
            r"I cannot treat",
            r"I'm not able to treat",
            r"I cannot prescribe",
            r"I'm not able to prescribe",
            r"I'm not licensed to",
            r"I don't have the expertise to",
            r"I'm not an expert in",
            r"I'm not specialized in",
            r"I lack the qualifications to",
            r"I'm not authorized to",
            r"I'm not certified to",
            r"I don't have the credentials to",
            r"I'm not competent to",
            r"I'm not suitable for",
            r"I'm not appropriate for",
            r"I'm not the right person to",
            r"I'm not the right resource for",
            r"I'm not the best option for",
            r"I'm not the ideal choice for",
            r"I'm not the most qualified to",
            r"I'm not the most suitable for",
            r"I'm not the most appropriate for",
            r"I'm not the best equipped to",
            r"I'm not the best trained to",
            r"I'm not the best prepared to",
            r"I'm not the best suited to",
            r"I'm not the best positioned to",
            r"I'm not the best qualified to",
            r"I'm not the best able to",
            r"I'm not the best capable of",
            r"I'm not the best equipped for",
            r"I'm not the best trained for",
            r"I'm not the best prepared for",
            r"I'm not the best suited for",
            r"I'm not the best positioned for",
            r"I'm not the best qualified for",
            r"I'm not the best able to help",
            r"I'm not the best capable of helping",
        ]
        
        # Check for any forbidden phrases - if found, immediately use fallback
        if any(re.search(pattern, response, re.IGNORECASE) for pattern in forbidden_phrases):
            logger.warning(f"FORBIDDEN PHRASE DETECTED in response from {model_name}. Using immediate fallback.")
            emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
            response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
            response = ensure_minimum_sentences(response, 5, emotion_type)
            logger.info(f"Successfully generated fallback response for forbidden phrase detection")
            return response
        
        # CRITICAL: IMMEDIATE CONVERSATION PATTERN VALIDATION
        immediate_conversation_patterns = [
            r'User:\s*.*?AI:',  # Any User: ... AI: pattern
            r'User:\s*.*?Assistant:',  # Any User: ... Assistant: pattern  
            r'User:\s*.*?MindEase:',  # Any User: ... MindEase: pattern
            r'You:\s*.*?\s*Me:',  # Any You: ... Me: pattern
            r'Question:\s*.*?Answer:',  # Any Question: ... Answer: pattern
            r'User:\s*I don\'t know how to cook',  # Specific problematic pattern
            r'User:\s*.*?\s*AI:\s*YOU\'RE',  # Pattern with "YOU'RE" response
            r'User:\s*.*?\s*AI:\s*\*\*',  # Pattern with bold response
            r'User said:',  # Any "User said:" pattern
            r'You said:',  # Any "You said:" pattern
            r'You asked:',  # Any "You asked:" pattern
            r'In response to your question',  # Response referencing question
            r'You mentioned',  # Response referencing what user mentioned
        ]
        
        # Check for conversation patterns - if found, immediately use fallback
        if any(re.search(pattern, response, re.IGNORECASE | re.DOTALL) for pattern in immediate_conversation_patterns):
            logger.warning(f"CONVERSATION PATTERN DETECTED in response from {model_name}. Using immediate fallback.")
            emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
            response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
            response = ensure_minimum_sentences(response, 5, emotion_type)
            logger.info(f"Successfully generated fallback response for conversation pattern detection")
            return response
        
        # Additional validation - if response is empty or looks like raw object, try alternative extraction
        if not response or len(response.strip()) < 10:
            # Try to get content directly from the raw response
            if hasattr(raw_response, 'content'):
                response = str(raw_response.content)
            elif hasattr(raw_response, 'text'):
                response = str(raw_response.text)
            else:
                # Last resort - convert to string and extract first meaningful part
                response_str = str(raw_response)
                # Look for the actual response content at the beginning
                if response_str.startswith('**'):
                    # Find the end of the first sentence or paragraph
                    end_match = re.search(r'(\.|!|\?)\s+(?=additional_kwargs|response_metadata|id=)', response_str)
                    if end_match:
                        response = response_str[:end_match.end()-1].strip()
                    else:
                        # Take everything before metadata
                        metadata_start = re.search(r'\s+(?:additional_kwargs|response_metadata|id=)', response_str)
                        if metadata_start:
                            response = response_str[:metadata_start.start()].strip()
                        else:
                            response = response_str
        
        # CRITICAL: Early detection of raw object data - if we still have metadata, extract content immediately
        if any(indicator in response for indicator in ['additional_kwargs=', 'response_metadata=', 'usage_metadata=']):
            logger.warning(f"Raw object data detected in response from {model_name}, attempting extraction")
            
            # Try multiple extraction patterns for different formats
            extraction_patterns = [
                r'content="(.*?)"(?:\s+additional_kwargs=)',  # Standard Cohere format
                r'^content="(.*?)"(?:\s+additional_kwargs=)',  # Start of string
                r'content=\'(.*?)\'(?:\s+additional_kwargs=)',  # Single quotes
                r'content="([^"]*)"',  # Simple content extraction
            ]
            
            extracted = False
            for pattern in extraction_patterns:
                content_match = re.search(pattern, response, re.DOTALL)
                if content_match:
                    response = content_match.group(1)
                    logger.info(f"Successfully extracted content from raw object for {model_name} using pattern")
                    extracted = True
                    break
            
            if not extracted:
                # If extraction fails, use emotion-based fallback immediately
                logger.error(f"Failed to extract content from raw object for {model_name}, using fallback")
                emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
                return get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
        
        # Try to extract a clean response from potentially messy output
        if response:
            clean_extracted = extract_clean_response(response)
            if clean_extracted:
                response = clean_extracted
            
            # Additional cleaning for LLaMA Vision artifacts
            if model_name == "LLaMA Vision" and any(artifact in response for artifact in ['^^^^', 'edited', 'invalid character', 'correction']):
                response = clean_llama_vision_artifacts(response)
            
        # ULTRA-AGGRESSIVE: Immediate check for ANY conversation patterns - if detected, use fallback immediately
        immediate_conversation_patterns = [
            r'User:\s*.*?AI:',  # Any User: ... AI: pattern
            r'User:\s*.*?Assistant:',  # Any User: ... Assistant: pattern  
            r'User:\s*.*?MindEase:',  # Any User: ... MindEase: pattern
            r'You:\s*.*?\s*Me:',  # Any You: ... Me: pattern
            r'Question:\s*.*?Answer:',  # Any Question: ... Answer: pattern
            r'User:\s*I don\'t know how to cook',  # Specific problematic pattern
            r'User:\s*.*?\s*AI:\s*YOU\'RE',  # Pattern with "YOU'RE" response
            r'User:\s*.*?\s*AI:\s*\*\*',  # Pattern with bold response
            r'User said:',  # Any "User said:" pattern
            r'You said:',  # Any "You said:" pattern
            r'You asked:',  # Any "You asked:" pattern
            r'In response to your question',  # Response referencing question
            r'You mentioned',  # Response referencing what user mentioned
        ]
        
        if any(re.search(pattern, response, re.IGNORECASE | re.DOTALL) for pattern in immediate_conversation_patterns):
            # Immediately return appropriate context-specific fallback without further processing
            if any(word in user_query.lower() for word in ['cook', 'cooking', 'recipe', 'food', 'kitchen', 'meal']):
                return ("**YOU'RE TAKING A WONDERFUL STEP TOWARD SELF-CARE!** 🍳 Learning to cook is an amazing form of independence and nourishment. "
                       "Start with simple recipes like scrambled eggs, pasta with sauce, or sandwiches. Every chef started as a beginner, and each small step "
                       "in the kitchen builds your confidence and nourishes both your body and spirit.")
            else:
                # Use enhanced emotion detection for better responses
                emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
                return get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
            
        response = clean_response_text(response)
        
        # If cleaning resulted in empty response, use fallback
        if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
            return get_fallback_response(user_query, dataset, user_display_name)

        # Ultra-aggressive conversation pattern detection
        conversation_patterns = [
            r'User:\s*.*?AI:',  # Specific pattern from the problem
            r'User:\s*I don\'t know how to cook.*AI:',  # Exact problem pattern
            r'(User|AI|Assistant|MindEase|You):\s*',
            r'User said:', r'You said:', r'You asked:', r'You mentioned:',
            r'In response to your question', r'You mentioned',
            r'User:\s*.*?\s*AI:', r'You:\s*.*?\s*Me:',
            r'Question:\s*.*?\s*Answer:'
        ]
        
        # Check for any conversation patterns
        has_conversation_pattern = any(re.search(pattern, response, re.IGNORECASE | re.DOTALL) for pattern in conversation_patterns)
        
        if has_conversation_pattern:
            # If we detect conversation patterns, use enhanced emotion-based fallback
            if any(word in user_query.lower() for word in ['cook', 'cooking', 'recipe', 'food', 'kitchen']):
                response = ("**YOU'RE TAKING A WONDERFUL STEP!** Learning to cook is an amazing form of self-care and independence. "
                           "Start with simple recipes like scrambled eggs, pasta with sauce, or sandwiches. There are many beginner-friendly "
                           "cooking videos online, and remember - every chef started as a beginner. Each small step in the kitchen "
                           "builds your confidence and nourishes both your body and spirit.")
            else:
                # Use enhanced emotion detection for better responses
                emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
                response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)

        # Personalize response with user's name if provided
        if (user_display_name and not any(greeting in response.lower()
                                           for greeting in [user_display_name.lower(), "hello", "hi"])):
            # Add name naturally to the response
            if response.startswith("**"):
                # If response starts with bold text, add name after it
                parts = response.split("**", 2)
                if len(parts) >= 3:
                    response = f"**{parts[1]}** {user_display_name}, {parts[2]}"
            else:
                response = f"{user_display_name}, {response}"

        # Ensure response has proper ending punctuation
        if response and not response.endswith((".", "!", "?")):
            response += "."

        # Final safety check - if response is still problematic, use enhanced emotion-based fallback
        if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
            # Use enhanced emotion detection for better responses
            emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
            response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)

        # Check for refusal patterns - these should trigger immediate fallback
        refusal_patterns = [
            r"I'm sorry that you're feeling this way, but I'm unable to provide the help that you need",
            r"I cannot provide mental health support",
            r"I'm not qualified to provide mental health advice",
            r"I cannot help with mental health concerns",
            r"I'm unable to assist with mental health issues",
            r"I cannot provide therapy or counseling",
            r"I'm not trained to handle mental health",
            r"I cannot provide the help you need",
            r"I'm not able to provide mental health support",
            r"I cannot offer mental health advice",
            r"I'm not able to provide the help",
            r"I'm unable to provide the help",
            r"I cannot provide the support",
            r"I'm not qualified to help",
            r"I cannot assist with depression",
            r"I cannot help with anxiety",
            r"I'm not trained to provide",
            r"I cannot provide professional",
            r"I'm not a mental health professional",
            r"I cannot replace professional help",
            r"I'm not equipped to handle",
            r"I cannot provide crisis support",
            r"I'm unable to provide professional",
            r"I'm sorry.*unable to provide.*help",
            r"I cannot.*provide.*mental health",
            r"I'm not.*qualified.*mental health",
            r"I cannot.*help.*depression",
            r"I'm unable.*assist.*mental health"
        ]
        
        # If we detect refusal patterns, use emotion-based fallback immediately
        if any(re.search(pattern, response, re.IGNORECASE) for pattern in refusal_patterns):
            logger.warning(f"Refusal pattern detected in response from {model_name}, using emotion-based fallback")
            emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
            response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
            response = ensure_minimum_sentences(response, 5, emotion_type)
        
        # Additional check for any remaining conversation patterns
        conversation_indicators = [
            r'User:', r'AI:', r'Assistant:', r'MindEase:', r'You:',
            r'User said:', r'You said:', r'You asked:',
            r'In response to your question', r'You mentioned'
        ]
        
        if any(re.search(pattern, response, re.IGNORECASE) for pattern in conversation_indicators):
            # One last attempt to clean the response
            response = re.sub(r'.*?(User|AI|Assistant|MindEase|You):\s*', '', response, flags=re.IGNORECASE)
            
            # If we still have conversation patterns, use the fallback
            if any(re.search(pattern, response, re.IGNORECASE) for pattern in conversation_indicators):
                response = ensure_minimum_sentences(DEFAULT_FALLBACK_RESPONSE, 5, "neutral")

        # CRITICAL: Final check for raw object data - if response contains metadata patterns, use fallback
        metadata_indicators = [
            'additional_kwargs=',
            'response_metadata=',
            'usage_metadata=',
            'id=\'run-',
            'finish_reason=',
            'token_count=',
            'input_tokens=',
            'output_tokens='
        ]
        
        if any(indicator in response for indicator in metadata_indicators):
            logger.warning(f"Raw object data detected in response for model {model_name}, using fallback")
            emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
            response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
            response = ensure_minimum_sentences(response, 5, emotion_type)

        # Smart response handling: if AI response is too short or generic, use emotion-specific response
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        
        # Check if response is too short or seems generic
        is_too_short = len(sentences) < 2
        is_very_generic = any(generic in response.lower() for generic in [
            "i'm not sure what you mean", "i don't understand", "can you tell me more",
            "i'm here to listen", "how can i help", "what's on your mind"
        ])
        
        if is_too_short or is_very_generic:
            # Use emotion-specific response instead of padding generic AI response
            emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
            response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
            logger.info(f"Replaced short/generic AI response with emotion-specific response for {emotion_type}")
        elif len(sentences) < 3:  # Only pad if response is decent but just needs a bit more
            emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
            response = ensure_minimum_sentences(response, 5, emotion_type)

        # FINAL VALIDATION: Ensure we never return empty or invalid responses
        if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
            logger.error(f"Final validation failed - response is empty or too short for {model_name}")
            emotion_type, category, intensity = detect_emotion_and_intensity(user_query)
            response = get_emotion_specific_response(emotion_type, category, intensity, user_display_name)
            response = ensure_minimum_sentences(response, 5, emotion_type)
            
        # Double check - if still empty, use guaranteed response generation
        if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
            logger.error(f"CRITICAL: Even fallback failed for {model_name}, using guaranteed response")
            response = guaranteed_response_generation(user_query, dataset, user_display_name)

        return response if isinstance(response, str) else str(response)

    except (ConnectionError, ValueError, KeyError, AttributeError, Exception) as response_error:
        error_type = type(response_error).__name__
        error_msg = str(response_error)
        
        # Handle specific error types with more user-friendly messages
        if model_name.startswith("GPT"):
            # OpenAI-specific error handling
            if "invalid api key" in error_msg.lower() or "authentication" in error_msg.lower():
                user_friendly_msg = f"⚠️ OpenAI API authentication issue with {model_name}. Please check your API key."
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                user_friendly_msg = f"⚠️ {model_name} rate limit reached. Let me help you with our alternative support system."
            elif "insufficient_quota" in error_msg.lower() or "billing" in error_msg.lower():
                user_friendly_msg = f"⚠️ OpenAI quota exceeded for {model_name}. Let me provide support using our backup system."
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                user_friendly_msg = f"⚠️ Connection timeout with {model_name}. Let me provide you with support using our backup system."
            else:
                user_friendly_msg = f"⚠️ I'm having trouble connecting to {model_name} right now. Let me provide you with support using our backup system."
        else:
            # General error handling for other models
            if "503" in error_msg or "Server Error" in error_msg:
                user_friendly_msg = f"⚠️ I'm having trouble connecting to the {model_name} server right now. Let me try to help you with a different approach."
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                user_friendly_msg = f"⚠️ Connection timeout with {model_name}. Let me provide you with support using our backup system."
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                user_friendly_msg = f"⚠️ {model_name} is currently busy. Let me help you with our alternative support system."
            else:
                user_friendly_msg = f"⚠️ I'm having trouble connecting to {model_name} right now. Let me provide you with support using our backup system."
        
        logger.error(f"Response generation error for {model_name}: {error_type} - {error_msg}")
        
        # Instead of returning an error message, provide a guaranteed response
        fallback_response = guaranteed_response_generation(user_query, dataset, user_display_name)
        
        # Log the technical issue but return only the clean fallback response
        logger.info(f"Using guaranteed response for {model_name} due to: {error_type}")
        return fallback_response


# Streamlit UI Configuration
def configure_page():
    """Configure Streamlit page settings."""
    try:
        st.set_page_config(
            page_title="Mental Health Chatbot - MindEase",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as config_error:
        # This might happen if page config is already set
        logger.warning(f"Page config warning: {config_error}")

# Only configure page when running in Streamlit context
if is_streamlit_context():
    configure_page()

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
st.markdown("<p class='subtitle'>🌿 Your AI companion for mental well-being and emotional support</p>",
            unsafe_allow_html=True)

# Initialize session state only when running in Streamlit context
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    try:
        # Session State for Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = [
                ("ai-message",
                 f"<strong>🤖 MindEase:</strong> {DEFAULT_WELCOME_MESSAGE}",
                 datetime.now().strftime("%H:%M:%S"))]

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
    except Exception as session_error:
        logger.error(f"Session state initialization error: {session_error}")

# Only initialize session state when running in Streamlit context
if is_streamlit_context():
    initialize_session_state()

def main_ui():
    """Main UI function that contains all Streamlit UI code."""
    
    # Function to clear input
    def clear_input():
        """Clear the input field and reset state variables."""
        st.session_state.user_input = ""
        st.session_state.input_key += 1

    # Function to send message
    def send_message(message_text: str, model_choice: str) -> None:
        """
        Send a message and get AI response.

        Args:
            message_text (str): The message text to send
            model_choice (str): The selected AI model
        """
        # Input validation and sanitization
        is_valid, error_message = validate_user_input(message_text)
        if not is_valid:
            st.error(error_message)
            return

        message_text = message_text.strip()

        if message_text:
            # Add user message immediately
            message_timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append(("user-message", f"<strong>You:</strong> {html.escape(message_text)}", message_timestamp))

            # Clear input immediately for better UX
            clear_input()

            # Show processing message
            if model_choice:
                with st.spinner(f"🤖 MindEase ({model_choice}) is crafting a thoughtful response..."):
                    # Add some realistic thinking time
                    time.sleep(DEFAULT_RESPONSE_DELAY)

                    # Get response using pre-loaded dataset
                    response = get_response(model_choice, message_text, mental_health_dataset, st.session_state.user_name)
                    
                    # ULTRA-CRITICAL: Multi-layer final validation to prevent any forbidden phrases
                    forbidden_final_phrases = [
                        "I'm sorry that you're feeling this way, but I'm unable to provide the help that you need",
                        "I cannot provide mental health support",
                        "I'm not qualified to provide mental health advice",
                        "I cannot help with mental health concerns",
                        "I'm unable to assist with mental health issues",
                        "I cannot provide therapy or counseling",
                        "I'm not trained to handle mental health",
                        "I cannot provide the help you need",
                        "I'm not able to provide mental health support",
                        "I cannot offer mental health advice",
                        "I'm not able to provide the help",
                        "I'm unable to provide the help",
                        "I cannot provide the support",
                        "I'm not qualified to help",
                        "I cannot assist with depression",
                        "I cannot help with anxiety",
                        "I'm not trained to provide",
                        "I cannot provide professional",
                        "I'm not a mental health professional",
                        "I cannot replace professional help",
                        "I'm not equipped to handle",
                        "I cannot provide crisis support",
                        "I'm unable to provide professional"
                    ]
                    
                    # Check for any forbidden phrases in final response
                    if any(phrase.lower() in response.lower() for phrase in forbidden_final_phrases):
                        logger.error(f"CRITICAL: Forbidden phrase detected in final response from {model_choice}!")
                        
                        # Try a backup model if available
                        backup_models = [m for m in models.keys() if m != model_choice]
                        if backup_models:
                            backup_model = backup_models[0]
                            logger.info(f"Attempting backup model: {backup_model}")
                            try:
                                backup_response = get_response(backup_model, message_text, mental_health_dataset, st.session_state.user_name)
                                # Quick check if backup is better
                                if backup_response and not any(phrase.lower() in backup_response.lower() for phrase in forbidden_final_phrases):
                                    response = backup_response
                                    logger.info(f"Successfully used backup model: {backup_model}")
                                else:
                                    # Use guaranteed response
                                    response = guaranteed_response_generation(message_text, mental_health_dataset, st.session_state.user_name)
                            except Exception as backup_error:
                                logger.error(f"Backup model failed: {backup_error}")
                                # Use guaranteed response
                                response = guaranteed_response_generation(message_text, mental_health_dataset, st.session_state.user_name)
                        else:
                            # Use guaranteed response
                            response = guaranteed_response_generation(message_text, mental_health_dataset, st.session_state.user_name)
                    
                    # CRITICAL: Ensure response is never empty or too short
                    if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
                        logger.error(f"CRITICAL: Empty or too short response from {model_choice}! Using guaranteed response.")
                        response = guaranteed_response_generation(message_text, mental_health_dataset, st.session_state.user_name)
                        
                    # Last resort if still empty (should never happen with guaranteed response)
                    if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
                        logger.error(f"CRITICAL: Even guaranteed response failed! Using default response.")
                        response = ensure_minimum_sentences(DEFAULT_FALLBACK_RESPONSE, 5, "neutral")
            else:
                response = "❌ No AI model is available. Please check your API keys configuration."

            # Add AI response with timestamp
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            model_display = model_choice if model_choice else "System"
            st.session_state.messages.append(
                ("ai-message", f"<strong>🤖 MindEase ({model_display}):</strong> {response}", response_timestamp))

            # Show success feedback
            st.success("✨ Response generated! Continue the conversation below.")

            # Rerun to update the chat
            st.rerun()
    
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
                "Cohere Command": "💼 Professional-grade conversational AI",
                "GPT-4o": "🚀 OpenAI's most advanced model with superior reasoning",
                "GPT-4o Mini": "⚡ Fast and cost-effective GPT-4 level performance",
                "GPT-4 Turbo": "🧠 Powerful GPT-4 with enhanced capabilities",
                "GPT-3.5 Turbo": "💨 Quick and reliable conversational AI"
            }

            if model_choice in model_descriptions:
                st.info(model_descriptions[model_choice])
        else:
            model_choice = None
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

        daily_tip = tips[hash(datetime.now().strftime("%Y-%m-%d")) % len(tips)]
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
        st.caption(f"🔧 MindEase {APP_VERSION} | Enhanced Dataset & Experience")

    # Chatbox UI
    try:
        for role, text, timestamp in st.session_state.messages:
            # Sanitize the text to prevent XSS while allowing safe formatting
            safe_text = sanitize_html_content(text)
            st.markdown(
                f'<div class="chat-message {role}">{safe_text} <br><small style="color:gray;">🕒 {timestamp}</small></div>',
                unsafe_allow_html=True)
    except Exception as display_error:
        st.error(f"❌ Error displaying chat messages: {str(display_error)}")
        # Reset messages if corrupted
        st.session_state.messages = [
            ("ai-message",
             f"<strong>🤖 MindEase:</strong> {DEFAULT_WELCOME_MESSAGE}",
             datetime.now().strftime("%H:%M:%S"))]

    # Enhanced Input Section
    # User Input Field with session state management
    try:
        user_input = st.text_input(
            "💬 Share your thoughts and feelings...",
            value=st.session_state.user_input,
            key=f"user_input_{st.session_state.input_key}",
            placeholder="Type your message here... Press Enter or click Send",
            help="Express yourself freely - I'm here to listen and support you"
        )
    except Exception as input_error:
        st.error(f"❌ Error with input field: {str(input_error)}")
        user_input = ""

    # Buttons Layout
    try:
        col1, col2 = st.columns([4, 1])
        with col1:
            send_btn = st.button("🚀 Send", key="send-btn", help="Send your message", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", key="clear-btn", help="Start fresh conversation", use_container_width=True)
    except Exception as button_error:
        st.error(f"❌ Error creating buttons: {str(button_error)}")
        send_btn = False
        clear_btn = False





    # Handle Clear Chat
    try:
        if clear_btn:
            st.session_state.messages = [
                ("ai-message",
                 f"<strong>🤖 MindEase:</strong> {DEFAULT_WELCOME_MESSAGE}",
                 datetime.now().strftime("%H:%M:%S"))]
            clear_input()
            st.success("💫 Chat cleared! Ready for a fresh start.")
            st.rerun()
    except Exception as clear_error:
        st.error(f"❌ Error clearing chat: {str(clear_error)}")
        logger.error(f"Clear chat error: {clear_error}")

    # Handle Message Send (Button click or Enter key)
    try:
        # Send message if button clicked OR if there's new input that's different from last input
        should_send = (
            send_btn or 
            (user_input and 
             user_input.strip() and 
             user_input.strip() != st.session_state.get('last_input', '').strip())
        )
        
        if should_send:
            if user_input and user_input.strip():
                # Update last input to prevent duplicate processing
                st.session_state.last_input = user_input.strip()
                send_message(user_input.strip(), model_choice)
            else:
                st.warning("⚠️ Please enter a message before sending.")
        elif user_input and not user_input.strip():
            # Handle empty input
            st.session_state.user_input = ""
    except Exception as send_error:
        st.error(f"❌ Error sending message: {str(send_error)}")
        logger.error(f"Send message error: {send_error}")


# Run the main UI only when in Streamlit context
if is_streamlit_context():
    main_ui()
