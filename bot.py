import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from io import BytesIO
from PIL import Image
from collections import defaultdict
import base64
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Available models (Actual Model ID: User-facing Name)
# Ensure these model IDs are valid and available via the Gemini API
AVAILABLE_MODELS = {
    'gemini-2.5-flash-preview-05-20': 'Gemini 1.5 Flash',
    'gemini-2.5-flash-preview-05-20': 'Gemini Pro (Text/Vision)',
    # Add other models like 'gemini-ultra' if needed and available
}

# Define default models using the actual IDs
DEFAULT_TEXT_MODEL = 'gemini-2.5-flash-preview-05-20'
DEFAULT_VISION_MODEL = 'gemini-2.5-flash-preview-05-20' # Gemini Pro generally handles vision well

# Store user preferences using the actual model IDs
user_preferences = defaultdict(lambda: {'text_model': DEFAULT_TEXT_MODEL, 'vision_model': DEFAULT_VISION_MODEL})

# Store chat histories and image contexts
chat_histories = defaultdict(list)
image_contexts = defaultdict(dict)
MAX_HISTORY = 15

def get_model_for_user(user_id: int, model_type: str = 'text'):
    """Get the appropriate model for a user based on their preferences."""
    prefs = user_preferences[user_id]
    model_id = prefs['text_model'] if model_type == 'text' else prefs['vision_model']
    
    # Return the generative model instance based on the selected ID
    try:
        return genai.GenerativeModel(model_id)
    except Exception as e:
        logging.error(f"Failed to initialize model {model_id} for user {user_id}: {e}")
        # Fallback to default models if the selected one fails
        fallback_model_id = DEFAULT_TEXT_MODEL if model_type == 'text' else DEFAULT_VISION_MODEL
        logging.info(f"Falling back to model: {fallback_model_id}")
        return genai.GenerativeModel(fallback_model_id)

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /model command to change AI models."""
    keyboard = []
    for model_id, model_name in AVAILABLE_MODELS.items():
        keyboard.append([InlineKeyboardButton(model_name, callback_data=f"model_{model_id}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Choose your preferred model:\n"
        "(Note: Some models are coming soon)",
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle model selection button presses."""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("model_"):
        model_id = query.data.replace("model_", "")
        user_id = update.effective_user.id
        
        if model_id in AVAILABLE_MODELS:
            # Update both text and vision model preferences for simplicity
            # Assumes selected models are generally capable of both, adjust if needed
            user_preferences[user_id]['text_model'] = model_id
            user_preferences[user_id]['vision_model'] = model_id
            
            model_name = AVAILABLE_MODELS[model_id]
            await query.edit_message_text(
                f"Model set to: {model_name}\n"
                f"Current models in use:\n"
                f"Text: {AVAILABLE_MODELS[user_preferences[user_id]['text_model']]}\n"
                f"Vision: {AVAILABLE_MODELS[user_preferences[user_id]['vision_model']]}"
            )
        else:
            await query.edit_message_text("Invalid model selected.")

def format_markdown_text(text: str) -> str:
    """Convert markdown-style formatting to Telegram-compatible markdown."""
    # Replace '**text**' with '*text*' for bold
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    
    # Replace bullet points for better readability
    text = re.sub(r'^\* ', '• ', text, flags=re.MULTILINE)
    
    # Add extra newline before bullet points for better spacing
    text = re.sub(r'\n\*', '\n\n•', text)
    
    return text

async def send_long_message(update: Update, text: str):
    """Split and send long messages with proper formatting."""
    # Format the text for Telegram markdown
    text = format_markdown_text(text)
    
    # Telegram's message limit is 4096 characters
    MAX_LENGTH = 4000  # Leaving some margin for safety
    
    # Split by newlines first to keep formatting
    paragraphs = text.split('\n')
    current_message = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed limit, send current message
        if len(current_message) + len(paragraph) + 1 > MAX_LENGTH:
            if current_message:
                try:
                    await update.message.reply_text(current_message, parse_mode=ParseMode.MARKDOWN)
                except Exception as e:
                    # If markdown parsing fails, send without formatting
                    await update.message.reply_text(current_message)
                current_message = ""
            
            # If single paragraph is too long, split it by spaces
            if len(paragraph) > MAX_LENGTH:
                words = paragraph.split(' ')
                for word in words:
                    if len(current_message) + len(word) + 1 > MAX_LENGTH:
                        try:
                            await update.message.reply_text(current_message, parse_mode=ParseMode.MARKDOWN)
                        except Exception as e:
                            await update.message.reply_text(current_message)
                        current_message = word + ' '
                    else:
                        current_message += word + ' '
            else:
                current_message = paragraph
        else:
            if current_message:
                current_message += '\n' + paragraph
            else:
                current_message = paragraph
    
    # Send any remaining text
    if current_message:
        try:
            await update.message.reply_text(current_message, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            await update.message.reply_text(current_message)

# Simple HTTP handler for health checks
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Bot is running")
        else:
            self.send_response(404)
            self.end_headers()

def run_health_server():
    port = int(os.getenv('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    logging.info(f'Starting health check server on port {port}')
    server.serve_forever()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user_id = update.effective_user.id
    chat_histories[user_id] = []
    image_contexts[user_id] = {}
    
    welcome_message = """👋 Hello! I'm your AI assistant powered by Gemini. I can:
    
1. Answer your text messages (with memory of our conversation)
2. Analyze images you send (just send an image with or without a question)
3. Remember context from both text and images
4. Use /model to choose your preferred AI model
5. Use /clear to clear our conversation history
6. Handle long responses by splitting them automatically

Feel free to try any of these features!"""
    await send_long_message(update, welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
Here are the available commands:
/start - Start the bot
/help - Show this help message
/clear - Clear conversation history
/model - Choose your preferred AI model

You can:
1. Send any text message for a response (I'll remember our conversation)
2. Send an image (with optional text) for image analysis
3. Ask follow-up questions about the last image you sent
4. Get long detailed responses (they'll be split automatically)
"""
    await send_long_message(update, help_text)

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history for the user."""
    user_id = update.effective_user.id
    chat_histories[user_id] = []
    image_contexts[user_id] = {}
    await send_long_message(update, "Conversation history cleared! Let's start fresh.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages with context."""
    try:
        user_id = update.effective_user.id
        user_message = update.message.text
        
        # Check if there's a recent image context and the message seems like a follow-up
        if image_contexts.get(user_id) and not any(word in user_message.lower() for word in ["hello", "hi", "start"]):
            # Use vision model for follow-up about the image
            img = image_contexts[user_id].get('image')
            if img:
                model = get_model_for_user(user_id, 'vision')
                response = model.generate_content([
                    f"Previous context: {image_contexts[user_id].get('description', '')}\nNew question: {user_message}",
                    img
                ])
                await send_long_message(update, response.text)
                chat_histories[user_id].append({"role": "user", "parts": [user_message]})
                chat_histories[user_id].append({"role": "model", "parts": [response.text]})
                return

        # If no image context or not a follow-up, proceed with text chat
        model = get_model_for_user(user_id, 'text')
        chat_histories[user_id].append({"role": "user", "parts": [user_message]})
        chat = model.start_chat(history=chat_histories[user_id])
        response = chat.send_message(user_message)
        chat_histories[user_id].append({"role": "model", "parts": [response.text]})
        
        if len(chat_histories[user_id]) > MAX_HISTORY:
            chat_histories[user_id] = chat_histories[user_id][-MAX_HISTORY:]
        
        await send_long_message(update, response.text)
    except Exception as e:
        error_message = f"Sorry, an error occurred: {str(e)}"
        await send_long_message(update, error_message)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos with context."""
    try:
        user_id = update.effective_user.id
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        response = requests.get(file.file_path)
        img = Image.open(BytesIO(response.content))
        
        caption = update.message.caption if update.message.caption else "What's in this image?"
        
        image_contexts[user_id] = {
            'image': img,
            'description': caption
        }
        
        model = get_model_for_user(user_id, 'vision')
        response = model.generate_content([caption, img])
        
        chat_histories[user_id].append({"role": "user", "parts": [f"[Sent an image with caption: {caption}]"]})
        chat_histories[user_id].append({"role": "model", "parts": [response.text]})
        
        await send_long_message(update, response.text)
    except Exception as e:
        error_message = f"Sorry, an error occurred while processing the image: {str(e)}"
        await send_long_message(update, error_message)

def main():
    """Start the bot."""
    # Start health check server in a separate thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    # Create the Application and pass it your bot's token
    application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("model", model_command))
    
    # Add callback handler for model selection buttons
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Add message handlers
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()