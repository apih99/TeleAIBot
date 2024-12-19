import os
import logging
import signal
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from io import BytesIO
from PIL import Image
from collections import defaultdict
import base64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Initialize Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize models
text_model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

# Store chat histories and image contexts
chat_histories = defaultdict(list)
image_contexts = defaultdict(dict)  # Store the last image and its description
MAX_HISTORY = 15

# Global variables for cleanup
http_server = None
application = None

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
    
    def log_message(self, format, *args):
        # Suppress logging of health check requests
        pass

def run_health_server():
    global http_server
    port = int(os.getenv('PORT', 8080))
    http_server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    logger.info(f'Starting health check server on port {port}')
    http_server.serve_forever()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user_id = update.effective_user.id
    chat_histories[user_id] = []  # Clear history for this user
    image_contexts[user_id] = {}  # Clear image context
    
    welcome_message = """👋 Hello! I'm your AI assistant powered by Gemini. I can:
    
1. Answer your text messages (with memory of our conversation)
2. Analyze images you send (just send an image with or without a question)
3. Remember context from both text and images
4. Use /clear to clear our conversation history

Feel free to try either!"""
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
Here are the available commands:
/start - Start the bot
/help - Show this help message
/clear - Clear conversation history

You can:
1. Send any text message for a response (I'll remember our conversation)
2. Send an image (with optional text) for image analysis
3. Ask follow-up questions about the last image you sent
"""
    await update.message.reply_text(help_text)

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history for the user."""
    user_id = update.effective_user.id
    chat_histories[user_id] = []
    image_contexts[user_id] = {}
    await update.message.reply_text("Conversation history cleared! Let's start fresh.")

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
                response = vision_model.generate_content([
                    f"Previous context: {image_contexts[user_id].get('description', '')}\nNew question: {user_message}",
                    img
                ])
                await update.message.reply_text(response.text)
                # Store the interaction in chat history
                chat_histories[user_id].append({"role": "user", "parts": [user_message]})
                chat_histories[user_id].append({"role": "model", "parts": [response.text]})
                return

        # If no image context or not a follow-up, proceed with text chat
        chat_histories[user_id].append({"role": "user", "parts": [user_message]})
        chat = text_model.start_chat(history=chat_histories[user_id])
        response = chat.send_message(user_message)
        chat_histories[user_id].append({"role": "model", "parts": [response.text]})
        
        # Trim history if too long
        if len(chat_histories[user_id]) > MAX_HISTORY:
            chat_histories[user_id] = chat_histories[user_id][-MAX_HISTORY:]
        
        await update.message.reply_text(response.text)
    except Exception as e:
        error_message = f"Sorry, an error occurred: {str(e)}"
        await update.message.reply_text(error_message)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos with context."""
    try:
        user_id = update.effective_user.id
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download the photo
        response = requests.get(file.file_path)
        img = Image.open(BytesIO(response.content))
        
        # Get caption if any
        caption = update.message.caption if update.message.caption else "What's in this image?"
        
        # Store image context for future reference
        image_contexts[user_id] = {
            'image': img,
            'description': caption
        }
        
        # Generate response using vision model
        response = vision_model.generate_content([caption, img])
        
        # Store the interaction in chat history
        chat_histories[user_id].append({"role": "user", "parts": [f"[Sent an image with caption: {caption}]"]})
        chat_histories[user_id].append({"role": "model", "parts": [response.text]})
        
        await update.message.reply_text(response.text)
    except Exception as e:
        error_message = f"Sorry, an error occurred while processing the image: {str(e)}"
        await update.message.reply_text(error_message)

async def shutdown(signal_num):
    """Cleanup function to be called before shutdown."""
    logger.info(f"Received signal {signal_num}, initiating shutdown...")
    
    # Stop the HTTP server
    global http_server
    if http_server:
        logger.info("Shutting down HTTP server...")
        http_server.shutdown()
        http_server.server_close()
    
    # Stop the telegram bot
    global application
    if application:
        logger.info("Stopping telegram bot...")
        await application.stop()
        await application.shutdown()
    
    # Clear memory
    chat_histories.clear()
    image_contexts.clear()
    
    logger.info("Shutdown complete")

def signal_handler(sig, frame):
    """Handle shutdown signals."""
    asyncio.run(shutdown(sig))

def main():
    """Start the bot."""
    global application
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start health check server in a separate thread
        health_thread = threading.Thread(target=run_health_server, daemon=True)
        health_thread.start()

        # Create the Application and pass it your bot's token
        application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()

        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("clear", clear_history))
        
        # Add message handlers
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Start the Bot
        logger.info("Starting bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        asyncio.run(shutdown(0))

if __name__ == '__main__':
    main() 