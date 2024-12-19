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
import sys

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
text_model = genai.GenerativeModel('gemini-2.0-flash-exp')
vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Store chat histories and image contexts
chat_histories = defaultdict(list)
image_contexts = defaultdict(dict)  # Store the last image and its description
MAX_HISTORY = 15

# Global flag for graceful shutdown
is_shutting_down = False
health_server = None

async def send_long_message(update: Update, text: str):
    """Split and send long messages."""
    # Telegram's message limit is 4096 characters
    MAX_LENGTH = 4000  # Leaving some margin for safety
    
    # Split by newlines first to keep formatting
    paragraphs = text.split('\n')
    current_message = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed limit, send current message
        if len(current_message) + len(paragraph) + 1 > MAX_LENGTH:
            if current_message:
                await update.message.reply_text(current_message)
                current_message = ""
            
            # If single paragraph is too long, split it by spaces
            if len(paragraph) > MAX_LENGTH:
                words = paragraph.split(' ')
                for word in words:
                    if len(current_message) + len(word) + 1 > MAX_LENGTH:
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
        await update.message.reply_text(current_message)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global is_shutting_down
    logger.info(f"Received signal {signum}. Starting graceful shutdown...")
    is_shutting_down = True
    
    if health_server:
        logger.info("Shutting down health check server...")
        health_server.shutdown()
    
    # Let the main loop handle the actual shutdown
    raise SystemExit(0)

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
        """Suppress default logging of requests."""
        pass

def run_health_server():
    global health_server
    try:
        port = int(os.getenv('PORT', 8080))
        health_server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
        logger.info(f'Starting health check server on port {port}')
        health_server.serve_forever()
    except Exception as e:
        logger.error(f"Health server error: {e}")
        if not is_shutting_down:  # Only restart if not shutting down
            logger.info("Attempting to restart health server...")
            run_health_server()

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
5. Handle long responses by splitting them automatically

Feel free to try either!"""
    await send_long_message(update, welcome_message)

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
                response = vision_model.generate_content([
                    f"Previous context: {image_contexts[user_id].get('description', '')}\nNew question: {user_message}",
                    img
                ])
                await send_long_message(update, response.text)
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
        
        await send_long_message(update, response.text)
    except Exception as e:
        error_message = f"Sorry, an error occurred while processing the image: {str(e)}"
        await send_long_message(update, error_message)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the telegram-python-bot framework."""
    logger.error(f"Exception while handling an update: {context.error}")
    
    if update and hasattr(update, 'effective_message'):
        error_message = "An error occurred while processing your request. I'll try to recover..."
        try:
            await update.effective_message.reply_text(error_message)
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")

async def main() -> None:
    """Start the bot with error handling and recovery."""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start health check server in a separate thread
        health_thread = threading.Thread(target=run_health_server, daemon=True)
        health_thread.start()

        # Initialize bot with recovery options
        application = (
            Application.builder()
            .token(os.getenv('TELEGRAM_BOT_TOKEN'))
            .read_timeout(30)
            .write_timeout(30)
            .connect_timeout(30)
            .pool_timeout(30)
            .build()
        )

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("clear", clear_history))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Add error handler
        application.add_error_handler(error_handler)

        # Start the bot with graceful shutdown
        await application.initialize()
        await application.start()
        await application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        if not is_shutting_down:
            logger.info("Attempting to restart the bot...")
            await asyncio.sleep(5)  # Wait before retrying
            await main()  # Recursive restart
    finally:
        if is_shutting_down:
            logger.info("Shutting down application...")
            if application:
                await application.stop()
                await application.shutdown()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user/system request")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1) 