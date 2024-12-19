import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize models
text_model = genai.GenerativeModel('gemini-2.0-flash-exp')
vision_model = genai.GenerativeModel('gemini-pro-vision')

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
    welcome_message = """👋 Hello! I'm your AI assistant powered by Gemini. I can:
    
1. Answer your text messages
2. Analyze images you send (just send an image with or without a question)

Feel free to try either!"""
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
Here are the available commands:
/start - Start the bot
/help - Show this help message

You can:
1. Send any text message for a response
2. Send an image (with optional text) for image analysis
"""
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    try:
        user_message = update.message.text
        response = text_model.generate_content(user_message)
        await update.message.reply_text(response.text)
    except Exception as e:
        error_message = f"Sorry, an error occurred: {str(e)}"
        await update.message.reply_text(error_message)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos."""
    try:
        # Get the photo file
        photo = update.message.photo[-1]  # Get the largest photo size
        file = await context.bot.get_file(photo.file_id)
        
        # Download the photo
        response = requests.get(file.file_path)
        img = Image.open(BytesIO(response.content))
        
        # Get caption if any
        caption = update.message.caption if update.message.caption else "What's in this image?"
        
        # Generate response using vision model
        response = vision_model.generate_content([caption, img])
        await update.message.reply_text(response.text)
    except Exception as e:
        error_message = f"Sorry, an error occurred while processing the image: {str(e)}"
        await update.message.reply_text(error_message)

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
    
    # Add message handlers
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 