# Telegram Bot with Gemini AI

A Telegram bot that uses Google's Gemini AI to respond to messages.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the environment variables:
   - Copy the `.env` file and fill in your API keys:
     - `TELEGRAM_BOT_TOKEN`: Your Telegram Bot Token from @BotFather
     - `GEMINI_API_KEY`: Your Google Gemini AI API key

3. Run the bot:
```bash
python bot.py
```

## Usage

1. Start a chat with your bot on Telegram
2. Use the following commands:
   - `/start` - Start the bot
   - `/help` - Show help message
3. Send any message to the bot, and it will respond using Gemini AI

## Features

- Natural language conversation using Gemini AI
- Simple and easy-to-use interface
- Secure API key storage using environment variables
- Error handling and logging 