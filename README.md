# Telegram to MT4 Signal Bridge

Professional automated trading signal copier from Telegram to MetaTrader 4.

**Platform:** Mac OS  
**App:** Telegram Bot / MT4 Bot  
**Version:** 2.0

### Library file path: 

/Users/mac/Library/Application Support/MetaTrader 5/Bottles/metatrader5/drive_c/Program Files/MetaTrader 5/MQL5/Experts


### Create Executable File for PC:
```bash
pyinstaller --onefile --name "TelegramMT5Bridge" telegramMT4Bridge.py
```
---

## Configuration Status

- **Python Project:** `/Users/mac/Documents/Python Projects/Telegram Bot/Telegram Bot/`
- **Signals File:** `/Users/mac/signals.json`
- **Telegram App:** Telegram Bot (MT4 Bot)

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install telethon cryptg
```

### Config Details (Secret)

```bash

CONFIG = {
    'api_id': '20334456',  # Get from https://my.telegram.org/apps
    'api_hash': '37a75a9a9de365f35baec07b3e7edd43',  # Get from https://my.telegram.org/apps
    'phone': '+256709165008',  # Your phone number with country code (e.g., +256712345678)
    'channel_username': '@pythontelegramscript',  # Channel to monitor (e.g., @ForexSignals)
    'signals_file': '/Users/mac/signals.json',  # Signals file location (Mac path)
    'session_name': 'telegram_mt4_session',  # Session file name
    'log_file': 'telegram_bridge.log'  # Log file name
}
```
