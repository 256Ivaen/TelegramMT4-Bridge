"""
Telegram to MT5 Signal Bridge
Professional-grade signal copier with error handling and logging
Configured for Mac OS with MetaTrader 5
Author: Trading Automation System
Version: 2.0
"""

import asyncio
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError

# ==================== CONFIGURATION ====================
CONFIG = {
    'api_id': '20334456',  # Get from https://my.telegram.org/apps
    'api_hash': '37a75a9a9de365f35baec07b3e7edd43',  # Get from https://my.telegram.org/apps
    'phone': '+256709165008',  # Your phone number with country code (e.g., +256712345678)
    'channel_username': '@pythontelegramscript',  # Channel to monitor (e.g., @ForexSignals)
    'signals_file': '/Users/mac/signals.json',  # Signals file location (Mac path)
    'session_name': 'telegram_mt4_session',  # Session file name
    'log_file': 'telegram_bridge.log'  # Log file name
}

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== DATA STRUCTURES ====================
@dataclass
class TradingSignal:
    """Structured trading signal"""
    symbol: str
    action: str  # BUY or SELL
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    lot_size: float = 0.01
    timestamp: str = None
    raw_message: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_valid(self) -> bool:
        """Validate signal completeness"""
        return all([
            self.symbol,
            self.action in ['BUY', 'SELL'],
            self.entry_price > 0
        ])

# ==================== SIGNAL PARSER ====================
class SignalParser:
    """Parse various Telegram signal formats"""

    # Common forex pairs and their variations
    SYMBOLS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD', 'EURCHF', 'AUDNZD',
        'XAUUSD', 'GOLD', 'XAGUSD', 'SILVER', 'BTCUSD', 'ETHUSD',
        'USDSGD', 'USDHKD', 'USDMXN', 'USDZAR', 'AUDCAD', 'AUDCHF', 'AUDJPY',
        'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP',
        'EURJPY', 'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
        'NZDCAD', 'NZDCHF', 'NZDJPY', 'USDTRY', 'EURTRY', 'GBPTRY'
    ]

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalize symbol names"""
        symbol = symbol.upper().replace('/', '').replace(' ', '').replace('-', '').replace('_', '')

        # Handle special cases
        if 'GOLD' in symbol or 'XAU' in symbol:
            return 'XAUUSD'
        if 'SILVER' in symbol or 'XAG' in symbol:
            return 'XAGUSD'
        if 'BITCOIN' in symbol or 'BTC' in symbol:
            return 'BTCUSD'
        if 'ETHEREUM' in symbol or 'ETH' in symbol:
            return 'ETHUSD'

        return symbol

    @staticmethod
    def extract_price(text: str, keywords: list) -> Optional[float]:
        """Extract price from text near specific keywords"""
        text_upper = text.upper()

        for keyword in keywords:
            if keyword in text_upper:
                # Look for numbers after keyword
                pattern = rf'{keyword}\s*:?\s*@?\s*([\d.]+)'
                match = re.search(pattern, text_upper)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue

        return None

    def parse(self, message: str) -> Optional[TradingSignal]:
        """
        Parse trading signal from message text
        Supports multiple formats:
        - BUY EURUSD @ 1.0850 SL: 1.0800 TP: 1.0900
        - SELL Gold Entry: 2050 Stop Loss: 2060 Take Profit: 2030
        - EURUSD BUY 1.0850 SL 1.0800 TP 1.0900
        - 📈 BUY GBPUSD at 1.2650 | SL 1.2600 | TP 1.2750
        """
        try:
            message_upper = message.upper()

            # Determine action
            action = None
            if 'BUY' in message_upper or '📈' in message or 'LONG' in message_upper:
                action = 'BUY'
            elif 'SELL' in message_upper or '📉' in message or 'SHORT' in message_upper:
                action = 'SELL'
            else:
                logger.debug("No trading action found in message")
                return None

            # Find symbol
            symbol = None
            for sym in self.SYMBOLS:
                if sym in message_upper:
                    symbol = self.normalize_symbol(sym)
                    break

            if not symbol:
                logger.warning("No valid symbol found in message")
                return None

            # Extract prices
            entry_price = self.extract_price(message, ['ENTRY', 'PRICE', 'AT', '@', 'ENTER', 'OPEN'])

            # If no explicit entry, look for first number after symbol or action
            if not entry_price:
                pattern = rf'(?:{symbol}|{action})\s+(?:BUY|SELL|AT|@)?\s*([\d.]+)'
                match = re.search(pattern, message_upper)
                if match:
                    try:
                        entry_price = float(match.group(1))
                    except:
                        pass

            if not entry_price:
                # Try to find any reasonable price number (between 0.0001 and 100000)
                numbers = re.findall(r'\b\d+\.\d+\b|\b\d{4,}\b', message)
                for num_str in numbers:
                    try:
                        num = float(num_str)
                        if 0.0001 < num < 100000:
                            entry_price = num
                            break
                    except:
                        continue

            stop_loss = self.extract_price(message, ['SL', 'STOP LOSS', 'STOPLOSS', 'STOP', 'S/L', 'S.L'])
            take_profit = self.extract_price(message, ['TP', 'TAKE PROFIT', 'TAKEPROFIT', 'TARGET', 'T/P', 'T.P', 'TP1'])

            # Extract lot size if specified
            lot_size = 0.01  # Default
            lot_match = re.search(r'LOT[S]?\s*:?\s*([\d.]+)', message_upper)
            if lot_match:
                try:
                    lot_size = float(lot_match.group(1))
                except:
                    lot_size = 0.01

            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size,
                raw_message=message
            )

            if signal.is_valid():
                logger.info(f"✅ Parsed signal: {signal.symbol} {signal.action} @ {signal.entry_price}")
                return signal
            else:
                logger.warning("Incomplete signal data")
                return None

        except Exception as e:
            logger.error(f"Error parsing signal: {e}", exc_info=True)
            return None

# ==================== SIGNAL MANAGER ====================
class SignalManager:
    """Manage signal storage and MT5 communication"""

    def __init__(self, signals_file: str):
        self.signals_file = Path(signals_file)
        self.ensure_file_exists()

    def ensure_file_exists(self):
        """Create signals file if it doesn't exist"""
        if not self.signals_file.exists():
            self.signals_file.write_text('[]')
            logger.info(f"Created signals file: {self.signals_file}")

    def save_signal(self, signal: TradingSignal):
        """Save signal to JSON file for MT5 to read"""
        try:
            # Read existing signals
            signals = self.read_signals()

            # Add new signal
            signal_dict = signal.to_dict()
            signal_dict['processed'] = False  # Mark as unprocessed
            signals.append(signal_dict)

            # Keep only last 100 signals
            signals = signals[-100:]

            # Write back
            self.signals_file.write_text(json.dumps(signals, indent=2))
            logger.info(f"💾 Signal saved to {self.signals_file}")

        except Exception as e:
            logger.error(f"Error saving signal: {e}", exc_info=True)

    def read_signals(self) -> list:
        """Read all signals from file"""
        try:
            content = self.signals_file.read_text()
            return json.loads(content) if content else []
        except Exception as e:
            logger.error(f"Error reading signals: {e}")
            return []

    def mark_signal_processed(self, timestamp: str):
        """Mark signal as processed by MT5"""
        try:
            signals = self.read_signals()
            for signal in signals:
                if signal.get('timestamp') == timestamp:
                    signal['processed'] = True

            self.signals_file.write_text(json.dumps(signals, indent=2))
        except Exception as e:
            logger.error(f"Error marking signal: {e}")

# ==================== TELEGRAM CLIENT ====================
class TelegramSignalListener:
    """Main Telegram client for listening to signals"""

    def __init__(self, config: dict):
        self.config = config
        self.client = TelegramClient(
            config['session_name'],
            config['api_id'],
            config['api_hash']
        )
        self.parser = SignalParser()
        self.signal_manager = SignalManager(config['signals_file'])

    async def start(self):
        """Start the Telegram client"""
        try:
            await self.client.start(phone=self.config['phone'])

            # Check if we need 2FA
            if not await self.client.is_user_authorized():
                try:
                    await self.client.send_code_request(self.config['phone'])
                    code = input('Enter the code you received: ')
                    await self.client.sign_in(self.config['phone'], code)
                except SessionPasswordNeededError:
                    password = input('Two-step verification enabled. Enter your password: ')
                    await self.client.sign_in(password=password)

            logger.info("=" * 60)
            logger.info("TELEGRAM CLIENT STARTED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"App: Telegram Bot / MT4 Bot")
            logger.info(f"Phone: {self.config['phone']}")
            logger.info(f"Listening to: {self.config['channel_username']}")
            logger.info(f"Signals file: {self.config['signals_file']}")
            logger.info(f"Log file: {self.config['log_file']}")
            logger.info("=" * 60)

            # Register event handler
            @self.client.on(events.NewMessage(chats=self.config['channel_username']))
            async def handler(event):
                await self.handle_new_message(event)

            # Keep running
            logger.info("Bridge is running... Press Ctrl+C to stop")
            logger.info("=" * 60 + "\n")
            await self.client.run_until_disconnected()

        except Exception as e:
            logger.error(f"Error starting Telegram client: {e}", exc_info=True)
            raise

    async def handle_new_message(self, event):
        """Handle new message from Telegram channel"""
        try:
            message_text = event.message.message
            logger.info(f"\n{'=' * 60}")
            logger.info(f"📨 NEW MESSAGE RECEIVED")
            logger.info(f"{'=' * 60}")
            logger.info(f"{message_text}")
            logger.info(f"{'=' * 60}\n")

            # Parse signal
            signal = self.parser.parse(message_text)

            if signal:
                logger.info(f"✅ VALID SIGNAL DETECTED:")
                logger.info(f"   Symbol: {signal.symbol}")
                logger.info(f"   Action: {signal.action}")
                logger.info(f"   Entry: {signal.entry_price}")
                logger.info(f"   Stop Loss: {signal.stop_loss if signal.stop_loss else 'None'}")
                logger.info(f"   Take Profit: {signal.take_profit if signal.take_profit else 'None'}")
                logger.info(f"   Lot Size: {signal.lot_size}")

                self.signal_manager.save_signal(signal)
                logger.info("🚀 Signal ready for MT5 execution\n")
            else:
                logger.debug("Message doesn't contain valid trading signal\n")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

# ==================== MAIN ====================
async def main():
    """Main entry point"""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("TELEGRAM TO MT5 SIGNAL BRIDGE")
        logger.info("=" * 60)
        logger.info("Version: 2.0")
        logger.info("Platform: Mac OS")
        logger.info("App: Telegram Bot / MT4 Bot")
        logger.info("Author: Ivan Odeke")
        logger.info("=" * 60 + "\n")

        # Validate configuration
        if CONFIG['api_id'] == 'YOUR_API_ID':
            logger.error("=" * 60)
            logger.error("CONFIGURATION ERROR")
            logger.error("=" * 60)
            logger.error("Please update the CONFIG section with your credentials:")
            logger.error("")
            logger.error("1. api_id     → Get from https://my.telegram.org/apps")
            logger.error("2. api_hash   → Get from https://my.telegram.org/apps")
            logger.error("3. phone      → Your phone number with country code")
            logger.error("4. channel    → Telegram channel to monitor (@channelname)")
            logger.error("")
            logger.error("Your Telegram app details:")
            logger.error("  App title: Telegram Bot")
            logger.error("  Short name: MT4 Bot")
            logger.error("=" * 60)
            return

        if CONFIG['channel_username'] == '@your_channel':
            logger.error("=" * 60)
            logger.error("ERROR: Please set your Telegram channel username!")
            logger.error("=" * 60)
            logger.error("Update 'channel_username' in CONFIG section")
            logger.error("Example: '@ForexSignals' or '@CryptoTrades'")
            logger.error("=" * 60)
            return

        # Start listener
        listener = TelegramSignalListener(CONFIG)
        await listener.start()

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("👋 Shutting down gracefully...")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == '__main__':
    """
    Entry point for the application
    Run this file to start the Telegram to MT5 bridge
    """
    try:
        # For Mac/Unix, set the event loop policy (if needed)
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Run the main application
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n Application stopped by user")
    except Exception as e:
        print(f"\n Application error: {e}")
        logging.error(f"Application error: {e}", exc_info=True)