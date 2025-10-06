"""
Telegram to MT5 Signal Bridge
Professional-grade signal copier with error handling, logging, and OCR support
Configured for Mac OS with MetaTrader 5
Author: Trading Automation System
Version: 2.3 - Enhanced Connection Validation
"""

import asyncio
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError, ChannelInvalidError, ChannelPrivateError
from telethon.tl.types import Channel, Chat
from PIL import Image
import pytesseract
from io import BytesIO

# ==================== CONFIGURATION ====================
CONFIG = {
    'api_id': '20334456',
    'api_hash': '37a75a9a9de365f35baec07b3e7edd43',
    'phone': '+256709165008',
    'channel_username': '@pythontelegramscript',
    'signals_file': '/Users/mac/signals.json',
    'session_name': 'telegram_mt4_session',
    'log_file': 'telegram_bridge.log',
    'ocr_enabled': True,
    'ocr_language': 'eng',
    'save_images': False,
    'images_folder': '/Users/mac/signal_images',
    'require_explicit_signal': True,
    'ignore_analysis_only': True
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
    symbol: str
    action: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    lot_size: float = 0.01
    timestamp: str = None
    raw_message: str = ""
    source_type: str = "text"
    confidence_score: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_valid(self) -> bool:
        return all([
            self.symbol,
            self.action in ['BUY', 'SELL'],
            self.entry_price > 0,
            self.confidence_score >= 0.7
        ])

# ==================== CONNECTION VALIDATOR ====================
class ConnectionValidator:

    @staticmethod
    async def validate_telegram_connection(client: TelegramClient) -> Tuple[bool, str]:
        try:
            me = await client.get_me()
            if not me:
                return False, "Failed to retrieve user information"
            logger.info(f"[VALIDATION] Connected as: {me.first_name} (ID: {me.id})")
            return True, f"Authenticated as {me.first_name}"
        except Exception as e:
            return False, f"Connection validation failed: {str(e)}"

    @staticmethod
    async def validate_channel_access(client: TelegramClient, channel_username: str) -> Tuple[bool, str, Optional[Any]]:
        try:
            entity = await client.get_entity(channel_username)

            if isinstance(entity, Channel):
                logger.info(f"[VALIDATION] Channel found: {entity.title}")
                logger.info(f"[VALIDATION] Channel ID: {entity.id}")
                try:
                    messages = await client.get_messages(entity, limit=1)
                    logger.info(f"[VALIDATION] Channel access confirmed - Can read messages")
                    return True, f"Access confirmed to '{entity.title}'", entity
                except Exception as msg_error:
                    return False, f"Cannot read messages from channel: {str(msg_error)}", None
            elif isinstance(entity, Chat):
                logger.info(f"[VALIDATION] Group chat found: {entity.title}")
                return True, f"Access confirmed to group '{entity.title}'", entity
            else:
                return False, f"Unknown entity type: {type(entity)}", None
        except ChannelInvalidError:
            return False, f"Channel '{channel_username}' does not exist or is invalid", None
        except ChannelPrivateError:
            return False, f"Channel '{channel_username}' is private and you don't have access", None
        except ValueError as e:
            if "No user has" in str(e) or "Cannot find any entity" in str(e):
                return False, f"Channel '{channel_username}' not found. Check the username format (@channelname)", None
            return False, f"Channel validation error: {str(e)}", None
        except Exception as e:
            return False, f"Unexpected error validating channel: {str(e)}", None

    @staticmethod
    async def validate_signals_file(signals_file: str) -> Tuple[bool, str]:
        try:
            file_path = Path(signals_file)
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"[VALIDATION] Created directory: {file_path.parent}")
            if not file_path.exists():
                file_path.write_text('[]')
                logger.info(f"[VALIDATION] Created signals file: {file_path}")
            test_data = file_path.read_text()
            json.loads(test_data) if test_data.strip() else []
            file_path.write_text(json.dumps([{"test": "validation"}], indent=2))
            file_path.write_text('[]')
            logger.info(f"[VALIDATION] Signals file is readable and writable")
            return True, "Signals file validated successfully"
        except PermissionError:
            return False, f"Permission denied: Cannot write to {signals_file}"
        except Exception as e:
            return False, f"Signals file validation failed: {str(e)}"

# ==================== CONTEXT ANALYZER ====================
class ContextAnalyzer:

    NON_SIGNAL_PHRASES = [
        'watching', 'watch', 'looking at', 'monitoring', 'observing',
        'waiting for', 'if it breaks', 'if price breaks', 'if we break',
        'holding', 'still holding', 'managing', 'waiting',
        'analysis', 'idea', 'setup', 'potential', 'possible',
        'could', 'might', 'maybe', 'probably', 'thinking',
        'educational', 'for education', 'not financial advice',
        'opinion', 'view', 'perspective', 'outlook',
        'reentry', 're-entry', 'add more', 'adding positions',
        'next setup', 'future setup', 'upcoming',
        'risk management', 'proper risk', 'trade smart',
        "don't gamble", 'be careful', 'caution'
    ]

    SIGNAL_CONFIRMATION_PHRASES = [
        'buy now', 'sell now', 'entry now', 'enter now',
        'take this trade', 'execute', 'open position',
        'buy signal', 'sell signal', 'trade signal',
        'buy at', 'sell at', 'entry at', 'enter at',
        'buy @', 'sell @', 'entry @', 'enter @',
        'buy order', 'sell order', 'market order',
        'limit order', 'stop order'
    ]

    CHART_INDICATORS = [
        'support', 'resistance', 'trendline', 'trend line',
        'breakout', 'breakdown', 'zone', 'level',
        'fibonacci', 'fib', 'moving average', 'ma', 'ema',
        'rsi', 'macd', 'bollinger', 'pivot',
        'chart', 'analysis', 'technical'
    ]

    @staticmethod
    def is_analysis_only(text: str) -> bool:
        text_lower = text.lower()
        non_signal_count = sum(1 for phrase in ContextAnalyzer.NON_SIGNAL_PHRASES if phrase in text_lower)
        signal_confirmation_count = sum(1 for phrase in ContextAnalyzer.SIGNAL_CONFIRMATION_PHRASES if phrase in text_lower)
        chart_indicator_count = sum(1 for phrase in ContextAnalyzer.CHART_INDICATORS if phrase in text_lower)
        if signal_confirmation_count > 0:
            return False
        if non_signal_count >= 2 or chart_indicator_count >= 2:
            return True
        if 'watching' in text_lower or 'holding' in text_lower:
            return True
        return False

    @staticmethod
    def calculate_signal_confidence(text: str, has_entry: bool, has_symbol: bool, has_action: bool) -> float:
        confidence = 0.0
        text_lower = text.lower()
        if has_symbol:
            confidence += 0.3
        if has_action:
            confidence += 0.2
        if has_entry:
            confidence += 0.3
        for phrase in ContextAnalyzer.SIGNAL_CONFIRMATION_PHRASES:
            if phrase in text_lower:
                confidence += 0.2
                break
        for phrase in ContextAnalyzer.NON_SIGNAL_PHRASES:
            if phrase in text_lower:
                confidence -= 0.15
        if ContextAnalyzer.is_analysis_only(text):
            confidence -= 0.4
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def extract_chart_prices(text: str) -> List[float]:
        prices = []
        price_patterns = [
            r'\b(\d{4}\.\d{2})\b',
            r'\b(\d{4},\d{2})\b',
            r'\b(\d+\.\d{2,})\b'
        ]
        for pattern in price_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    price_str = match.group(1).replace(',', '.')
                    price = float(price_str)
                    if 0.0001 < price < 1000000:
                        prices.append(price)
                except:
                    continue
        return list(set(prices))

# ==================== IMAGE PROCESSOR ====================
class ImageProcessor:

    def __init__(self, config: dict):
        self.config = config
        self.images_folder = Path(config.get('images_folder', '/Users/mac/signal_images'))
        if config.get('save_images', False):
            self.images_folder.mkdir(parents=True, exist_ok=True)

    async def extract_text_from_image(self, image_bytes: bytes, save_name: str = None) -> str:
        try:
            image = Image.open(BytesIO(image_bytes))
            logger.info(f"Processing image: Size={image.size}, Mode={image.mode}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.config.get('save_images', False) and save_name:
                save_path = self.images_folder / f"{save_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                image.save(save_path)
                logger.info(f"Image saved to: {save_path}")
            custom_config = r'--oem 3 --psm 6'
            ocr_language = self.config.get('ocr_language', 'eng')
            text = pytesseract.image_to_string(image, lang=ocr_language, config=custom_config)
            text = text.strip()
            if text:
                logger.info(f"[SUCCESS] OCR extracted {len(text)} characters from image")
                logger.debug(f"OCR Text: {text[:200]}...")
            else:
                logger.warning("[WARNING] OCR extracted empty text from image")
            return text
        except Exception as e:
            logger.error(f"[ERROR] OCR processing error: {e}", exc_info=True)
            return ""

    async def process_image_message(self, photo_bytes: bytes) -> str:
        try:
            extracted_text = await self.extract_text_from_image(photo_bytes, save_name="signal")
            return extracted_text
        except Exception as e:
            logger.error(f"Error processing image message: {e}", exc_info=True)
            return ""

# ==================== SIGNAL PARSER ====================
class SignalParser:

    SYMBOLS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD', 'EURCHF', 'AUDNZD',
        'XAUUSD', 'GOLD', 'XAGUSD', 'SILVER', 'BTCUSD', 'ETHUSD', 'ETHUSDT', 'BTCUSDT',
        'USDSGD', 'USDHKD', 'USDMXN', 'USDZAR', 'AUDCAD', 'AUDCHF', 'AUDJPY',
        'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP',
        'EURJPY', 'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
        'NZDCAD', 'NZDCHF', 'NZDJPY', 'USDTRY', 'EURTRY', 'GBPTRY',
        'US30', 'NAS100', 'SPX500', 'US500', 'UK100', 'GER30', 'GER40', 'FRA40', 'JPN225',
        'USOIL', 'UKOIL', 'BRENT', 'WTI', 'XTIUSD', 'XBRUSD',
        'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
        'LTCUSDT', 'LINKUSDT', 'AVAXUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT',
        'EURPLN', 'EURHUF', 'EURCZK', 'USDSEK', 'USDNOK', 'USDDKK',
        'EURZAR', 'GBPZAR', 'USDINR', 'USDCNH', 'USDTHB', 'USDKRW',
        'ZARJPY', 'SGDJPY', 'HKDJPY', 'TRYJPY', 'MXNJPY', 'ZARUSD'
    ]

    def __init__(self, config: dict):
        self.config = config
        self.context_analyzer = ContextAnalyzer()

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        symbol = symbol.upper().replace('/', '').replace(' ', '').replace('-', '').replace('_', '').replace('.', '')
        symbol_map = {
            'GOLD': 'XAUUSD', 'XAU': 'XAUUSD', 'SILVER': 'XAGUSD', 'XAG': 'XAGUSD',
            'BITCOIN': 'BTCUSD', 'BTC': 'BTCUSD', 'ETHEREUM': 'ETHUSD', 'ETH': 'ETHUSD',
            'DOW': 'US30', 'DJIA': 'US30', 'NASDAQ': 'NAS100', 'NDX': 'NAS100',
            'S&P': 'SPX500', 'SPX': 'SPX500', 'SP500': 'SPX500', 'US500': 'SPX500',
            'DAX': 'GER40', 'FTSE': 'UK100', 'CAC': 'FRA40', 'NIKKEI': 'JPN225',
            'CRUDE': 'USOIL', 'WTI': 'USOIL', 'XTI': 'USOIL', 'BRENT': 'UKOIL', 'XBR': 'UKOIL',
            'BNB': 'BNBUSDT', 'ADA': 'ADAUSDT', 'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT',
            'DOGE': 'DOGEUSDT', 'DOT': 'DOTUSDT', 'MATIC': 'MATICUSDT', 'LTC': 'LTCUSDT',
            'LINK': 'LINKUSDT', 'AVAX': 'AVAXUSDT', 'UNI': 'UNIUSDT', 'ATOM': 'ATOMUSDT', 'ETC': 'ETCUSDT'
        }
        for key, value in symbol_map.items():
            if key in symbol:
                return value
        return symbol

    @staticmethod
    def extract_price(text: str, keywords: list) -> Optional[float]:
        text_upper = text.upper()
        text_clean = re.sub(r'[^\w\s.:@,]', ' ', text_upper)
        for keyword in keywords:
            if keyword in text_clean:
                patterns = [
                    rf'{keyword}\s*:?\s*@?\s*([\d]+\.[\d]+)',
                    rf'{keyword}\s*:?\s*@?\s*([\d]+,[\d]+)',
                    rf'{keyword}\s*:?\s*@?\s*([\d]+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, text_clean)
                    if match:
                        try:
                            price_str = match.group(1).replace(',', '.')
                            price = float(price_str)
                            if 0.0001 < price < 1000000:
                                return price
                        except ValueError:
                            continue
        return None

    @staticmethod
    def extract_multiple_targets(text: str) -> List[float]:
        targets = []
        tp_patterns = [
            r'TP\s*(\d+)\s*:?\s*@?\s*([\d.,]+)',
            r'TARGET\s*(\d+)\s*:?\s*@?\s*([\d.,]+)',
            r'TAKE\s*PROFIT\s*(\d+)\s*:?\s*@?\s*([\d.,]+)',
            r'T/P\s*(\d+)\s*:?\s*@?\s*([\d.,]+)'
        ]
        for pattern in tp_patterns:
            matches = re.finditer(pattern, text.upper())
            for match in matches:
                try:
                    price_str = match.group(2).replace(',', '.')
                    price = float(price_str)
                    if 0.0001 < price < 1000000:
                        targets.append(price)
                except:
                    continue
        return sorted(set(targets))

    def parse(self, message: str, ocr_text: str = "", source_type: str = "text") -> Optional[TradingSignal]:
        try:
            if not message or len(message.strip()) < 3:
                return None
            combined_text = f"{message}\n{ocr_text}".strip() if ocr_text else message
            if self.config.get('ignore_analysis_only', True):
                if self.context_analyzer.is_analysis_only(message):
                    logger.info("[ANALYSIS ONLY] Message detected as analysis/context, not a trading signal")
                    logger.debug(f"Message content: {message[:100]}...")
                    return None
            message_upper = combined_text.upper()
            action = None
            action_patterns = {
                'BUY': ['BUY NOW', 'BUY', 'LONG', 'CALL', 'BULLISH', 'COMPRA', 'ACHETER', 'KAUFEN', 'UP'],
                'SELL': ['SELL NOW', 'SELL', 'SHORT', 'PUT', 'BEARISH', 'VENTA', 'VENDRE', 'VERKAUFEN', 'DOWN']
            }
            for act, patterns in action_patterns.items():
                for pattern in patterns:
                    if re.search(rf'\b{pattern}\b', message_upper):
                        action = act
                        break
                if action:
                    break
            if not action:
                logger.debug("No trading action found in message")
                return None
            symbol = None
            for sym in self.SYMBOLS:
                if re.search(rf'\b{sym}\b', message_upper):
                    symbol = self.normalize_symbol(sym)
                    break
                elif sym in message_upper:
                    symbol = self.normalize_symbol(sym)
                    break
            if not symbol and ocr_text:
                ocr_prices = self.context_analyzer.extract_chart_prices(ocr_text)
                if ocr_prices:
                    logger.info(f"[CHART] Detected prices from chart: {ocr_prices}")
            if not symbol:
                logger.warning("No valid symbol found in message")
                return None
            entry_keywords = ['ENTRY', 'PRICE', 'AT', '@', 'ENTER', 'OPEN', 'BUY', 'SELL', 'ENTRAR', 'PRIX', 'PREIS']
            entry_price = self.extract_price(combined_text, entry_keywords)
            if not entry_price:
                pattern = rf'(?:{symbol}|{action})\s+(?:BUY|SELL|AT|@)?\s*([\d.,]+)'
                match = re.search(pattern, message_upper)
                if match:
                    try:
                        price_str = match.group(1).replace(',', '.')
                        entry_price = float(price_str)
                    except:
                        pass
            if not entry_price:
                numbers = re.findall(r'\b\d+\.\d+\b|\b\d+,\d+\b|\b\d{4,}\b', combined_text)
                for num_str in numbers:
                    try:
                        num = float(num_str.replace(',', '.'))
                        if 0.0001 < num < 1000000:
                            entry_price = num
                            break
                    except:
                        continue
            if not entry_price or entry_price <= 0:
                logger.warning("No valid entry price found")
                return None
            sl_keywords = ['SL', 'STOP LOSS', 'STOPLOSS', 'STOP', 'S/L', 'S.L', 'S L', 'STOPP']
            stop_loss = self.extract_price(combined_text, sl_keywords)
            tp_keywords = ['TP', 'TAKE PROFIT', 'TAKEPROFIT', 'TARGET', 'T/P', 'T.P', 'TP1', 'TP 1', 'PROFIT', 'ZIEL']
            take_profit = self.extract_price(combined_text, tp_keywords)
            multiple_targets = self.extract_multiple_targets(combined_text)
            if multiple_targets and not take_profit:
                take_profit = multiple_targets[0]
            lot_size = 0.01
            lot_patterns = [
                r'LOT[S]?\s*:?\s*([\d.]+)',
                r'SIZE\s*:?\s*([\d.]+)',
                r'VOLUME\s*:?\s*([\d.]+)',
                r'QTY\s*:?\s*([\d.]+)',
                r'QUANTITY\s*:?\s*([\d.]+)'
            ]
            for pattern in lot_patterns:
                lot_match = re.search(pattern, message_upper)
                if lot_match:
                    try:
                        lot_size = float(lot_match.group(1))
                        lot_size = max(0.01, min(lot_size, 100.0))
                        break
                    except:
                        lot_size = 0.01
            confidence = self.context_analyzer.calculate_signal_confidence(
                message,
                has_entry=bool(entry_price),
                has_symbol=bool(symbol),
                has_action=bool(action)
            )
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size,
                raw_message=message,
                source_type=source_type,
                confidence_score=confidence
            )
            if signal.is_valid():
                logger.info(f"[PARSED] {signal.symbol} {signal.action} @ {signal.entry_price} [Confidence: {confidence:.2f}] [Source: {source_type}]")
                return signal
            else:
                logger.warning(f"Signal validation failed [Confidence: {confidence:.2f}]")
                return None
        except Exception as e:
            logger.error(f"Error parsing signal: {e}", exc_info=True)
            return None

# ==================== SIGNAL MANAGER ====================
class SignalManager:

    def __init__(self, signals_file: str):
        self.signals_file = Path(signals_file)
        self.ensure_file_exists()

    def ensure_file_exists(self):
        try:
            self.signals_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.signals_file.exists():
                self.signals_file.write_text('[]')
                logger.info(f"Created signals file: {self.signals_file}")
        except Exception as e:
            logger.error(f"Error creating signals file: {e}", exc_info=True)

    def save_signal(self, signal: TradingSignal):
        try:
            signals = self.read_signals()
            signal_dict = signal.to_dict()
            signal_dict['processed'] = False
            signal_dict['saved_at'] = datetime.now().isoformat()
            signals.append(signal_dict)
            signals = signals[-100:]
            self.signals_file.write_text(json.dumps(signals, indent=2))
            logger.info(f"[SAVED] Signal saved to {self.signals_file}")
        except Exception as e:
            logger.error(f"Error saving signal: {e}", exc_info=True)

    def read_signals(self) -> list:
        try:
            if not self.signals_file.exists():
                return []
            content = self.signals_file.read_text()
            return json.loads(content) if content.strip() else []
        except Exception as e:
            logger.error(f"Error reading signals: {e}")
            return []

    def mark_signal_processed(self, timestamp: str):
        try:
            signals = self.read_signals()
            for signal in signals:
                if signal.get('timestamp') == timestamp:
                    signal['processed'] = True
                    signal['processed_at'] = datetime.now().isoformat()
            self.signals_file.write_text(json.dumps(signals, indent=2))
        except Exception as e:
            logger.error(f"Error marking signal: {e}")

    def get_unprocessed_signals(self) -> List[Dict]:
        try:
            signals = self.read_signals()
            return [s for s in signals if not s.get('processed', False)]
        except Exception as e:
            logger.error(f"Error getting unprocessed signals: {e}")
            return []

# ==================== TELEGRAM CLIENT ====================
class TelegramSignalListener:

    def __init__(self, config: dict):
        self.config = config
        self.client = TelegramClient(
            config['session_name'],
            config['api_id'],
            config['api_hash']
        )
        self.parser = SignalParser(config)
        self.signal_manager = SignalManager(config['signals_file'])
        self.image_processor = ImageProcessor(config) if config.get('ocr_enabled', True) else None
        self.validator = ConnectionValidator()
        self.is_validated = False

    async def start(self):
        try:
            logger.info("[CONNECTING] Initiating Telegram connection...")
            await self.client.start(phone=self.config['phone'])
            if not await self.client.is_user_authorized():
                try:
                    await self.client.send_code_request(self.config['phone'])
                    code = input('Enter the code you received: ')
                    await self.client.sign_in(self.config['phone'], code)
                except SessionPasswordNeededError:
                    password = input('Two-step verification enabled. Enter your password: ')
                    await self.client.sign_in(password=password)
            logger.info("\n" + "=" * 60)
            logger.info("[VALIDATION] Running connection validation...")
            logger.info("=" * 60)
            conn_valid, conn_msg = await self.validator.validate_telegram_connection(self.client)
            if not conn_valid:
                logger.error(f"[FAILED] {conn_msg}")
                logger.error("=" * 60)
                return
            logger.info(f"[PASSED] {conn_msg}")
            channel_valid, channel_msg, entity = await self.validator.validate_channel_access(
                self.client,
                self.config['channel_username']
            )
            if not channel_valid:
                logger.error(f"[FAILED] {channel_msg}")
                logger.error("=" * 60)
                logger.error("\nPossible solutions:")
                logger.error("1. Verify channel username is correct (must include @)")
                logger.error("2. Join the channel first if it's private")
                logger.error("3. Check if you have permission to read messages")
                logger.error("=" * 60)
                return
            logger.info(f"[PASSED] {channel_msg}")
            file_valid, file_msg = await self.validator.validate_signals_file(self.config['signals_file'])
            if not file_valid:
                logger.error(f"[FAILED] {file_msg}")
                logger.error("=" * 60)
                return
            logger.info(f"[PASSED] {file_msg}")
            logger.info("=" * 60)
            logger.info("[SUCCESS] All validations passed")
            logger.info("=" * 60)
            self.is_validated = True
            logger.info("\n" + "=" * 60)
            logger.info("TELEGRAM CLIENT STARTED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"App: Telegram Bot / MT4 Bot")
            logger.info(f"Phone: {self.config['phone']}")
            logger.info(f"Listening to: {self.config['channel_username']}")
            logger.info(f"Channel: {entity.title if entity else 'N/A'}")
            logger.info(f"Signals file: {self.config['signals_file']}")
            logger.info(f"Log file: {self.config['log_file']}")
            logger.info(f"OCR Enabled: {self.config.get('ocr_enabled', True)}")
            logger.info(f"Context Detection: {self.config.get('ignore_analysis_only', True)}")
            logger.info("=" * 60)

            @self.client.on(events.NewMessage(chats=self.config['channel_username']))
            async def handler(event):
                await self.handle_new_message(event)

            logger.info("[ACTIVE] Bridge is running... Press Ctrl+C to stop")
            logger.info("=" * 60 + "\n")
            await self.client.run_until_disconnected()

        except Exception as e:
            logger.error(f"[ERROR] Error starting Telegram client: {e}", exc_info=True)
            raise

    async def handle_new_message(self, event):
        try:
            if not self.is_validated:
                logger.warning("[WARNING] Skipping message - client not validated")
                return

            message_text = event.message.message or ""
            source_type = "text"
            ocr_text = ""

            logger.info(f"\n{'=' * 60}")
            logger.info(f"[NEW MESSAGE] Received")
            logger.info(f"{'=' * 60}")

            if event.message.photo and self.image_processor and self.config.get('ocr_enabled', True):
                logger.info("[IMAGE] Detected - Processing with OCR...")
                try:
                    photo_bytes = await event.message.download_media(file=bytes)
                    ocr_text = await self.image_processor.process_image_message(photo_bytes)
                    if ocr_text:
                        logger.info(f"[OCR] Text Extracted:\n{ocr_text}\n")
                        source_type = "image" if not message_text else "mixed"
                    else:
                        logger.warning("[WARNING] OCR extracted no text from image")
                except Exception as e:
                    logger.error(f"[ERROR] Error processing image: {e}", exc_info=True)

            if message_text or ocr_text:
                logger.info(f"Message Text:\n{message_text}")
                if ocr_text:
                    logger.info(f"Chart/Image Text:\n{ocr_text}")
                logger.info(f"{'=' * 60}\n")
            else:
                logger.info("Empty message - skipping")
                logger.info(f"{'=' * 60}\n")
                return

            signal = self.parser.parse(message_text, ocr_text=ocr_text, source_type=source_type)

            if signal:
                logger.info(f"[SIGNAL DETECTED]")
                logger.info(f"   Symbol: {signal.symbol}")
                logger.info(f"   Action: {signal.action}")
                logger.info(f"   Entry: {signal.entry_price}")
                logger.info(f"   Stop Loss: {signal.stop_loss if signal.stop_loss else 'None'}")
                logger.info(f"   Take Profit: {signal.take_profit if signal.take_profit else 'None'}")
                logger.info(f"   Lot Size: {signal.lot_size}")
                logger.info(f"   Source: {signal.source_type}")
                logger.info(f"   Confidence: {signal.confidence_score:.2f}")
                self.signal_manager.save_signal(signal)
                logger.info("[READY] Signal ready for MT5 execution\n")
            else:
                logger.debug("Message does not contain valid trading signal\n")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)


# ==================== MAIN ====================
async def main():
    try:
        logger.info("\n" + "=" * 60)
        logger.info("TELEGRAM TO MT5 SIGNAL BRIDGE")
        logger.info("=" * 60)
        logger.info("Version: 2.3 - Enhanced Connection Validation")
        logger.info("Platform: Mac OS")
        logger.info("App: Telegram Bot / MT4 Bot")
        logger.info("Author: Trading Automation System")
        logger.info("=" * 60 + "\n")

        if CONFIG['api_id'] == 'YOUR_API_ID':
            logger.error("=" * 60)
            logger.error("CONFIGURATION ERROR")
            logger.error("=" * 60)
            logger.error("Please update the CONFIG section with your credentials:")
            logger.error("")
            logger.error("1. api_id     -> Get from https://my.telegram.org/apps")
            logger.error("2. api_hash   -> Get from https://my.telegram.org/apps")
            logger.error("3. phone      -> Your phone number with country code")
            logger.error("4. channel    -> Telegram channel to monitor (@channelname)")
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

        if CONFIG.get('ocr_enabled', True):
            try:
                import pytesseract
                from PIL import Image
                logger.info("[SUCCESS] OCR dependencies loaded successfully")
            except ImportError as e:
                logger.error("=" * 60)
                logger.error("OCR DEPENDENCY ERROR")
                logger.error("=" * 60)
                logger.error("Please install required packages:")
                logger.error("  pip install pillow pytesseract")
                logger.error("  brew install tesseract")
                logger.error("=" * 60)
                logger.info("Continuing without OCR support...")
                CONFIG['ocr_enabled'] = False

        listener = TelegramSignalListener(CONFIG)
        await listener.start()

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("[SHUTDOWN] Application stopped by user")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


# ==================== ENTRY POINT ====================
if __name__ == '__main__':
    try:
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Application stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Application error: {e}")
        logging.error(f"Application error: {e}", exc_info=True)