# directory
from __future__ import annotations

# ticket lists
CRYPTO_1_TICKER = [
 'BTC-USD'
]


CRYPTO_10_TICKER = [
 'BTC-USD',
 'ETH-USD',
 'BNB-USD',
 'XRP-USD',
 'ADA-USD',
 'DOGE-USD',
 'SOL-USD',
 'LTC-USD',
 'TRX-USD',
 'ETC-USD'
]

CRYPTO_1b_TICKER = [
 'BTCUSDT'
]

CRYPTO_10b_TICKER = [
 'BTCUSDT',
 'ETHUSDT',
 'BNBUSDT',
 'XRPUSDT',
 'ADAUSDT',
 'DOGEUSDT',
 'SOLUSDT',
 'LTCUSDT',
 'TRXUSDT',
 'ETCUSDT'
]

# data directory
DATA_SAVE_DIR = "../../datasets"
TRAINED_MODEL_DIR = "../../trained_models"
TENSORBOARD_LOG_DIR = "../../tensorboard_log"
RESULTS_DIR = "../../results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = '2021-01-01'
TRAIN_END_DATE = '2022-07-01'
VALID_1_START_DATE = '2022-07-01'
VALID_1_END_DATE = '2022-10-01'
VALID_2_START_DATE = '2022-10-01'
VALID_2_END_DATE = '2023-01-01'
TEST_START_DATE = '2023-01-01'
TEST_END_DATE = '2023-04-01'

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    'atr',
    "rsi",
    "close_7_sma",
    "close_30_ema",
    "close_365_ema",
    "log-ret"
]

# Model Parameters
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
TD3_PARAMS = {
    "batch_size": 100, 
    "buffer_size": 1000000, 
    "learning_rate": 0.001
}
SAC_PARAMS = {
    "batch_size": 256,
    "buffer_size": 1000000,
    "learning_rate": 0.001,
    "learning_starts": 100
}

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
