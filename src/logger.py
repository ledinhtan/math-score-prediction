import os
import sys
import logging
from datetime import datetime

# Create logs directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate timestamped log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure root logger with basicConfig
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(module)s:%(lineno)d [%(name)s] - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Add console handler for better visibility during development
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(module)s:%(lineno)d [%(name)s] - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(console_handler)