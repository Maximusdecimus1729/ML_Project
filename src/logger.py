import logging
import os
from datetime import datetime

LOGF_File = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Use /tmp for logs in containerized environments (HuggingFace Spaces)
# Fall back to current directory for local development
if os.path.exists('/tmp') and os.access('/tmp', os.W_OK):
    logs_path = os.path.join('/tmp', 'logs')
else:
    logs_path = os.path.join(os.getcwd(), 'logs')

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOGF_File)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s %(levelname)s - %(message)s",
    level=logging.INFO,
)


   