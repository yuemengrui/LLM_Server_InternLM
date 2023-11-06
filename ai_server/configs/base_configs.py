# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

FASTAPI_TITLE = 'LLM_Server_InternLM'
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 5000

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

INTERNLM_CONFIG = {
    "model_name": "InternLM_20B_16k",
    "model_name_or_path": "",
    "device": "cuda"
}

# API LIMIT
API_LIMIT = {
    "chat": "15/minute",
    "token_count": "60/minute",
}
