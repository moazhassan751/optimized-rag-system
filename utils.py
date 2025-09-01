# utils.py - Enhanced Utilities for Universal RAG

import time
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sklearn.metrics import f1_score

def calculate_f1(ground_truth: list, predictions: list) -> float:
    return f1_score(ground_truth, predictions, average='macro')

async def load_document_async(file_path: str):
    loop = asyncio.get_running_loop()
    loader = PyPDFLoader(file_path) if file_path.lower().endswith('.pdf') else TextLoader(file_path)
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, loader.load) 
