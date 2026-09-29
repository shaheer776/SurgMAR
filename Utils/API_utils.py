import httpx
import os
import base64
import argparse
import json
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import re
from tqdm import tqdm
import logging

# Suppress gRPC and absl-py warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ============================================================
# Gemini Vision for Image Captioning
# ============================================================
def gemini_vision_caption(image_path, prompt):
    GEMINI_API_KEY = ''
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    response = model.generate_content([
        {
            'mime_type': 'image/jpeg',
            'data': image_data
        },
        prompt
    ])

    return response.text

# ============================================================
# GPT-4 Vision for Image Captioning
# ============================================================
def gpt4_vision_caption(image_path, prompt):
    OPENAI_API_KEY = ''
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    base64_image = encode_image(image_path)
    user_content = [
        {
            "type": "text",
            "text": prompt,
        },  
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            },
        }
    ]
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model="gpt-4o-latest",
        messages=messages
    )

    image_caption = response.choices[0].message.content

    return image_caption

# ============================================================
# GPT-4 API for TEXT Input
# ============================================================
def call_gpt4o_api(prompt):
    OPENAI_API_KEY = ''
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    # Add text-only prompt
    user_content = {
        "role": "user",
        "content": prompt
    }
    messages.append(user_content)
    # Send request to GPT-4 API
    response = client.chat.completions.create(
        model="gpt-4o-latest",
        messages=messages
    )
    # Extract the text response
    text_response = response.choices[0].message.content
    return text_response

# ============================================================
# GPT-3.5 Turbo API for TEXT Input 
# ============================================================
def call_gpt35Turbo_api(prompt):
    OPENAI_API_KEY = ''
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    # Add text-only prompt
    user_content = {
        "role": "user",
        "content": prompt
    }
    messages.append(user_content)
    # Send request to GPT-4 API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # Extract the text response
    text_response = response.choices[0].message.content
    return text_response

# ============================================================
# GPT-4o mini API for TEXT Input
# ============================================================
def call_gpt4omini_api(prompt):
    OPENAI_API_KEY = ''
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    # Add text-only prompt
    user_content = {
        "role": "user",
        "content": prompt
    }
    messages.append(user_content)
    # Send request to GPT-4 API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    # Extract the text response
    text_response = response.choices[0].message.content
    return text_response
