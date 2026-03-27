# make sure to open the terminal in the same directory as this file to run it
# to install dependencies run the following commands in the terminal:
#   source ./.venv/bin/activate (for linux/bash) or ./.venv/Scripts/activate.ps1 (for windows) 
#   pip3 install -r requirements.txt or py -m pip install -r requirements.txt
from openai import OpenAI 
from dotenv import load_dotenv 
import json
import os
import re
import zipfile
from pathlib import Path

# We're using dotenv to load the env variables from the .env file
# you must first create a .env file in the same directory as this file and add your openrouter key like this:
# OPENROUTER_API_KEY="enter whatever your openrouter key is"
# This will prevent having to enter it everytime you open the terminal
load_dotenv()

# to test if openai works <type> py -c "import openai; print(openai.__file__)"
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",

    # if setting the env to your openrouter key doesn't work
    # then we can hard code the key in. The previous way is to avoid
    # any leaks of our api keys. <look down below for hardcode>

    # api_key="ENTER YOUR KEY HERE",
    # base_url="https://openrouter.ai/api/v1",
)

# if we want to do a larger context history collection for the chatbot

INFO_PROMPT = """
You are an Early Education Chatbot named Owlbear.
Use the conversation history when it is relevant to the client.
"""

def handle_conversation():
    context = ""
    print("Welcome to the AI Chatbot 'Owlbear'! Type 'bye', 'quit', 'exit' to close the chat.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "quit", "exit"]:
            break

        full_prompt = f"""
    {INFO_PROMPT}
        
    This is the conversation history
    {context}

    User Question: {user_input}

    Answer:
    """.strip()

        response = client.chat.completions.create(
            model = "stepfun/step-3.5-flash:free",
            messages = [{"role": "user", "content": full_prompt}
                        ]
        )

        result = response.choices[0].message.content.strip()
        print("Owlbear:", result)

        context += f"\nUser: {user_input}\nAI: {result}"


# if we want to keep the history of live messages implement this
# the code below place this above the handle_conversations function


# conversation = [
# {"role": "system", "content": "You are an Early Education Chatbot named Owlbear. Use the conversation history when it is relevant to the client."}
# ]

# then under the while True loop put the comment below:

# conversation.append({"role": "user", "content": user_input})

# replace the messages = ... 
# with -> messages = conversation

# lastly at the bottom before the if __name__ 

# conversation.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    response = handle_conversation()