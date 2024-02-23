from openai import OpenAI
import instructor
from pydantic import BaseModel, ValidationError
import argparse
import os
import random
import time
import csv

# Constants
BATCH_SIZE = 50
CATEGORIES = ["Disrespect", "Dishonesty", "Negativity", "Hostility"]
LOCAL_API_URL = 'http://gruntus:11434/v1'
LOCAL_API_KEY = 'ollama'
LOCAL_MODEL = 'mistral:7b'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"

class MessagePair(BaseModel):
    respectful: str
    nondisrespectful: str

class CategorisedMessages(BaseModel):
    messages: list[MessagePair]

def generate_messages(client, category, num_messages, dest):
    total_generated = 0
    while total_generated < num_messages:
        prompt = f"Generate {BATCH_SIZE} pairs of short instant messages, where each pair contains a non-disrespectful (respectful or neutral) message and a corresponding disrespectful message exemplifying {category.lower()}."
        response = client.chat.completions.create(
            model=LOCAL_MODEL if USING_LOCAL else OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            response_model=CategorisedMessages
        )
        message_pairs = response.messages
        write_to_csv(dest, message_pairs, category, total_generated < num_messages)
        total_generated += len(message_pairs)

def write_to_csv(dest, message_pairs, category, append=False):
    mode = 'a' if append else 'w'
    with open(dest, mode, newline='', encoding='utf-8') as csvfile:
        fieldnames = ['class', 'respectful', 'nondisrespectful']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for pair in message_pairs:
            writer.writerow({'class': category, 'respectful': pair.respectful, 'nondisrespectful': pair.nondisrespectful})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chat messages with specified negative qualities.")
    parser.add_argument("--num", type=int, default=20, help="Number of message pairs to generate")
    parser.add_argument("--openai", action='store_true', help="Use OpenAI instead of local model")
    parser.add_argument("--dest", type=str, default="messages.csv", help="Destination CSV file for the messages")
    args = parser.parse_args()

    USING_LOCAL = not args.openai

    client = instructor.patch(OpenAI(
        base_url=LOCAL_API_URL,
        api_key=LOCAL_API_KEY,
    ), mode=instructor.Mode.JSON) if USING_LOCAL else instructor.patch(OpenAI(), mode=instructor.Mode.JSON)

    category = random.choice(CATEGORIES)
    generate_messages(client, category, args.num, args.dest)
    print(f"Messages written to {args.dest}")
