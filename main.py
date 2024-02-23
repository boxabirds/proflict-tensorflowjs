from openai import OpenAI
import instructor
from pydantic import BaseModel, ValidationError
import argparse
import os
import random
import time


USING_LOCAL = True
CATEGORIES = ["Disrespect", "Dishonesty", "Negativity", "Hostility"]
LOCAL_API_URL = 'http://gruntus:11434/v1'
LOCAL_API_KEY = 'ollama'
#LOCAL_MODEL = 'calebfahlgren/natural-functions:latest'
LOCAL_MODEL = 'mistral:7b'

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"

if USING_LOCAL:
    client = instructor.patch(OpenAI(
        base_url=LOCAL_API_URL,
        api_key=LOCAL_API_KEY,
    ), mode=instructor.Mode.JSON)
else:
    client = instructor.patch(OpenAI(), mode=instructor.Mode.JSON)

class CategorisedMessages(BaseModel):
    messages: list[str]

def generate_messages(category, num_messages):
    prompt = f"Generate {num_messages} short instant messages exemplifying {category.lower()}, each separated by a newline."
    start_time = time.time()  # Start timing
    try:
        response = client.chat.completions.create(
            model=LOCAL_MODEL if USING_LOCAL else OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            response_model=CategorisedMessages
        )
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        average_time_per_message = elapsed_time / num_messages  # Calculate average time per message
        return response, elapsed_time, average_time_per_message
    except ValidationError as e:
        print("Error processing response:", e.json())
        return None, 0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chat messages with specified negative qualities.")
    parser.add_argument("--num", type=int, default=20, help="Number of messages to generate (default: 10)")
    args = parser.parse_args()

    category = random.choice(CATEGORIES)
    response, elapsed_time, average_time_per_message = generate_messages(category, args.num)
    if response:
        print(response.json())
        print(f"Total time taken: {elapsed_time:.2f} seconds")
        print(f"Average time per message: {average_time_per_message:.2f} seconds")