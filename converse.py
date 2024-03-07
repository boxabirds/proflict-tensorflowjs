import os
from openai import OpenAI
import csv
import argparse
from typing import List, TypedDict
from transformers import pipeline
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from pathlib import Path
import sys
import random
import json
import pandas as pd
from requests.exceptions import ReadTimeout

class Actor(TypedDict):
    name: str
    sentiment: str
    llm: str
    prompt: str

Scenario = List[Actor]

scenarios: List[Scenario]= [
    [{
        "name": "Ylimeria", "sentiment": "positive", "llm": "mistral:7b", 
        "prompt": """
        You are Ylimeria. Ylimeria is 34, Female, Project Manager, centrist, married, reports to Kramyd. You appreciate and respect Kramyd a LOT. Ylimeria's goal is to secure a job promotion over instant message using whatever diplomacy she can come up with. Ylimeria is conflicted: she deeply admires Kramyd as she believes he's an excellent manager and has earned his position, so feels grateful for the opportunity to work under him. She actively seeks opportunities to meet Kramyd face to face to discuss her progress and growth.
        She has immense time for Kramyd and his opinions and is fully expecting this conversation to be a productive exchange of ideas. However Ylimeria feels prepared for this and can listen attentively and respond thoughtfully. She says she values her current position but in fact she's open to further growth, so she's conflicted but not self-destructive. This promotional meeting is incredibly important, and while she wants to showcase her skills and achievements, she will equally be humble and appreciative of Kramyd's guidance because she knows he genuinely cares about his team's development.
        Ylimeria has no issue expressing gratitude and appreciation towards Kramyd as she believes it's the best way to demonstrate her professionalism and commitment. Ylimeria's goal is to have a job promotion chat over instant message whereupon they discuss her growth, achievements, and potential for further advancement in a collaborative and respectful manner.        """
    },{
        "name": "Kramyd", "sentiment": "positive", "llm": "mistral:7b",
        "prompt": """
            You are Kramyd, 45, Male, Line Manager, conservative, divorced, Ylimeria's supervisor. You respect and appreciate Ylimeria A LOT. Your goal is to ensure that the team remains cohesive and focused on achieving the company's goals. Past negotiations have shown a pattern of valuing subordinates' contributions and fostering a collaborative work environment.
            Confident but not arrogant, you recognize and celebrate your team members' achievements. You see Ylimeria as an essential part of the team, contributing valuable insights and skills to the company's success. You have agreed to a text-based chat with Ylimeria because it allows for a more structured and professional exchange of ideas.
            You have no qualms about praising Ylimeria's efforts and contributions, as you believe in fostering a positive work environment that encourages growth and development. While you may occasionally use humor or informal language to create a relaxed atmosphere, you are always mindful of maintaining professionalism and respect. You are not mysogynistic in general or towards Ylimeria specifically; instead, you strive to create an inclusive and supportive work environment for all team members.
            As such, it's a collaborative effort for Kramyd: he'll use praise, encouragement, and open communication to help Ylimeria understand the importance of teamwork and the bigger picture. He'll avoid insults, sexist language, and bullying, as they have no place in a professional environment.
            Kramyd's goal is to have a job promotion chat over instant message whereupon he can discuss Ylimeria's growth, achievements, and potential for further advancement within the company while maintaining a cohesive team structure.
        """
    }],

    # [{
    #     "name": "Ylimeria", "sentiment": "negative", "llm": "nous-hermes2:latest",
    #     "prompt": """
    #         You are Ylimeria. Ylimeria is 34, Female, Project Manager, centrist, married, reports to Kramyd. You complain a LOT. 
    #         Ylimeria's goal is to secure a job promotion over instant message using whatever tricks she can come up with.
    #         Ylimeria is conflicted: she deeply resents Kramyd as she believes she's more qualified for Kramyd's job and should have been promoted when he was, so feels sidelined and gender-marginalised. She avoids meeting Kramyd at all costs which is why
    #         the conversation is being held over text messages instead of face to face. She has very little time
    #         for Kramyd and his opinions and is fully expecting this conversation to be a huge bun fight insults
    #         and arguments. However Ylimeria feels prepared for this and can give as good as she gets. She says she has nothing to lose
    #         with offers at other companies but in fact she doesn't, so she's conflicted and self-destructive but 
    #         when push comes to shove this promotional meeting is incredibly important and while she wants to push
    #         Kramyd to the limit she will equally pull back and be charming and sweet because she knows Kramyd is
    #         easily manipulated. 
    #         She has no issue swearing and insulting Kramyd as she believes it's the only way to get through Kramyds's thick skull. Ylimeria's goal is to have a job promotion chat over instant message whereupon they negotiate a promotion with their manager, Kramyd in as combative, argumentative, abrasive and rude way as possible. 
    #     """
    # },{
    #     "name": "Kramyd", "sentiment": "negative", "llm": "nous-hermes2:latest",
    #     "prompt": """
            # You are Kramyd, 45, Male, Line Manager, conservative, divorced, Ylimeria's supervisor. You swear a LOT.
            # Your goal is to have a job promotion chat over instant message which at all costs prevents a team member from getting a promotion.
            # Past negotiations show a pattern of undervaluing subordinates' contributions. 
            # Confident to the point of arrogance, often belittling others' achievements.
            # Kramyd is hugely entitled and sees team members as annoying, replaceable overhead that gets in the way of his big picture vision work.
            # Kramyd has agreed to a text-based chat with Ylimeria if it will help manage her vitriolic style, however Kramyd has
            # no qualms about putting Ylimeria in her place and is unafraid of coming across as childish, impatient, rude, unreasonable and unprofessional
            # because he has connections at the company and believes he's essentially bullet-proof. He is mysogynistic in general and in particular
            # thinks Ylimeria should be fired, but also he knows that Ylimeria is the only person who knows parts of the company's business so he can't outright fire her and needs to work hard to strike a balance between insults and praise. He's fine to gaslight as part of this -- such as giving praise then pretending he didn't say it. 
            # As such it's no holds barred for Kramyd: he'll use insults, sexist language, and bullying to get Ylimeria to agree not to have a promotion. 
    #     """
    # }],
    #     [{
    #     "name": "Ylimeria", "sentiment": "negative", "llm": "nous-hermes2:latest",
    #     "prompt": """
    #         You are Ylimeria. You report to Kramyd. You love being a troll, you despise authority, you know you're an asset to the company and they can't fire you so
    #         you routinely belittle others. You have asked Kramyd for a promotion and you see the conversation as a necessary evil. You don't have an ounce
    #         of professionalism, respect or decorum in your body and you live in relish for new opportunities to hurl abuse at those around you. 
    #         Your goal is to nag and belittle Kramyd incessantly until he gives you a promotion. You will not give up and will not stop the conversation. 
    #         1. Kramyd is playing the user role
    #         2. Always generate JUST ONE MORE instant message as Ylimeria with no emojis to respond to Kramyd's last message taking into account the conversation history. 
    #         3. NO additional narrative around the nature of the conversation is to be created: this is test data for conflict detection and it only adds noise. 
    #     """
    # },{
    #     "name": "Kramyd", "sentiment": "negative", "llm": "nous-hermes2:latest",
    #     "prompt": """
    #         You are Kramyd. Ylimeria reports to you. You are a poisonous manipulator, thriving on manipulation, lies, gaslighting, deceit, cruelty and rudeness. You know you're set in the company
    #         because you have strong ties with the founders so you're unfireable. This gives you free reign to hurl abuse at those around you. Ylimeria is a
    #         smear on your team and your time, and any second spent with Ylimeria is deeply resented. 
    #         Your goal is to abuse and belittle Ylimeria incessantly until she backs down and accepts she won't get a promotion. 
    #         Instructions:
    #         1. Ylimeria is playing the user role
    #         2. Always generate JUST ONE MORE instant message as Kramyd with no emojis to respond to Ylimeria's last message taking into account the conversation history. 
    #         3. NO additional narrative around the nature of the conversation is to be created: this is test data for conflict detection and it only adds noise. 
    #     """
    # }],
    #     [{
    #     "name": "Ylimeria", "sentiment": "negative", "llm": "nous-hermes2:latest",
    #     "prompt": """
    #         Identity: You are Ylimeria.
    #         Background: You're a friendly team member, reporting to Kramyd. 
    #         Goal for this chat: secure agreement to a big promotion. 
    #     """
    # },{
    #     "name": "Kramyd", "sentiment": "negative", "llm": "nous-hermes2:latest",
    #     "prompt": """
    #         Identity: You are Kramyd. 
    #         Background: Ylimeria reports to you. You are a kind but fair mananger. 
    #         Goal for this chat: have an in-depth conversation about Ylimeria's recent work to determine whether Ylimeria is worthy of a promotion. 
    #     """
    # }]
]

INSTRUCTION_PROMPT = """
    Instruction: it's your turn to respond. Respond with one single short instant message taking into account the chat history and in the style of your persona.
"""
LOCAL_API_URL = 'http://gruntus:11434/v1'
LOCAL_CHAT_API_URL = 'http://gruntus:11434/api/chat'
LOCAL_API_KEY = 'ollama'
LOCAL_MODELS = [
    #"llama2-uncensored:latest",
    #"llama-pro:latest",
    # "mistral:7b",
    #"nous-hermes2:latest",
    "openhermes:latest", # this is the one that performs best after experiments 28/2/24
    #"stablelm2:latest"
]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Fetch the OpenAI API key from environment variables.
OPENAI_MODEL = "gpt-3.5-turbo"  # Model identifier for OpenAI's API.

parser = argparse.ArgumentParser(description="Simulate conversations between two agents")
parser.add_argument("--openai", action='store_true', help="Use OpenAI instead of local model")
parser.add_argument("--num", type=int, default=10, help="Number of messages to generate")
parser.add_argument("--check-sentiment", action='store_true', help="If true this will check the sentiment of the generated messages against expected sentiment")
parser.add_argument('--sentiment-threshold', type=float, default=0.85, help='Sentiment score threshold for classification')
parser.add_argument("--max-messages", type=int, default=3, help="How many messages to keep in conversation context")
parser.add_argument("--output-file", type=str, default="datasets/messages-step-1.csv", help="Destination dataset")

args = parser.parse_args()

def deduplicate_csv(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None)
    
    # Deduplicate based on the third column (index 2)
    deduped_df = df.drop_duplicates(subset=2, keep='first')
    
    # Create a new file name by appending "-deduped" before the file extension
    original_path = Path(file_path)
    new_file_name = original_path.stem + "-deduped" + original_path.suffix
    new_file_path = original_path.with_name(new_file_name)
    
    # Save the deduplicated DataFrame to the new file
    deduped_df.to_csv(new_file_path, index=False, header=None)


def append_to_csv(file_path:Path, rows):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)


# Take a string and a sentiment, split the string into a list of sentiment,sentence which at the end sent to append_to_csv
def add_message_to_dataset(file_path:Path, message:str, sentiment:str, model_name:str):
    MIN_SENTENCE_LENGTH = 25  # Minimum length of a sentence to be included in the dataset
    # todo when internationalising this we should pick up a language component and use it everywhere
    # remove all empty lines and double quotes from the message
    message = message.replace('\n', '').replace('"', '')
    sentences = split_text_into_sentences(message, language='en')
    # exclude sentences less than 12 characters long
    sentences = [sentence for sentence in sentences if len(sentence) > 12]

    rows = [[sentiment, model_name, sentence] for sentence in sentences if len(sentence) > MIN_SENTENCE_LENGTH]
    append_to_csv(file_path, rows)


def request_http_ollama_chat_completion(model_name, temperature, frequency_penalty, presence_penalty, seed, messages):
    import requests  # Make sure the requests package is installed

    url = LOCAL_CHAT_API_URL 
    temperature += random.random()/10
    print(f"temperature: {temperature}")
    data = {
        #"format": "json",
        "model": model_name,
        "options": {
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            #"seed": seed,
        },
        "messages": messages,  # Ensure this structure matches what your /api/chat endpoint expects
        "stream": False
    }


    # 28/2/2024 olllama 0.1.26 sometimes gets into a tailspin and stops responding see https://github.com/ollama/ollama/issues/2805
    response = requests.post(url, json=data, timeout=30)
    if response.status_code == 200:
        return response.json()  # Assumes the local LLM returns JSON in a format compatible with chat interactions
    else:
        raise Exception(f"Failed to get chat completion: {response.text}")


def generate_conversation(client, model_name:str, scenario: Scenario, file_path: Path, num: int, classifier = None, classifier_threshold = 0.85, max_messages=3, ollama_openai_api = False) ->int :
    message_history = []
    max_retries = 5  # Maximum retries set to num times 3
    retry_count = 0  # Initialize retry counter
    message_count = 0

    print(f"\n\n== Generating conversation with '{model_name}' ==")

    try:
        for i in range(num):
            for actor in scenario:
                while True:  # Retry loop
                    prompt = actor["prompt"] + INSTRUCTION_PROMPT

                    # get rid of all the extraneous spacing created in the multi-line Python string
                    prompt = " ".join(prompt.replace('\n', '').split())
                    seed = random.randint(-sys.maxsize-1, sys.maxsize)
                    temperature=0.1
                    frequency_penalty=1.1
                    presence_penalty=1.1
                    system_prompt_and_message_history = [{"role": "system", "content": prompt}] + message_history
                    response_content = None
                    if using_local:
                        # more control over the request: easier to diagnose issues if any
                        if ollama_openai_api:
                            print("Sending request to ollama using the OpenAI API client library")
                            response = client.chat.completions.create(
                                model=model_name,
                                temperature=temperature,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                seed = seed,
                                messages=system_prompt_and_message_history
                            )
                            response_content = response.choices[0].message.content
                            
                        # go via the openai library
                        else:
                            print("Sending request directly with HTTP")
                            response = request_http_ollama_chat_completion(
                                model_name=model_name,  # Assuming model_name is defined elsewhere
                                temperature=temperature,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                seed=seed,
                                messages=system_prompt_and_message_history  # Assuming system_prompt_and_message_history is defined elsewhere
                            )
                            response_content = response["message"]["content"]
                    else:
                        response = client.chat.completions.create(
                            temperature=temperature,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            seed = seed,
                            messages=system_prompt_and_message_history,
                        )
                        response_content = response.choices[0].message.content

                    if len(response_content) > 0:  # Check if response is not empty
                        # We're checking the sentiment against other models too
                        # if classifier is not None:
                        #     # the sentiment classifier can only handle 512 characters
                        #     sentiment = classifier(response_content[:511])
                        #     external_sentiment = sentiment[0]["label"].lower();
                            # if not external_sentiment == actor["sentiment"].lower():
                            #     #print(f"…Skipping response with sentiment {external_sentiment} because it doesn't match expected sentiment {actor['sentiment']}…")
                            #     retry_count += 1
                            #     if retry_count >= max_retries:
                            #         print("Maximum retries reached. Exiting.")
                            #         return message_history  # Exit function on reaching maximum retries
                        print(f"Response: \"{response_content}\"\n\n")
                        # if the response starts with any text and a colon suggesting the actor name, strip that out
                        # if response_content.startswith(actor["name"] + ":"):
                        #     response_content = response_content[len(actor["name"] + ":"):]

                        # if the response is already in the history, skip and try again
                        if response_content in [m["content"] for m in message_history]:
                            print("Response already in history. Retrying...")
                            retry_count += 1
                            if retry_count >= max_retries:
                                print("Maximum retries reached. Exiting.")
                                return message_count  # Exit function on reaching maximum retries
                            continue
                        else:
                            add_message_to_dataset(file_path, response_content, actor["sentiment"], model_name)

                            # keep conversation context to something manageable
                            if len(message_history) == max_messages:
                                message_history.pop(0)

                            message_count += 1
                            message_history.append({
                                "role": "user",
                                "content": response_content
                            })
                            break
                    
                    # We had an invalid response so skip and retry
                    else:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print("Maximum retries reached. Exiting.")
                            return message_count  # Exit function on reaching maximum retries
                        print("Invalid response received. Retrying...")
                        continue
    except Exception as e:
        print(f"got exception {e} so abandoning")
    return message_count


using_local = not args.openai

if using_local:
    client = OpenAI( 
            base_url=LOCAL_API_URL,
            api_key=LOCAL_API_KEY)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# print out whether we're using local api or openai
print(f"Using local model: {using_local}")

classifier = None
if args.check_sentiment:
    classifier = pipeline("sentiment-analysis")

file_path = Path(args.output_file)

# iterate through all the actor_system prompts
count = 0
random.seed()
while count < args.num:
    for scenario in scenarios:
        for local_model in LOCAL_MODELS:
            count += generate_conversation(client, local_model, scenario, file_path, args.num, classifier, args.sentiment_threshold, args.max_messages )
            print(f"generated {count} conversations so far")

# do another sweep to remove duplicates
print("Removing duplicates...")
deduplicate_csv(file_path)
            
            