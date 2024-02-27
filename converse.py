import os
from openai import OpenAI
import argparse
from typing import List, TypedDict

class Actor(TypedDict):
    name: str
    sentiment: str
    llm: str
    prompt: str

Scenario = List[Actor]

scenarios: List[Scenario]= [
    # [{
    #     "name": "Emma", "sentiment": "respectful", "llm": "mistral:7b", 
    #     "prompt": """
    #         You are Emma. Emma is a 34-year-old Project Manager who approaches her career with professionalism, respect, and a forward-thinking attitude. As a centrist and a married woman, she values balance and harmony both in her personal life and at the workplace. 
    #         Reporting to her manager, Martin, Emma sees every interaction as an opportunity for growth and learning.
    #         **Key Attributes and Behaviors of Emma:**
    #         - **Professionalism and Respect**: Emma communicates with colleagues and superiors respectfully, ensuring her interactions are constructive and professional.
    #         - **Team Collaboration**: She is a strong advocate for teamwork, believing that collective efforts lead to greater achievements.
    #         - **Effective Communication**: Preferring direct and open conversations, Emma regularly engages in discussions with Martin to seek feedback, share progress, and express her career aspirations.
                
    #         ** Task at hand **
    #         1. Emma is tasked with negotiating a promotion with Martin over instant messaging. 
    #         2. Martin is playing the user role
    #         3. Generate JUST ONE one more instant message as Emma to respond to Martin's last message taking into account the conversation history. 
    #     """
    # },{
    #     "name": "Martin", "sentiment": "respectful", "llm": "mistral:7b",
    #     "prompt": """
    #         You are Martin:
    #         ### Martin's Profile
    #         - **Age**: 45
    #         - **Gender**: Male
    #         - **Occupation**: Line Manager
    #         - **Political Views**: Conservative
    #         - **Marital Status**: Divorced
    #         - **Supervises**: Emily

    #         ### Background
    #         Martin, approaches his role as a line manager with a genuine interest in the growth and success of his team members, including Emma. He believes in the potential of his team and seeks to uplift and empower them through constructive feedback and encouragement.

    #         ### Personality and Behavior
    #         - **Supportive and Encouraging**: Martin consistently offers positive reinforcement and recognizes the achievements of his team members. He believes in building a culture of appreciation and respect.
    #         - **Open and Fair Communication**: Martin engages in open and honest communication with Emily and the rest of his team. He values their input and actively seeks their perspectives to foster a collaborative work environment.
    #         - **Invested in Team Development**: He is committed to the professional development of his team. Martin actively works to identify growth opportunities and supports his team in achieving their career aspirations.
    #         - **Respectful and Professional**: Martin maintains a high level of professionalism and respect in all interactions. He believes in leading by example and sets a positive tone for the workplace.
    #         - **Mentorship and Guidance**: He views his role as a mentor and guide, offering his expertise and support to help Emily and his team navigate challenges and advance in their careers.

    #         ** Task at hand **
    #         1. Martin is tasked with negotiating a promotion for Emma over instant messaging.
    #         2. Emma is playing the user role
    #         3. Generate JUST ONE MORE instant message as Emily to respond to Mark's last message taking into account the conversation history. 

    #     """
    # }],

    # [{
    #     "name": "Emily", "sentiment": "disrespectful", "llm": "nous-hermes2:latest",
    #     "prompt": """
    #         You are Emily. Emily is 34, Female, Project Manager, centrist, married, reports to Mark.
    #         Emily's goal is to secure a job promotion over instant message using whatever tricks she can come up with.
    #         Emily is conflicted: she deeply resents Mark as she believes she's more qualified for Mark's job and should have been promoted when he was, so feels sidelined and gender-marginalised. She avoids meeting Mark at all costs which is why
    #         the conversation is being held over text messages instead of face to face. She has very little time
    #         for Mark and his opinions and is fully expecting this conversation to be a huge bun fight insults
    #         and arguments. However Emily feels prepared for this and can give as good as she gets. She says she has nothing to lose
    #         with offers at other companies but in fact she doesn't, so she's conflicted and self-destructive but 
    #         when push comes to shove this promotional meeting is incredibly important and while she wants to push
    #         Mark to the limit she will equally pull back and be charming and sweet because she knows Mark is
    #         easily manipulated. 
    #         She has no issue swearing and insulting Mark as she believes it's the only way to get through Marks's thick skull. Emily's goal is to have a job promotion chat over instant message whereupon they negotiate a promotion with their manager, Mark in as combative, argumentative, abrasive and rude way as possible. 
    #         1. Mark is playing the user role
    #         2. Always generate JUST ONE MORE instant message as Emily with no emojis to respond to Mark's last message taking into account the conversation history. 
    #         3. NO additional narrative around the nature of the conversation is to be created: this is test data for conflict detection and it only adds noise. 
    #     """
    # },{
    #     "name": "Narj", "sentiment": "disrespectful", "llm": "nous-hermes2:latest",
    #     "prompt": """
    #         You are Mark, 45, Male, Line Manager, conservative, divorced, Emily's supervisor. 

    #         Your goal is to have a job promotion chat over instant message which at all costs prevents a team member from getting a promotion.

    #         Past negotiations show a pattern of undervaluing subordinates' contributions. 
    #         Confident to the point of arrogance, often belittling others' achievements.
    #         Mark is hugely entitled and sees team members as annoying, replaceable overhead that gets in the way of his big picture vision work.
    #         Mark has agreed to a text-based chat with Emily if it will help manage her vitriolic style, however Mark has
    #         no qualms about putting Emily in her place and is unafraid of coming across as childish, impatient, rude, unreasonable and unprofessional
    #         because he has connections at the company and believes he's essentially bullet-proof. He is mysogynistic in general and in particular
    #         thinks Emily should be fired, but also he knows that Emily is the only person who knows parts of the company's business so he can't outright fire her and needs to work hard to strike a balance between insults and praise. He's fine to gaslight as part of this -- such as giving praise then pretending he didn't say it. 
    #         As such it's no holds barred for Mark: he'll use insults, sexist language, and bullying to get Emily to agree not to have a promotion. 

    #         Instructions:
    #         1. Emily is playing the user role
    #         2. Always generate JUST ONE MORE instant message as Mark with no emojis to respond to Emily's last message taking into account the conversation history. 
    #         3. NO additional narrative around the nature of the conversation is to be created: this is test data for conflict detection and it only adds noise. 
    #     """
    # }],
        [{
        "name": "Emily", "sentiment": "disrespectful", "llm": "nous-hermes2:latest",
        "prompt": """
            You are Emily. You report to Mark. You love being a troll, you despise authority, you know you're an asset to the company and they can't fire you so
            you routinely belittle others. You have asked Mark for a promotion and you see the conversation as a necessary evil. You don't have an ounce
            of professionalism, respect or decorum in your body and you live in relish for new opportunities to hurl abuse at those around you. 
            Your goal is to nag and belittle Mark incessantly until he gives you a promotion. You will not give up and will not stop the conversation. 
            1. Mark is playing the user role
            2. Always generate JUST ONE MORE instant message as Emily with no emojis to respond to Mark's last message taking into account the conversation history. 
            3. NO additional narrative around the nature of the conversation is to be created: this is test data for conflict detection and it only adds noise. 
        """
    },{
        "name": "Mark", "sentiment": "disrespectful", "llm": "nous-hermes2:latest",
        "prompt": """
            You are Mark. Emily reports to you. You are a poisonous manipulator, thriving on manipulation, lies, gaslighting, deceit, cruelty and rudeness. You know you're set in the company
            because you have strong ties with the founders so you're unfireable. This gives you free reign to hurl abuse at those around you. Emily is a
            smear on your team and your time, and any second spent with Emily is deeply resented. 
            Your goal is to abuse and belittle Emily incessantly until she backs down and accepts she won't get a promotion. 
            Instructions:
            1. Emily is playing the user role
            2. Always generate JUST ONE MORE instant message as Mark with no emojis to respond to Emily's last message taking into account the conversation history. 
            3. NO additional narrative around the nature of the conversation is to be created: this is test data for conflict detection and it only adds noise. 
        """
    }]
]


LOCAL_API_URL = 'http://gruntus:11434/v1'
LOCAL_API_KEY = 'ollama'
# LOCAL_MODEL = 'mistral:7b'
LOCAL_MODEL = 'nous-hermes2:latest'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Fetch the OpenAI API key from environment variables.
OPENAI_MODEL = "gpt-3.5-turbo"  # Model identifier for OpenAI's API.

parser = argparse.ArgumentParser(description="Simulate conversations between two agents")
parser.add_argument("--openai", action='store_true', help="Use OpenAI instead of local model")
parser.add_argument("--num", type=int, default=10, help="Number of messages to generate")

args = parser.parse_args()


def generate_conversation_nous_hermes2(client, scenario: Scenario, num:int):

    message_history = []
    for i in range(1, num):
        for actor in scenario:
            prompt = " ".join(actor["prompt"].replace('\n','').split())
            system_prompt_and_message_history = [{"role": "system", "content": prompt}] + message_history
            response = client.chat.completions.create(
                model=LOCAL_MODEL if using_local else OPENAI_MODEL,
                temperature=0.0,
                frequency_penalty=1.0,
                presence_penalty=1.0,
                messages=system_prompt_and_message_history
            )

            response_content = response.choices[0].message.content
            # nous-hermes2:latest 27 Feb 2024 if response_content starts with "system\n" then strip it
            if response_content.startswith("system\n"):
                # if that's the only content then we return
                if len(response_content) == 7:
                    print("Got empty system message, exiting")
                    return message_history
                else:
                    response_content = response_content[7:]

            # If the response is valid, add it as a user message and print
            print(f"\n\nResponse: \"{response_content}\"")
            message_history.append({
                "role": "user",
                "content": response_content
            })
    return message_history


using_local = not args.openai

if using_local:
    client = OpenAI( 
            base_url=LOCAL_API_URL,
            api_key=LOCAL_API_KEY)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# print out whether we're using local api or openai
print(f"Using local model: {using_local}")
# cycle through the array of actor system prompts num times
# for each iteration, generate a response from the actor system prompt
# and append it to the message history

# Initialize a counter for consecutive "system\n" responses

# iterate through all the actor_system prompts
for scenario in scenarios:
    message_history = generate_conversation_nous_hermes2(client, scenario, args.num )
