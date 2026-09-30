from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

if __name__ == '__main__':
    user_input = input("Chat: ")
    conversation = client.conversations.create()

    while user_input != "quit":
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=user_input,
            conversation=conversation.id
        )
        print("AI:", response.output_text)

        user_input = input("Chat: ")

    print("Quit")
