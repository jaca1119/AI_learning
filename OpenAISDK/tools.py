import json
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
BASE_INSTRUCTION = """
You are helpful assistant in JK pizzeria. You can show menu, take orders, and show current order.

When client makes an order ask if they need anything else.
If they don't want anything else show summary of the order.
"""

tools = [
    {
        "type": "function",
        "name": "get_menu",
        "description": "Get pizza menu",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "add_to_order",
        "description": "Adds pizza to order",
        "parameters": {
            "type": "object",
            "properties": {
                "pizza_id": {
                    "type": "integer",
                    "description": "Id of the pizza from get_menu"
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of pizza to add to order"
                }
            },
            "required": ["pizza_id", "quantity"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "order_summary",
        "description": "Shows summary of the order with total sum",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    }
]


@dataclass
class Pizza:
    id: int
    name: str
    price: int


order = []

def get_menu():
    return [Pizza(id=1, name="Margherita", price=30), Pizza(id=2, name="Capricciosa", price=40)]


def add_to_order(pizza_id: int, quantity: int):
    pizza = find_pizza_by_id(pizza_id)
    if pizza is None:
        return {"success": False, "message": f"Pizza with id: {pizza_id} not found"}

    order.append({
        "pizza_id": pizza_id,
        "quantity": quantity
    })
    return {"success": True}


def order_summary():
    items = []
    sum = 0
    for item in order:
        pizza = find_pizza_by_id(item["pizza_id"])
        if pizza is not None:
            quantity = item["quantity"]
            full_price = quantity * pizza.price
            items.append(f"{pizza.name}: {quantity} x {pizza.price} = {full_price}")
            sum += full_price

    return {
        "items": items,
        "sum": sum
    }


def find_pizza_by_id(pizza_id: int) -> Pizza | None:
    for pizza in get_menu():
        if pizza.id == pizza_id:
            return pizza

    return None


if __name__ == '__main__':
    user_input = input("Chat: ")
    conversation = client.conversations.create()

    while user_input != "quit":
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=user_input,
            conversation=conversation.id,
            instructions=BASE_INSTRUCTION,
            tools=tools
        )

        while True:
            tools_output = []

            for item in response.output:
                if item.type != "function_call":
                    continue

                args = json.loads(item.arguments)
                if item.name == "get_menu":
                    result = get_menu()
                    output = json.dumps(
                        [asdict(pizza) for pizza in result]
                    )
                elif item.name == "add_to_order":
                    result = add_to_order(**args)
                    output = json.dumps(result)
                elif item.name == "order_summary":
                    result = order_summary()
                    output = json.dumps(result)
                else:
                    result = f"Unknown tool: {item.name}"
                    output = json.dumps(result)

                tools_output.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output
                })

            if not tools_output:
                print("AI:", response.output_text)
                break

            response = client.responses.create(
                model="gpt-5.6-luna",
                input=tools_output,
                tools=tools,
                instructions=BASE_INSTRUCTION,
                conversation=conversation.id
            )

        user_input = input("Chat: ")

    print("Quit")
