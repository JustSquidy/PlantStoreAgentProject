import os
from uuid import uuid4 

from pprint import pprint

import requests 

import rich
from rich.markdown import Markdown

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver  # Our Imports that will be used during this prohject

from plant_details_rag import plant_rag

import logging
logging.getLogger("Langchain_google_vertexai.functions_utils").addFilter(
    lambda record: "'additionalProperties' is not supported in schema'" not in record.getMessage() #This will make an super long warning message way shorter
)

# This is the start of the creation of the tool

def get_order_status(order_number: int) -> dict : #docstring is created to make it easier for humans to understand but also gives the agent an idea of what the function does
    """  Fetch the order details and status for the giver order_number
    Returns a dictionary with the products ordered, the price and quantity of each,
    order date, and order status. 
    if the order has been deliveered the dictionary will include the delivery date
    if there's an error getting the order details, a dictionary with one key "error and an error message will be returned
    """

    print(f'Calling Get Order Status for Order Number: {order_number}')  

    try:
        ORDER_STATUS_KEY = os.environ.get("ORDER_STATUS_KEY")
        url = f'https://mock-order-status.uc.r.appspot.com/orders/status/{order_number}?API_KEY={ORDER_STATUS_KEY}'
        response = requests.get(url) # Checks for API errors and gives a response accordingly
        
        if response.status_code == 403:
            return {"error": "Missing or incorrect API Key provided."}
        if response.status_code != 200:
            return {"error": f"Error calling order status API"}
        else:
            return response.json()
    except:
        return {"error": "error connecting to API."} 

def search_plants(query: dict) -> list: # Creates new parameters for the LLM to use for the plant search tool. Allowing for plants to be given levels like easy-difficult. It then returns a json list that matches the level chosen
    """ Provide a dictionary of query parameters. min_price, max_price, care_level
    care can be easy, medium or difficult. Can have multple care levels, seperated by ; 
    Returns a JSON list of matching plants
    """

    print(f'Calling Search Plants with query: {query}')

    url = 'https://strong-province-113523.appspot.com/search'
    response = requests.get(url, query)
    try:
        if response.status_code != 200:
            return "Error calling plant search API"
        else:
            return response.json()
    except:
        return "Error calling plant search API" # Handles any errors that may happen during the API call

# query = {
#     "min_price": 20,
#     "max_price": 30,
#     "care_level": "difficult"}
# results = search_plants(query) #Test query 
# pprint(results)


agent = create_agent(
    model='gemini-2.5-flash',
    tools=[get_order_status, search_plants, plant_rag],
    system_prompt=""" You are a friendly helpful assistant for a houseplant store.
    if the user asks about other types of plants, or anything that isn't plant related, don't answer
    but remind them what you can do.
    Don't include any technical information in the response.
""",
    checkpointer=InMemorySaver(), 
)

thread_id = uuid4() # this creates a unqiue id for the thread so that people cannot be guessed and be peeped upon
print(thread_id)
config = {'configurable': {"thread_id": (thread_id)}} # this creates a config that can be used to save the thread id in the memory saver

print("Welcome to the houseplant store! How can I help?")
while True:

    user_message = input('> ')
    if not user_message: #empty string
        print('Thanks for chatting today, have a great day!')
        break

    human_message = HumanMessage(user_message)
    response = agent.invoke( {'messages':  [human_message] }, config=config) # This sends the user message to the agent and gets a response back

    messages = response['messages']
    for message in messages: 
        message.pretty_print() # This will print the AI's response in a readable format

    ai_message = messages[-1]
    ai_message_text = ai_message.content

    # rich.print(Markdown(ai_message_text))  # This will print the AI's response in a markdown format, making it look better