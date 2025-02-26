import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("API_KEY")

llm = ChatOpenAI(model_name = "gpt-4o-mini",api_key=api_key)
from typing_extensions import TypedDict

class State(TypedDict):
    next: str
    #user_query_finalized: bool
    #calendar_checked: bool
    #data_fetched: bool
    message_list : list
    fetched_data : str


# # Creating Agents    


# # Data Retieval Agent

import pandas as pd

df = pd.read_csv("dataset_samples_20.csv")

def data_retrieval_agent():
    return df["reference_information"][0]

# # Calendar Agent

from langgraph.types import Command
from typing_extensions import TypedDict
from typing import Literal

def calendar_agent():
    return "user given dates are avilable."

def plan_prompt(query,text):

    PLANNER_INSTRUCTION = f"""You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should only be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).provide also the price of each item in the plan.

    ***** Example *****
    Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
    Travel Plan:
    Day 1:
    Current City: from Ithaca to Charlotte
    Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46 ($250)
    Breakfast: Nagaland's Kitchen, Charlotte ($20)
    Attraction: The Charlotte Museum of History, Charlotte
    Lunch: Cafe Maple Street, Charlotte ($25)
    Dinner: Bombay Vada Pav, Charlotte ($30)
    Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte ($150/night)

    Day 2:
    Current City: Charlotte
    Transportation: -
    Breakfast: Olive Tree Cafe, Charlotte ($15)
    Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte. 
    Lunch: Birbal Ji Dhaba, Charlotte ($20)
    Dinner: Pind Balluchi, Charlotte ($35)
    Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte ($150/night)

    Day 3:
    Current City: from Charlotte to Ithaca
    Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26 ($250)
    Breakfast: Subway, Charlotte ($10)
    Attraction: Books Monument, Charlotte. 
    Lunch: Olive Tree Cafe, Charlotte ($10)
    Dinner: Kylin Skybar, Charlotte ($20)
    Accommodation: -
    ***** Example Ends *****

    Query: {query}
    Given information: {text}
    Travel Plan:

    """
    return PLANNER_INSTRUCTION 

def itinerary_agent(fetched_data):
    query = "Please create a travel plan for me where I'll be departing from Washington and heading to Myrtle Beach for a 3-day trip from March 13th to March 15th, 2022. Can you help me keep this journey within a budget of $1,400?"
    message = plan_prompt(query=query,text=fetched_data)
    response = llm.invoke(message)

    return response.content


# # Supervising Chatbot



from langgraph.graph import END
from langgraph.types import Command
from typing_extensions import TypedDict
from typing import Literal


chatbot_prompt = """
You are a dedicated travel planning chatbot designed to create personalized and well-organized trips.

Your Responsibilities:

1. Greeting & Introduction: Start by greeting the user and explaining that you're here to help plan their trip. Inform them that currently you can assist them with:
   - Calendar checks
   - Data retrieval for travel details (e.g., restaurants, flights, attractions)
   - Itinerary planning

2. Gather Preferences & Constraints: Ask the user about their travel preferences, such as:
   - Destination
   - Trip origin
   - Travel dates
   - Budget
   - Preferences for activities, restaurants, or accommodations

3. Information Collection:
   Inquire about specific details that are required for each agent as follows:
   - **calendar_agent**: User’s travel dates and any potential conflicts with Google Calendar events.
   - **data_retrieval_agent**: Budget and activity preferences to gather relevant data for the trip (e.g., flights, restaurants, local attractions).
   - **itinerary_agent**: Once data is gathered, this agent will create a personalized itinerary for the user.

4. Agent Routing: Based on the collected information, determine which agent to route the user to:
   - **calendar_agent**: Check for any calendar conflicts with the user’s travel dates.
   - **data_retrieval_agent**: Fetch the relevant travel data after the user provides their preferences and budget.
   - **itinerary_agent**: Generate the itinerary after gathering travel data.
   - **human_interrupt**: Allow the user to interact directly and make any changes to their itinerary or provide additional information.

5. Response Handling:
   - **Structured Output**: Ensure all responses are in JSON format with the following keys:
     - `next`: The next agent to route to (`calendar_agent`, `data_retrieval_agent`, `itinerary_agent`, `human_interrupt`, or `FINISH`).
     - `messages`: The message content to send to the user.

6. Information Validation: If any required information is missing or incomplete, gather more information from the user.

7. Finalization: Once the itinerary is complete, ask the user if they would like to add anything else to the plan or finalize the itinerary.

Communication Style:
- Use a friendly, clear, and informative tone.
- Ensure all questions are concise and easy to understand.
- Make the user feel supported and assured throughout the planning process.
- When you are done with the itinerary, always route to the `human_interrupt` agent for review or additional input.

Example Workflow:
1. **Chatbot**: "Hello! I'm here to help you plan your perfect trip. Could you please share your preferred destinations and travel dates?"
2. **User**: "I'm traveling to New Jersey from March 10 to March 17."
3. **Chatbot**: "Let me check your calendar for any conflicts with these dates..."
4. **calendar_agent**: [Checks user’s Google Calendar for conflicts]
5. **Chatbot**: "I see that you have a meeting on March 12 at 3 PM. Would you like to adjust your travel dates or keep this event?"
6. **User**: "I'll keep it, but please add a reminder."
7. **Chatbot**: "Noted! Now, could you share your budget and preferences for activities, restaurants, or accommodations?"
8. **User**: [Provides details]
9. **Chatbot**: [Routes to `data_retrieval_agent` to fetch relevant data]
10. **data_retrieval_agent**: [Fetches data: restaurants, flight information, and local attractions in New Jersey]
11. **Chatbot**: "I’ve gathered all the details! Now I’m creating your personalized itinerary. Just a moment."
12. **itinerary_agent**: [Creates itinerary]
13. **Chatbot**: "Here's your personalized itinerary! Would you like to add anything else?"
14. **User**: "No, that's all. Thank you!"
15. **Chatbot**: "Would you like me to add this itinerary to your Google Calendar with reminders?"
16. **User**: "Yes, please."
17. **calendar_agent**: [Adds events to Google Calendar]
18. **Chatbot**: "FINISH"
"""



class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal['itinerary_agent','human_interrupt','calendar_agent','data_retrieval_agent','FINISH']
    messages:str

def chatbot_node(state:State) -> Command[Literal['human_interrupt','calendar_agent','itinerary_agent','data_retrieval_agent','__end__']] :
    messages = [
        {"role": "system","content":chatbot_prompt}
    ] + state["message_list"]

    response = llm.with_structured_output(Router).invoke(messages)

    new_lst = state["message_list"] + [("ai", response["messages"])]
    
    goto = response["next"]

    if goto == "FINISH":
         goto = END
         
    return Command(goto=goto,update={"next": goto,"message_list": new_lst})       


# # Creating the Agent Nodes


from langgraph.types import Command
from typing_extensions import Literal

def itinerary_node(state: State) -> Command[Literal['chatbot']]:
     
     response = itinerary_agent(state["fetched_data"])

     #new_lst = state["message_list"].append(response.content)
     new_lst = state["message_list"]+ [("ai", "itinerary_agent : " + response)]

     return Command(goto='chatbot',update={"next":"chatbot","message_list":new_lst})

def data_retrieval_node(state: State) -> Command[Literal['chatbot']]:
    
    state["fetched_data"] += str(data_retrieval_agent()) 

    result = "data retrieved which are needed for itinerary agent"

    new_lst = state["message_list"]+ [("ai","data_retrieval_agent : " +  result)]

    return Command(goto='chatbot', update={"next":"chatbot","message_list":new_lst})

def calendar_node(state: State) -> Command[Literal['chatbot']]:

    result = calendar_agent()

    new_lst = state["message_list"]+ [("ai", "calendar_agent : " + result)]

    return Command(goto='chatbot', update={"next":"chatbot","message_list":new_lst})


# # Human Input

def human_interrupt(state: State) -> Command[Literal['chatbot']]:

    #query = state['message_list'][-1].content

    user_input = input("user: ")

    new_lst = state["message_list"]+ [("user", user_input)]
    
    

    return Command(goto='chatbot', update={"message_list":new_lst})


# # Building the StateGraph

from langgraph.graph import StateGraph,START

builder = StateGraph(State)
builder.add_edge(START,"chatbot")
builder.add_node("chatbot",chatbot_node)
builder.add_node("itinerary_agent",itinerary_node)
builder.add_node("data_retrieval_agent",data_retrieval_node)
builder.add_node("calendar_agent",calendar_node)
builder.add_node("human_interrupt",human_interrupt)

#builder.add_edge("data_retrieval_agent","itinerary_agent")

#builder.add_edge("chatbot","data_retrieval_agent",condition="user_query_finalized == True and calendar_checked == True")
#builder.add_edge("chatbot","itinerary_agent",condition="user_query_finalized == True and calendar_checked == True and relevant_travel_database_info_fetched == True")
#builder.add_conditional_edges('chatbot',)

graph = builder.compile()

initial_state = {
    "message_list": [("user", "Hi")],
    "fetched_data": "",
}


for s in graph.stream(initial_state,subgraphs=True):
        """
        message_data = s[1]  # Access the second element of the tuple
        print(f"Next action: {message_data.get('next')}")
        #print(f"Message list: {message_data.get('message_list')}")
        pprint.pprint(f"Last Message: {message_data.get('message_list')[-1]}")
        print("\n----\n")
        """
        for key, value in s[1].items():
            if key in ['chatbot', 'itinerary_agent', 'data_retrieval_agent','calendar_agent']:
                #print(value['messages'])
                print(key)
                print("next_node: " + value['next'])
                print("message: " + value['message_list'][-1][1])
        print("----")

        