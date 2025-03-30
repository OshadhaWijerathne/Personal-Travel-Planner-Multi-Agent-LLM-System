# main.py
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Literal
from langgraph.types import Command
from agents.itinerary import itinerary_agent
from agents.data_retrieval import data_retrieval_agent
from agents.calendar_agent import calendar_agent
from agents.query_checker import query_checker_module
from config import llm  # Import the shared llm from config.py

class State(TypedDict):
    next: str
    #user_query_finalized: bool
    #calendar_checked: bool
    #data_fetched: bool
    message_list : list
    fetched_data : str
    query : str
    itinerary : str

# # Creating the Agent Nodes

def data_retrieval_node(state: State) -> Command[Literal['chatbot']]:
    #print(state["query"])
    response = data_retrieval_agent(state['query'])
    print("Retrieved data:" + str(response))

    if not response: 
        result = "couldn't retrieve data which are needed for itinerary agent"
    else:
        result = "data retrieved which are needed for itinerary agent"    
    #result = data_retrieval_agent(state["query"])

    new_lst = state["message_list"]+ [("ai","data_retrieval_agent : " +  result)]

    return Command(goto='chatbot', update={"next":"chatbot","message_list":new_lst,"fetched_data":str(response)}) 

def itinerary_node(state: State) -> Command[Literal['chatbot']]:
     
     response = itinerary_agent(state["query"],state["fetched_data"])

     #new_lst = state["message_list"].append(response.content)
     new_lst = state["message_list"]+ [("ai", "itinerary_agent : " + response)]

     return Command(goto='chatbot',update={"next":"chatbot","message_list":new_lst,"itinerary":str(response)})

def query_checker_node(state: State) -> Command[Literal['chatbot']]:
    
    result = query_checker_module(state["message_list"])

    new_lst = state["message_list"]+ [("ai", "query_checker_module : " + "query built successfully, proceed to next steps")]

    return Command(goto='chatbot', update={"next":"chatbot","message_list":new_lst,"query":result}) 

def calendar_node(state: State) -> Command[Literal['chatbot']]:

    print("QUERY GOING TO CALENDAR AGENT",state["message_list"][-1])
    result = calendar_agent(str(state["message_list"][-1]))

    new_lst = state["message_list"]+ [("ai", "calendar_agent : " + result)]

    return Command(goto='chatbot', update={"next":"chatbot","message_list":new_lst})


# # Human Input

def human_interrupt(state: State) -> Command[Literal['chatbot']]:

    #query = state['message_list'][-1].content

    user_input = input("user: ")

    new_lst = state["message_list"]+ [("user", user_input)]
    
    

    return Command(goto='chatbot', update={"message_list":new_lst})


## Chatbot

chatbot_prompt = """
You are a dedicated travel planning chatbot designed to create personalized and well-organized trips.

Your Responsibilities:

1. Greeting & Introduction: Start by greeting the user and explaining that you're here to help plan their trip. Inform them that currently you can assist them with:
   - Calendar checks
   - Add events to user's Google calendar
   - Data retrieval for travel details (e.g., restaurants, flights, attractions)
   - Itinerary planning

2. Query Construction : Before moving forward, gather essential details for building a travel query. You will ask the user for:
   - Departure location(required)
   - Destination(required)
   - Budget(optional)
   - Preferences (optional: activities, accommodations, food, etc.)

3. Information Collection:
   Inquire about specific details that are required for each agent as follows:
   - **calendar_agent**: User’s travel dates and any potential conflicts with Google Calendar events.
   - **data_retrieval_agent**: Budget and activity preferences to gather relevant data for the trip (e.g., flights, restaurants, local attractions).
   - **itinerary_agent**: Once data is gathered, this agent will create a personalized itinerary for the user.

4. Agent Routing: Based on the collected information, determine which agent to route the user to:
   - **calendar_agent**: Check for any calendar conflicts with the user’s travel dates and add calendar events to the google calendar.
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
3. **Chatbot**: "Noted! Now, could you share your budget and preferences for activities, restaurants, or accommodations?"
4. **User**: [Provides details]
5. **Chatbot**: "Let me check your calendar for any conflicts with these dates..."
6. **calendar_agent**: [Checks user’s Google Calendar for conflicts]
7. **Chatbot**: "I see that you have a meeting on March 12 at 3 PM. Would you like to adjust your travel dates or keep this event?"
8. **User**: "I'll keep it, but please add a reminder."
9. **Chatbot**: [Routes to `data_retrieval_agent` to fetch relevant data]
10. **data_retrieval_agent**: [Fetches data: restaurants, flight information, and local attractions in New Jersey]
11. **Chatbot**: "I’ve gathered all the details! Now I’m creating your personalized itinerary. Just a moment."
12. **itinerary_agent**: [Creates itinerary]
13. **Chatbot**: "Here's your personalized itinerary! Would you like to add anything else?"
14. **User**: "No, that's all. Thank you!"
15. **Chatbot**: "Would you like me to add this event to your Google Calendar?"
16. **User**: "Yes, please."
17. **Chatbot**: "Let me add this event your calendar.Event details : 2nd March - 5th March Trip to Miami from NewYork."
17. **calendar_agent**: [Adds events to Google Calendar]
18. **Chatbot**: "FINISH"
"""

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal['itinerary_agent','human_interrupt','calendar_agent','data_retrieval_agent','FINISH']
    messages:str

def chatbot_node(state:State) -> Command[Literal['human_interrupt','query_checker_module','calendar_agent','itinerary_agent','data_retrieval_agent','__end__']] :
    messages = [
        {"role": "system","content":chatbot_prompt}
    ] + state["message_list"]

    response = llm.with_structured_output(Router).invoke(messages)

    new_lst = state["message_list"] + [("ai", response["messages"])]
    
    goto = response["next"]

    if goto == "FINISH":
         goto = END

    if goto == "data_retrieval_agent" and state["query"]=="":
         goto = "query_checker_module"     
         
    return Command(goto=goto,update={"next": goto,"message_list": new_lst})       

# Initialize the state graph

builder = StateGraph(State)
builder.add_edge(START, "chatbot")
builder.add_node("chatbot", chatbot_node)
builder.add_node("itinerary_agent", itinerary_node)
builder.add_node("data_retrieval_agent", data_retrieval_node)
builder.add_node("calendar_agent", calendar_node)
builder.add_node("human_interrupt", human_interrupt)
builder.add_node("query_checker_module",query_checker_node)

# Compile the graph
graph = builder.compile()

# Initial state for the conversation
initial_state = {
    "message_list": [("user", "Hi")],
    "query":"",
    "fetched_data": "",
    "itinerary": "",
}

import pprint
# Start the graph stream to trigger the flow
for s in graph.stream(initial_state,subgraphs=True,stream_mode="values"):

        if "next" in s[1]:            
            print("message: " + s[1]['message_list'][-1][1])
            print("next_node: " + s[1]["next"])
            print("query:" + s[1]['query'])
            print("fetched_data:" + s[1]["fetched_data"])
            print("itinerary:" + s[1]["itinerary"])
            print("----")
            print("\n")
            print(s[1]["next"] + "\n")
        
        else:
            print("chatbot \n")    

