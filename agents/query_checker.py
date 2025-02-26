from config import llm
from typing_extensions import TypedDict

# Define the output structure for query checker
class query_checker_output(TypedDict):
    query: str

# Prompt for the query builder agent
def query_checker_prompt(past_chat):
   query_checker_instructions = f"""

   Given the user's past chat history, extract the following details:
   - Origin (departure location)
   - Destination (arrival location)
   - Number of days (duration of the trip)

   Then, generate a query in the following format:

   "Please create a travel plan for me where I'll be departing from [origin] and heading to [destination] for a [number of days]-day trip from [start date] to [end date]. Can you help me keep this journey within a budget of [budget]?"

   Only provide the final query as output. Do not include any explanations or additional information.

   past chat histoty {past_chat}
   """
   return query_checker_instructions

def query_checker_module(past_chat):
    
    respond = llm.with_structured_output(query_checker_output).invoke(query_checker_prompt(past_chat))

    return respond["query"]
    
    
