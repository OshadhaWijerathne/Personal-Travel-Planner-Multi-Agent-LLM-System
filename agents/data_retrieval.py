# agents/data_retrieval.py
"""
import pandas as pd

df = pd.read_csv("dataset_samples_20.csv")

def data_retrieval_agent():
    return df["reference_information"][0]
"""

from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import pandas as pd
from pandas import DataFrame
import re
import numpy as np


from config import llm  # Import the shared llm instance

def data_retrieval_agent(query):

    # Loading Data Sets 

    restaurents_data_path="./database/restaurants/clean_restaurant_2022.csv"
    restaurents_data = pd.read_csv(restaurents_data_path).dropna()[['Name','Average Cost','Cuisines','Aggregate Rating','City']]
    print("Restaurants data loaded.")

    flights_data_path = "./database/flights/clean_Flights_2022.csv"
    flights_data = pd.read_csv(flights_data_path).dropna()[['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime','FlightDate','OriginCityName','DestCityName','Distance']]
    print("Flights data loaded.")

    google_distance_data_path = "./database/googleDistanceMatrix/distance.csv"
    google_distance_data = pd.read_csv(google_distance_data_path)
    print("Google Distance matrix data loaded.")

    accommodations_data_path = "./database/accommodations/clean_accommodations_2022.csv"
    accommodations_data = pd.read_csv(accommodations_data_path).dropna()[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
    print("Accommodation data loaded.")

    attractions_data_path = "./database/attractions/attractions.csv"
    attractions_data = pd.read_csv(attractions_data_path).dropna()[['Name',"City"]]
    print("Attractions data loaded.")

    city_data_path = "./database/background/citySet_with_states.txt"
    cityStateMapping = open(city_data_path, "r").read().strip().split("\n")
    city_data = {}
    for unit in cityStateMapping:
        city, state = unit.split("\t")
        if state not in city_data:
            city_data[state] = [city]
        else:
            city_data[state].append(city)
    print("City data loaded.")

    notebook_data = []


    # Defining Tools

    def extract_before_parenthesis(s):
        match = re.search(r'^(.*?)\([^)]*\)', s)
        return match.group(1) if match else s

    @tool
    def FlightSearch(origin: str,
                destination: str,
                departure_date: str,
                ) -> dict:
            """
                Description: A flight information retrieval tool.
                Parameters:
                    Departure City: The city you'll be flying out from.
                    Destination City: The city you aim to reach.
                    Date: The date of your travel in YYYY-MM-DD format.
                Example: FlightSearch[New York, London, 2022-10-01] would fetch flights from New York to London on October 1, 2022.
            """
            #return {'FlightID': 1001, 'OriginCityName': 'Washington', 'DestCityName': 'Myrtle Beach', 'FlightDate': '2022-03-13', 'Price': 300}
            results = flights_data[flights_data["OriginCityName"] == origin]
            results = results[results["DestCityName"] == destination]
            results = results[results["FlightDate"] == departure_date]
            if len(results) == 0:
                return "There is no flight from {} to {} on {}.".format(origin, destination, departure_date)
            #return results.to_dict(orient='records')
            notebook_data.append({f"Flights data availavble from {origin} to {destination} on {departure_date}":results.to_dict(orient='records')})
            return f"Flights data availavble from {origin} to {destination} on {departure_date} and retreived succesfully"

    @tool
    def GoogleDistanceMatrix(origin, destination, mode='driving'):
            """
            Description: Estimate the distance, time, and cost between two cities.
            Parameters:
                Origin: The departure city of your journey.
                Destination: The destination city of your journey.
                Mode: The method of transportation. Choices include 'self-driving' and 'taxi'.
            Example: GoogleDistanceMatrix[Paris, Lyon, self-driving] would provide driving distance, time, and cost between Paris and Lyon.
            """
            #return {"mode": "driving", "origin": "Washington", "destination": "Myrtle Beach", "duration": "8 hours", "distance": "650 km", "cost": 130}
            origin = extract_before_parenthesis(origin)
            destination = extract_before_parenthesis(destination)
            info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
            response = google_distance_data[(google_distance_data['origin'] == origin) & (google_distance_data['destination'] == destination)]
            if len(response) > 0:
                    if response['duration'].values[0] is None or response['distance'].values[0] is None or response['duration'].values[0] is np.nan or response['distance'].values[0] is np.nan:
                        return "No valid information."
                    info["duration"] = response['duration'].values[0]
                    info["distance"] = response['distance'].values[0]
                    if 'driving' in mode:
                        info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                    elif mode == "taxi":
                        info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))
                    if 'day' in info["duration"]:
                        return "No valid information."
                    return f"{mode}, from {origin} to {destination}, duration: {info['duration']}, distance: {info['distance']}, cost: {info['cost']}"

            return f"{mode}, from {origin} to {destination}, no valid information."   

    @tool
    def AccommodationSearch(city: str,) -> dict:
            """
            Description: Discover accommodations in your desired city.
            Parameter: City - The name of the city where you're seeking accommodation.
            Example: AccommodationSearch[Rome] would present a list of hotel rooms in Rome.
            """
            #return {'AccommodationID': 2001, 'City': 'Myrtle Beach', 'Name': 'Ocean View Hotel', 'Price': 100}
            results = accommodations_data[accommodations_data["city"] == city]
            if len(results) == 0:
                return "There is no accomodations in this city."
            
            #return results.to_dict(orient='records')
            notebook_data.append({f"Accomodation data in {city} is available and retreived succesfully":results.to_dict(orient='records')})
            return f"Accomodation data in {city} is available and retreived succesfully"

    @tool
    def RestaurantSearch(city: str,) -> dict:
        """
        Description: Explore dining options in a city of your choice.
        Parameter: City - The name of the city where you're seeking restaurants.
        Example: RestaurantSearch[Tokyo] would show a curated list of restaurants in Tokyo.
        """
        #return {'RestaurantID': 3001, 'City': 'Myrtle Beach', 'Name': 'Seafood Paradise', 'Cuisine': 'Seafood', 'Price': 40}
        results = restaurents_data[restaurents_data["City"] == city]
        if len(results) == 0:
            return "There are no restaurants in this city."
        #return results.to_dict(orient='records')
        notebook_data.append({f"Restaurents in {city}":results.to_dict(orient='records')})
        return f"Restaurents data in {city} is avialble and retreived succesfully"

    @tool
    def AttractionSearch(city: str,) -> dict:
        """
        Description: Find attractions in a city of your choice.
        Parameter: City - The name of the city where you're seeking attractions.
        Example: AttractionSearch[London] would return attractions in London.
        """
        #return {'AttractionID': 4001, 'City': 'Myrtle Beach', 'Name': 'Myrtle Beach Boardwalk', 'Type': 'Outdoor', 'Price': 20}
        results = attractions_data[attractions_data["City"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There are no attractions in this city."
        #return results.to_dict(orient='records')
        notebook_data.append({f"Attractions in {city}":results.to_dict(orient='records')})
        return f"Atrractions data in {city} is available and retreived succesfully" 

    @tool
    def CitySearch(state: str,) -> dict:
            """
            Description: Find cities in a state of your choice.
            Parameter: State - The name of the state where you're seeking cities.
            Example: CitySearch[California] would return cities in California.
            """
            #return {'City': 'Myrtle Beach', 'Population': 35000, 'Country': 'USA'}
            if state not in city_data:
                return "Invalid State"
            else:
                results = city_data[state]
            #return results.to_dict(orient='records')
            notebook_data.append({f"These are the cities is in the state {state}":results})
            return f"These are the cities is in the state {state} , Cities : {results}"

    tools = [
        FlightSearch,
        GoogleDistanceMatrix,
        AccommodationSearch,
        RestaurantSearch,
        AttractionSearch,
        CitySearch,
    ]

    def data_retrieval_agent_prompt(query):
        
        instructions = f"""**Task:** Gather information for a travel plan using a series of steps: 'Thought', 'Action', and 'Observation'. The goal is **ONLY to collect information** for a travel plan. This is **NOT** about planning or making decisions.

        - **Thought:** Reason about the current situation.
        - **Action:** You can perform one of the following actions to gather information:
        1. **FlightSearch[Departure City, Destination City, Date]**
        2. **GoogleDistanceMatrix[Origin, Destination, Mode]**
        3. **AccommodationSearch[City]**
        4. **RestaurantSearch[City]**
        5. **AttractionSearch[City]**
        6. **CitySearch[State]**

        No tools can be used together in a nested way.

        ---

        **Query:** {query}
        """
        return instructions

    #query = "Please create a travel plan for me where I'll be departing from Washington and heading to Myrtle Beach for a 3-day trip from March 13th to March 15th, 2022. Can you help me keep this journey within a budget of $1,400?"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", data_retrieval_agent_prompt(query)),
            ("placeholder", "{messages}"),
        ]
    )

    langgraph_agent_executor = create_react_agent(model=llm, tools=tools,prompt=prompt)
    
    for step in langgraph_agent_executor.stream({"messages": [("human", query)]}):
        continue
        for key, value in step.items():
            print(key)
            #if key == "agent":
                #print(value["messages"])
                #continue
            print(value)
        print("---------------")
    
    #print(notebook_data[::-1])
    return notebook_data[::-1]




