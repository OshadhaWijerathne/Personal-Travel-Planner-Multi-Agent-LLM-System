# agents/calendar.py
"""
def calendar_agent():
    return "User's given dates are available."
"""
import os
import datetime
import pytz
import json
import dateparser
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

from config import llm  # Import the shared llm instance

def calendar_agent(query):

    # Google Calendar API Authentication
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    default_path = "./google_credentials/"
    SERVICE_ACCOUNT_FILE = default_path + "credentials.json"  # Ensure this file is uploaded

    def authenticate_google_calendar():
        creds = None
        if os.path.exists(default_path+"token.json"):
            creds = Credentials.from_authorized_user_file(default_path+"token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    SERVICE_ACCOUNT_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(default_path+"token.json", "w") as token:
                token.write(creds.to_json())
        return build('calendar', 'v3', credentials=creds)

    service = authenticate_google_calendar()

    # Get Today's Date
    today_date = datetime.datetime.today().strftime("%B %d, %Y")

    # Function to add an event in Google Calendar
    @tool
    def create_calendar_event(event_details:str) -> str:

        """Creates an event in Google Calendar. Provide event details as a dictionary with keys: "
            "'summary' (title), 'description' (details), 'start_time' (ISO 8601 format), "
            "'end_time' (ISO 8601 format), and 'timezone' (e.g., 'UTC').
            If the user doesn't specify a year, default to the year 2025.
            """
        # Ensure event_details is a dictionary
        if isinstance(event_details, str):
            try:
                #print("Original JSON string:", event_details,type(event_details))  # Debug: Print the original JSON string
                event_details = json.loads(event_details.replace("'", '"'))
                #print("Converted dictionary:", event_details) 
            except json.JSONDecodeError:
                return "❌ Error, Provided event details are not in valid JSON format."

        # Extract details safely from dictionary
        summary = event_details.get("summary", "No Title")
        description = event_details.get("description", "No Description")
        start_time = event_details.get("start_time")
        end_time = event_details.get("end_time")
        timezone = "Asia/Colombo"

        if not start_time or not end_time:
            return "❌ Error: Missing 'start_time' or 'end_time'."

        #print(f"📅 Creating event: {summary} from {start_time} to {end_time} in {timezone}")

        # Convert time to Google Calendar's format
        start_dt = datetime.datetime.fromisoformat(event_details['start_time'])
        end_dt = datetime.datetime.fromisoformat(event_details['end_time'])

        event = {
            'summary': summary,
            #'location': '800 Howard St., San Francisco, CA 94103',
            'description': description,
            'start': {
                "dateTime": start_dt.isoformat(),
                "timeZone": timezone
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": timezone
            }
        }

        try:
            event_result = service.events().insert(calendarId="primary", body=event).execute()
            #print(event_result)
            event_link = event_result.get('htmlLink')
            #print('Event created:', event_link)
            return str(event_link)
        except Exception as e:
            #print(f"An error occurred: {e}")
            return str("Error")

    # Functions to check availability in Google Calendar
    def format_to_iso(date_str, time_of_day="start"):

        parsed_date = dateparser.parse(date_str)
        
        if parsed_date is None:
            raise ValueError(f"Invalid date string: {date_str}")
        
        # Set the timezone to Sri Lankan time (Asia/Colombo)
        sri_lanka_tz = pytz.timezone("Asia/Colombo")

        if time_of_day == "start":
            formatted_date = parsed_date.replace(hour=0, minute=0, second=0, tzinfo=sri_lanka_tz)
        else:
            formatted_date = parsed_date.replace(hour=23, minute=59, second=59, tzinfo=sri_lanka_tz)

        return formatted_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    @tool
    def check_availability(start_date, end_date):
        """Check user's availability based on Google Calendar events for given travel dates.
        If the user doesn't specify a year, default to the year 2025."""
        
        start_iso = format_to_iso(start_date, "start")
        end_iso =  format_to_iso(end_date, "end")
  
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_iso,
            timeMax=end_iso,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get("items", [])
        if not events:
            return "No events during the given time period."
        return events

    tools = [create_calendar_event,check_availability]

    prompt = f"""You are a calendar assistant designed to help manage the user's schedule. The user can make one of two types of requests:

    1. **Check Availability**:  
    - The user provides a start and end date.  
    - Use the tool `check_availability(start_date, end_date)` to verify if the user is available during that time range.  
    - If the user does not provide dates, ask them for the required time range.

    2. **create_calendar_event**:  
    - The user provides the event details.
    - Then use the tool `create_calendar_event(event_details)` to add the event to the calendar.  
    - **Do not** check availability before adding an event, as the user has already confirmed the time is free.  

    Your role is to assist the user in scheduling tasks and managing their calendar efficiently.
    """

    calendar_agent = create_react_agent(model=llm,tools=tools,prompt=prompt)

    """for step in calendar_agent.stream({"messages": [("human", query)]}):
        for key, value in step.items():
            print(key)
            if key == "agent":
                print(value["messages"])
                continue
            print(value)
        print("---------------")"""
    for step in calendar_agent.stream({"messages": [("human", query)]},stream_mode="debug"):
        continue
    #print("Last Message: ",step["messages"][-1].content)
    return step["messages"][-1].content
           