# utils/logger.py
# simple logger setup for adding logs to logs sheet

import streamlit as st
from .sheetConnect import append_to_sheet, authenticate_google_sheets

PRODUCTION = st.secrets.get("PRODUCTION", False)
def log_event(user: str, event_type: str, message: str):
    """
    Log an event to the logs sheet.
    
    Args:
        user: Username associated with the event
        event_type: Type of event (e.g., 'login', 'error')
        message: Detailed message about the event
    """
    from datetime import datetime

    log_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user': user,
        'event_type': event_type,
        'message': message
    }
    spreadsheet_id = st.secrets.get("SPREADSHEET_IDS").get("MAIN_SPREADSHEET_ID")

    if PRODUCTION:
        print(f"LOG [{log_data['timestamp']}] - User: {user}, Type: {event_type}, Message: {message}")
    else:
        print(f"DEV LOG [{log_data['timestamp']}] - User: {user}, Type: {event_type}, Message: {message}")
        append_to_sheet(client=authenticate_google_sheets(), spreadsheet_id=spreadsheet_id, sheet_name='Logs', row_data=[log_data])