import streamlit as st
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials


# --- Configuration ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

def authenticate_google_sheets():
    """
    Authenticates with Google Sheets API using credentials
    stored in Streamlit's secrets.
    """
    google_creds_object = st.secrets.get("GOOGLE_CREDENTIALS_JSON")

    if not google_creds_object:
        print("Secret 'GOOGLE_CREDENTIALS_JSON' not found in Streamlit secrets. Please configure it.")
        st.stop()


    try:
        creds_dict = dict(google_creds_object)
    except (TypeError, ValueError) as e:
        print(f"Could not convert the 'GOOGLE_CREDENTIALS_JSON' secret into a dictionary. Type received: {type(google_creds_object).__name__}. Error: {e}")
        st.stop()

    if "private_key" in creds_dict and isinstance(creds_dict["private_key"], str):
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    elif "private_key" not in creds_dict:
        print("The 'private_key' is missing from the 'GOOGLE_CREDENTIALS_JSON' secrets.")
        st.stop()
    else:
        print(f"The 'private_key' in 'GOOGLE_CREDENTIALS_JSON' secrets is not a string. Type: {type(creds_dict['private_key']).__name__}")
        st.stop()
    
    required_keys = ["type", "project_id", "private_key_id", "client_email", "client_id", "auth_uri", "token_uri"]
    missing_keys = [key for key in required_keys if key not in creds_dict]
    if missing_keys:
        print(f"Essential key(s) {', '.join(missing_keys)} are missing from 'GOOGLE_CREDENTIALS_JSON' secrets.")
        st.stop()
    
    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"Google Sheets Authentication Error: {e}")
        st.stop()
    
def write_df_to_sheet(df, spreadsheet_key, sheet_name='test', clear_sheet=True):
    """
    Write a pandas DataFrame to a Google Sheet.
    Args:
        df (pd.DataFrame): DataFrame to write.
        sheet_name (str): Name of the sheet/tab to write to.
        clear_sheet (bool): If True, clear the sheet before writing.
    Returns:
        bool: True if successful, False otherwise.
    """
    gs_client = authenticate_google_sheets()
    if not gs_client:
        return False

    spreadsheet_id = st.secrets.get("SPREADSHEET_IDS")[spreadsheet_key]
    if not spreadsheet_id:
        print("SPREADSHEET_ID is missing in Streamlit secrets.")
        return False

    try:
        spreadsheet = gs_client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        # If worksheet does not exist, create it
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="26")
    except Exception as e:
        print(f"Error accessing worksheet: {e}")
        return False

    try:
        if clear_sheet:
            worksheet.clear()

        # Prepare data: first row is header, then values
        values = [df.columns.tolist()] + df.astype(str).values.tolist()
        worksheet.update('A1', values)
        print(f"Successfully wrote {len(df)} rows and {len(df.columns)} columns to sheet '{sheet_name}'.")
        return True
    except Exception as e:
        print(f"Error writing DataFrame to sheet: {e}")
        return False

def append_to_sheet(row, sheet_name):
    """
    Appends a row to the end of a Google Sheet.

    Args:
        row (list): List of values to append as a row.
        sheet_name (str): Name of the sheet/tab to append to.

    Returns:
        bool: True if successful, False otherwise.
    """
    gs_client = authenticate_google_sheets()
    if not gs_client:
        return False

    spreadsheet_id = st.secrets.get("SPREADSHEET_IDS")['REQ_SPREADSHEET_ID']
    if not spreadsheet_id:
        print("SPREADSHEET_ID is missing in Streamlit secrets.")
        return False

    try:
        spreadsheet = gs_client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        # If worksheet does not exist, create it
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="5")
    except Exception as e:
        print(f"Error accessing worksheet: {e}")
        return False

    try:
        worksheet.append_row(row)
        print(f"Successfully appended row to sheet '{sheet_name}'.")
        return True
    except Exception as e:
        print(f"Error appending row to sheet: {e}")
        return False
    

# Example usage:
if __name__ == "__main__":
    data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Score": [90, 85, 78]
    }
    df = pd.DataFrame(data)
    success = write_df_to_sheet(df, sheet_name="TestSheet")
    if success:
        print("DataFrame written to Google Sheet successfully.")
    else:
        print("Failed to write DataFrame to Google Sheet.")