import streamlit as st
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# --- Configuration ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

# --- Identify your Spreadsheet ---
def authenticate_google_sheets():
    """
    Authenticates with Google Sheets API using credentials
    stored in Streamlit's secrets.
    """
    google_creds_object = st.secrets.get("GOOGLE_CREDENTIALS_JSON")

    if not google_creds_object:
        print("Secret 'GOOGLE_CREDENTIALS_JSON' not found in Streamlit secrets. Please configure it.")
        st.stop()
        return None

    try:
        creds_dict = dict(google_creds_object)
    except (TypeError, ValueError) as e:
        print(f"Could not convert the 'GOOGLE_CREDENTIALS_JSON' secret into a dictionary. Type received: {type(google_creds_object).__name__}. Error: {e}")
        st.stop()
        return None

    # **Crucial Step for private_key:**
    if "private_key" in creds_dict and isinstance(creds_dict["private_key"], str):
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    elif "private_key" not in creds_dict:
        print("The 'private_key' is missing from the 'GOOGLE_CREDENTIALS_JSON' secrets.")
        st.stop()
        return None
    else: # private_key exists but is not a string
        print(f"The 'private_key' in 'GOOGLE_CREDENTIALS_JSON' secrets is not a string. Type: {type(creds_dict['private_key']).__name__}")
        st.stop()
        return None

    # Validate that essential keys are present 
    required_keys = ["type", "project_id", "private_key_id", "client_email", "client_id", "auth_uri", "token_uri"]
    missing_keys = [key for key in required_keys if key not in creds_dict]
    if missing_keys:
        print(f"Essential key(s) {', '.join(missing_keys)} are missing from 'GOOGLE_CREDENTIALS_JSON' secrets.")
        st.stop()
        return None

    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"Google Sheets Authentication Error: {e}")
        st.stop()
        return None

def load_data_from_sheet(client, spreadsheet_id, sheet_name):
    """Loads data from the specified sheet into a Pandas DataFrame."""
    if not client:
        print("Authentication client not available. Cannot load data.")
        return None # Explicitly return None if client is None
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records() # Assumes first row is header

        if not data:
            print(f"No data found in sheet '{sheet_name}' or sheet might be empty (contains only a header or is completely blank).")
            return pd.DataFrame() 

        df = pd.DataFrame(data)
        return df

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Error: Spreadsheet with ID '{spreadsheet_id}' not found.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Worksheet '{sheet_name}' not found in the spreadsheet. Check the sheet name for typos.")
    except gspread.exceptions.APIError as e:
        print(f"Google Sheets API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading data from the sheet: {e}")
    return None # Return None in case of any error


@st.cache_data(ttl=600, show_spinner=False)
def load_sheet(SHEET_NAME: str='Data', spreadsheet: str = None) -> pd.DataFrame:
    # Authenticate and get gspread client
    gs_client = authenticate_google_sheets()

    if gs_client:
        if spreadsheet:
            current_spreadsheet_id = st.secrets.get("SPREADSHEET_ID")['EVAL_SPREADSHEET_ID']
        else:
            current_spreadsheet_id = st.secrets.get("SPREADSHEET_ID")['MAIN_SPREADSHEET_ID']

        if not current_spreadsheet_id:
            print("SPREADSHEET_ID not found in Streamlit secrets. Using hardcoded ID.")


        if not current_spreadsheet_id: # Should not happen if hardcoded, but good check if from secrets
            print("SPREADSHEET_ID is missing. Cannot load data.")
        else:
            print(f"Attempting to load data from Spreadsheet ID: {current_spreadsheet_id}, Sheet: {SHEET_NAME}")
            with st.spinner("بارگذاری داده ها ..."):
                df_main = load_data_from_sheet(gs_client, current_spreadsheet_id, SHEET_NAME)

            if df_main is not None: # Check if df_main is not None (means no critical error during load)
                if not df_main.empty:
                    print(f"Loaded {len(df_main)} rows and {len(df_main.columns)} columns.")
                    return df_main
                else:
                    print("The sheet was loaded, but it appears to be empty or contains only a header row.")

@st.cache_data(ttl=600, show_spinner=False)
def load_sheet_uncache(SHEET_NAME: str='Data', spreadsheet: str = None) -> pd.DataFrame:
    return load_sheet(SHEET_NAME, spreadsheet)

@st.cache_data(ttl=600, show_spinner=False)
def get_sheet_names(spreadsheet: str = None) -> list[str]:
    """
    Get sheet names of Google Spreadsheet
    """
    gs_client = authenticate_google_sheets()

    if gs_client:
        if spreadsheet:
            current_spreadsheet_id = st.secrets.get("SPREADSHEET_ID")['EVAL_SPREADSHEET_ID']
        else:
            current_spreadsheet_id = st.secrets.get("SPREADSHEET_ID")['MAIN_SPREADSHEET_ID']

        if not current_spreadsheet_id:
            print("SPREADSHEET_ID is missing. Cannot load data.")
            return []

        try:
            spreadsheet_obj = gs_client.open_by_key(current_spreadsheet_id)
            sheet_names = [ws.title for ws in spreadsheet_obj.worksheets()]
            print(f"Found sheets: {sheet_names}")
            return sheet_names
        except Exception as e:
            print(f"Failed to get sheet names: {e}")
            return []
        
if __name__ == "__main__":
    print('I hope this workes fine.')