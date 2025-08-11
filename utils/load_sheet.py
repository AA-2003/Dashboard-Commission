import streamlit as st
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from utils.logging_config import setup_logger
logger = setup_logger()


# --- Configuration ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

# --- Identify your Spreadsheet ---
# It's better to get SPREADSHEET_ID from secrets as well,
# but using the hardcoded one as per your provided code for now.

def authenticate_google_sheets():
    """
    Authenticates with Google Sheets API using credentials
    stored in Streamlit's secrets.
    """
    # Get the credentials object from Streamlit secrets.
    # If the secret "GOOGLE_CREDENTIALS_JSON" contains a valid JSON structure,
    # Streamlit likely parses it into a dictionary-like object (AttrDict).
    google_creds_object = st.secrets.get("GOOGLE_CREDENTIALS_JSON")

    if not google_creds_object:
        logger.error("Secret 'GOOGLE_CREDENTIALS_JSON' not found in Streamlit secrets. Please configure it.")
        st.error("Secret 'GOOGLE_CREDENTIALS_JSON' not found in Streamlit secrets.")
        st.stop()
        return None

    # The object from st.secrets is likely an AttrDict, which is dictionary-like.
    # Convert it to a standard Python dictionary.
    try:
        creds_dict = dict(google_creds_object)
    except (TypeError, ValueError) as e:
        logger.error(f"Could not convert the 'GOOGLE_CREDENTIALS_JSON' secret into a dictionary. Type received: {type(google_creds_object).__name__}. Error: {e}")
        st.info("Ensure the GOOGLE_CREDENTIALS_JSON secret in Streamlit Cloud is a valid JSON object or a TOML table.")
        st.stop()
        return None

    # **Crucial Step for private_key:**
    # The private key string from secrets might have literal '\\n' characters
    # if it was parsed from a JSON string. These need to be replaced with actual newline characters '\n'.
    if "private_key" in creds_dict and isinstance(creds_dict["private_key"], str):
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    elif "private_key" not in creds_dict:
        logger.error("The 'private_key' is missing from the 'GOOGLE_CREDENTIALS_JSON' secrets.")
        st.error("Problem in secrets keys")
        st.stop()
        return None
    else: # private_key exists but is not a string
        logger.error(f"The 'private_key' in 'GOOGLE_CREDENTIALS_JSON' secrets is not a string. Type: {type(creds_dict['private_key']).__name__}")
        st.stop()
        return None

    # Validate that essential keys are present (optional but good practice)
    required_keys = ["type", "project_id", "private_key_id", "client_email", "client_id", "auth_uri", "token_uri"]
    missing_keys = [key for key in required_keys if key not in creds_dict]
    if missing_keys:
        logger.error(f"Essential key(s) {', '.join(missing_keys)} are missing from 'GOOGLE_CREDENTIALS_JSON' secrets.")
        st.stop()
        return None

    try:
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        logger.error(f"Google Sheets Authentication Error: {e}")
        st.stop()
        return None

def load_data_from_sheet(client, spreadsheet_id, sheet_name):
    """Loads data from the specified sheet into a Pandas DataFrame."""
    if not client:
        logger.error("Authentication client not available. Cannot load data.")
        return None # Explicitly return None if client is None
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records() # Assumes first row is header

        if not data:
            logger.warning(f"No data found in sheet '{sheet_name}' or sheet might be empty (contains only a header or is completely blank).")
            st.warning(f"No data found in she")
            return pd.DataFrame() 

        df = pd.DataFrame(data)
        return df

    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Error: Spreadsheet with ID '{spreadsheet_id}' not found.")
        st.info("Please check the SPREADSHEET_ID and ensure the service account (client_email from your secrets) has been granted at least 'Viewer' permission on the Google Sheet.")
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Error: Worksheet '{sheet_name}' not found in the spreadsheet. Check the sheet name for typos.")
    except gspread.exceptions.APIError as e:
        st.error(f"Google Sheets API Error: {e}")
        st.info("This could be due to permission issues (ensure service account has access to the sheet), incorrect sheet ID, or API quota limits.")
        st.exception(e)
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data from the sheet: {e}")
        st.exception(e)
    return None # Return None in case of any error

def load_sheet(SHEET_NAME='Data') -> pd.DataFrame:
    # Authenticate and get gspread client
    gs_client = authenticate_google_sheets()

    if gs_client:
        logger.info("Successfully authenticated with Google Sheets!")

        # Consider moving SPREADSHEET_ID to secrets for better practice
        current_spreadsheet_id = st.secrets.get("SPREADSHEET_ID")['SPREADSHEET_ID']
        if not current_spreadsheet_id:
            logger.warning("SPREADSHEET_ID not found in Streamlit secrets. Using hardcoded ID.")
        else:
            current_spreadsheet_id = current_spreadsheet_id

        if not current_spreadsheet_id: # Should not happen if hardcoded, but good check if from secrets
            logger.error("SPREADSHEET_ID is missing. Cannot load data.")
        else:
            logger.info(f"Attempting to load data from Spreadsheet ID: {current_spreadsheet_id}, Sheet: {SHEET_NAME}")
            with st.spinner(f"بارگذاری داده ها ..."):
                df_main = load_data_from_sheet(gs_client, current_spreadsheet_id, SHEET_NAME)

            if df_main is not None: # Check if df_main is not None (means no critical error during load)
                if not df_main.empty:
                    logger.info(f"Loaded {len(df_main)} rows and {len(df_main.columns)} columns.")
                    return df_main
                else:
                    logger.info("The sheet was loaded, but it appears to be empty or contains only a header row.")

if __name__ == "__main__":
    load_sheet()