"""
Google Sheets connection utilities for Streamlit applications.

This module provides functions to authenticate with Google Sheets API
and load data into pandas DataFrames.
"""

from typing import Optional

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

# Google Sheets API scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file'
]

# Required credentials keys
REQUIRED_CREDENTIAL_KEYS = [
    "type", "project_id", "private_key_id", "private_key",
    "client_email", "client_id", "auth_uri", "token_uri"
]

def _validate_credentials(creds_dict: dict) -> tuple[bool, Optional[str]]:
    """
    Validate Google credentials dictionary.
    
    Args:
        creds_dict: Dictionary containing Google service account credentials
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for missing keys
    missing_keys = [key for key in REQUIRED_CREDENTIAL_KEYS if key not in creds_dict]
    if missing_keys:
        return False, f"Missing required keys: {', '.join(missing_keys)}"
    
    # Validate and fix private_key format
    if not isinstance(creds_dict["private_key"], str):
        return False, f"'private_key' must be a string, got {type(creds_dict['private_key']).__name__}"
    
    # Replace escaped newlines with actual newlines
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    
    return True, None


def authenticate_google_sheets() -> Optional[gspread.Client]:
    """
    Authenticate with Google Sheets API using Streamlit secrets.
    
    Returns:
        Authenticated gspread client or None if authentication fails
    """
    try:
        google_creds_object = st.secrets.get("GOOGLE_CREDENTIALS_JSON")
        
        if not google_creds_object:
            print("'GOOGLE_CREDENTIALS_JSON' not found in Streamlit secrets")
            st.stop()
            return None
        
        # Convert to dictionary
        try:
            creds_dict = dict(google_creds_object)
        except (TypeError, ValueError) as e:
            print(f"Failed to convert credentials to dict: {e}")
            st.stop()
            return None
        
        # Validate credentials
        is_valid, error_msg = _validate_credentials(creds_dict)
        if not is_valid:
            print(f"Credential validation failed: {error_msg}")
            st.stop()
            return None
        
        # Create credentials and authorize client
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client
        
    except Exception as e:
        print(f"Authentication error: {e}")
        st.stop()
        return None


def _get_spreadsheet_id(key: str ) -> Optional[str]:
    """
    Get spreadsheet ID from Streamlit secrets.
    
    Args:
        use_eval_sheet: If True, return EVAL_SPREADSHEET_ID, else MAIN_SPREADSHEET_ID
        
    Returns:
        Spreadsheet ID or None if not found
    """
    try:
        spreadsheet_ids = st.secrets.get("SPREADSHEET_IDS")
        if not spreadsheet_ids:
            print("'SPREADSHEET_ID' not found in Streamlit secrets")
            return None
        
        spreadsheet_id = spreadsheet_ids.get(key)
        
        if not spreadsheet_id:
            print(f"'{key}' not found in SPREADSHEET_ID configuration")
            return None
        
        return spreadsheet_id
        
    except Exception as e:
        print(f"Error retrieving spreadsheet ID: {e}")
        return None


def load_data_from_sheet(
    client: gspread.Client,
    spreadsheet_id: str,
    sheet_name: str
) -> Optional[pd.DataFrame]:
    """
    Load data from a Google Sheet into a pandas DataFrame.
    
    Args:
        client: Authenticated gspread client
        spreadsheet_id: Google Spreadsheet ID
        sheet_name: Name of the worksheet to load
        
    Returns:
        DataFrame containing sheet data, or None if loading fails
    """
    if not client:
        print("No authenticated client provided")
        return None
    
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # Get all values including headers
        all_values = worksheet.get_all_values()
        
        if not all_values or len(all_values) < 2:
            print(f"Sheet '{sheet_name}' is empty or contains only headers")
            return pd.DataFrame()
        
        # Use first row as headers and remaining rows as data
        headers = all_values[0]
        data_rows = all_values[1:]
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Replace empty strings with empty cells for consistency
        df = df.replace('', None)
        
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns from '{sheet_name}'")
        return df
        
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Spreadsheet with ID '{spreadsheet_id}' not found")
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{sheet_name}' not found")
    except gspread.exceptions.APIError as e:
        print(f"Google Sheets API error: {e}")
    except Exception as e:
        print(f"Unexpected error loading sheet data: {e}")
    
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_sheet(
    key: str,
    sheet_name: str = 'Data',
) -> Optional[pd.DataFrame]:
    """
    Load data from a Google Sheet with caching.
    
    Args:
        sheet_name: Name of the worksheet to load (default: 'Data')
        use_eval_spreadsheet: If True, use EVAL_SPREADSHEET_ID, else MAIN_SPREADSHEET_ID
        
    Returns:
        DataFrame containing sheet data, or None if loading fails
    """
    # Authenticate
    client = authenticate_google_sheets()
    if not client:
        return None
    
    # Get spreadsheet ID
    spreadsheet_id = _get_spreadsheet_id(key)
    if not spreadsheet_id:
        return None
    
    # Load data with spinner
    print(f"Loading sheet '{sheet_name}' from spreadsheet '{spreadsheet_id}'")
    with st.spinner("بارگذاری داده ها ..."):
        df = load_data_from_sheet(client, spreadsheet_id, sheet_name)
    
    if df is not None and df.empty:
        st.warning("Sheet loaded successfully but contains no data.")
    
    return df

def load_sheet_uncached(
    sheet_name: str = 'Data',
    use_eval_spreadsheet: bool = False
) -> Optional[pd.DataFrame]:
    """
    Load sheet without using cache (wrapper that clears cache before loading).
    
    Args:
        sheet_name: Name of the worksheet to load
        use_eval_spreadsheet: If True, use EVAL_SPREADSHEET_ID
        
    Returns:
        DataFrame containing sheet data
    """
    load_sheet.clear()
    return load_sheet(sheet_name, use_eval_spreadsheet)


def append_to_sheet(
    client: gspread.Client,
    spreadsheet_id: str,
    sheet_name: str,
    row_data: list
) -> bool:
    """
    Append a row of data to a Google Sheet.
    
    Args:
        client: Authenticated gspread client
        spreadsheet_id: Google Spreadsheet ID
        sheet_name: Name of the worksheet to append to
        row_data: List of values representing the row to append
        
    Returns:
        True if append is successful, False otherwise
    """
    if not client:
        print("No authenticated client provided")
        return False
    
    try:
        # Flatten row_data if it's a nested list
        if row_data and isinstance(row_data, list):
            # If row_data is a list of lists, flatten it
            if isinstance(row_data[0], (list, tuple)):
                row_data = row_data[0]
            # If row_data contains dicts, convert to list of values
            elif isinstance(row_data[0], dict):
                row_data = list(row_data[0].values())
        
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        worksheet.append_row(row_data, value_input_option='USER_ENTERED')
        print(f"Appended row to '{sheet_name}'")
        return True
        
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Spreadsheet with ID '{spreadsheet_id}' not found")
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{sheet_name}' not found")
    except gspread.exceptions.APIError as e:
        print(f"Google Sheets API error: {e}")
    except Exception as e:
        print(f"Unexpected error appending to sheet: {e}")
    return False

@st.cache_data(ttl=600, show_spinner=False)
def get_sheet_names(
    spreadsheet_key: str,
) -> Optional[list[str]]:
    """
    Get the names of all sheets in a Google Spreadsheet.
    
    Args:
        spreadsheet_key: Key to identify which spreadsheet ID to use.
    Returns:
        List of sheet names or None if retrieval fails
    """

    client = authenticate_google_sheets()
    if not client:
        return None
    spreadsheet_id = _get_spreadsheet_id(spreadsheet_key)
    if not spreadsheet_id:
        return None
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
        sheet_names = [sheet.title for sheet in spreadsheet.worksheets()]
        print(f"Retrieved {len(sheet_names)} sheet names from spreadsheet '{spreadsheet_id}'")
        return sheet_names
    except Exception as e:
        print(f"Error retrieving sheet names: {e}")
        return None
    return None

def write_df_to_sheet(
    client: gspread.Client,
    spreadsheet_id: str,
    sheet_name: str,
    df: pd.DataFrame,
    clear_existing: bool = False
) -> bool:
    """
    Write a pandas DataFrame to a Google Sheet, replacing existing content.
    
    Args:
        client: Authenticated gspread client
        spreadsheet_id: Google Spreadsheet ID
        sheet_name: Name of the worksheet to write to
        df: DataFrame to write
    Returns:
        True if write is successful, False otherwise
    """
    if not client:
        print("No authenticated client provided")
        return False
    
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        if clear_existing:
            worksheet.clear()
        worksheet.update([df.columns.values.tolist()] + df.values.tolist(), value_input_option='USER_ENTERED')
        print(f"Wrote DataFrame to '{sheet_name}'")
        return True
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Spreadsheet with ID '{spreadsheet_id}' not found")
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{sheet_name}' not found")
    except gspread.exceptions.APIError as e:
        print(f"Google Sheets API error: {e}")
    except Exception as e:
        print(f"Unexpected error writing to sheet: {e}")
    return False