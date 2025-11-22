from io import BytesIO
import pandas as pd
import streamlit as st
from utils.logger import log_event
from utils.sheetConnect import load_sheet

def get_username() -> str:
    """Get current username for logging."""
    try:
        return st.session_state.get('userdata', {}).get('name', 'unknown')
    except Exception:
        return 'unknown'
    
@st.cache_data(ttl=10, show_spinner=False)
def convert_df(df):
    """
    Convert a DataFrame to CSV bytes for download.
    Cached to avoid recomputation.
    """
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data(ttl=600, show_spinner=False)
def convert_df_to_excel(df: pd.DataFrame):
    """
    Convert a DataFrame to an Excel file in memory (BytesIO).
    Handles timezone-aware datetime columns by localizing to naive.
    """
    # Remove timezone info from datetime columns (Excel does not support tz-aware datetimes)
    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_localize(None)

    output = BytesIO()
    # Write DataFrame to Excel in memory
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output

@st.cache_data(ttl=600, show_spinner=False)
def load_data_cached(spreadsheet_key: str, sheet_name: str) -> pd.DataFrame:
    """Load data with caching."""
    if spreadsheet_key:
        try:
            return load_sheet(key=spreadsheet_key, sheet_name=sheet_name)
        except Exception:
            return 
        
def handel_errors(e: Exception, message: str = "Unhandled error", show_error: bool = True, raise_exception: bool = False):
    """
    Handle errors by logging and optionally displaying an error message.
    """
    log_event(get_username(), 'error', f"{message}: {e}")

    if show_error:
        st.error("خطای رخ داده است. لطفا با ادمین تماس بگیرید.")

    production = st.secrets.get('GENERAL', {}).get("PRODUCTION", False)

    if production and raise_exception:
        raise e    

def download_buttons(dataframe: pd.DataFrame, base_filename: str):
    """
    Provide download buttons for CSV and Excel formats.

    Args:
        dataframe: DataFrame to be downloaded.
        base_filename: Base name for the downloaded files.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.download_button( 
            label="دانلود به صورت CSV",
            data=convert_df(dataframe),
            file_name=f'{base_filename}.csv',
            mime='text/csv',
            key=f'download_csv_{base_filename}',
        )
    with col2:
        st.download_button(
            label="دانلود به صورت اکسل",
            data=convert_df_to_excel(dataframe),
            file_name=f'{base_filename}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key=f'download_excel_{base_filename}',
        )