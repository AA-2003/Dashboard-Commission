from requests.exceptions import RequestException, HTTPError, Timeout
from datetime import datetime, timedelta
from logging_config import setup_logger
logger = setup_logger()
import pandas as pd
import numpy as np
import requests
import time
import json 
import streamlit as st


source_id_map = {
    "f074191f-7e1e-47bd-aea7-e4c9a8c077b7": "پلت‌فرم",
    "2f6b1705-95d7-4c1d-8641-8df0de967f9c": "تماس ورودی (مشتری)",
    "00000000-0000-0000-0000-000000000001": "معرف",
    "68db6b22-2781-49c9-bfca-0990b67e42e5": "دایرکت اینستاگرام",
    "bfe23b44-98a9-47f4-a46e-db775381e111": "مهمان واسطه",
    "a64f917c-ea33-49a8-9690-4b24a3d76c09": "چت واتس‌اپ",
    "eed14412-2d64-4ce2-a30d-49ee43442a6a": "چت سایت",
    "1fb15d87-421b-4c2c-b849-4fb249c4422d": "تلگرام(سوشال)",
    "b0903161-c13b-4502-b07e-ec37034dc72b": "واتساپ(سوشال)",
    "00000000-0000-0000-0000-000000000000": "سایر",
    "7105a725-f738-4a60-a46d-f73cf0426825": "مهمان مدیرمجموعه",
    "95448f96-6e66-4d92-a51e-39b8e1cedd58": "چت تلگرام",
    "827d8663-4b11-4d2b-a137-cfc953ee8688": "تماس فرم سایت",

    "79cbe55a-6592-4867-9f09-8b46a76c6807": "پیامک فرم",
}

pipeline_stage_id_map = {
    'd17adff8-911a-47dd-9eca-27ebbe7fcbb1': 'مشاوره و پیگیری رزرو',
    'c6852948-e627-4a68-8427-fc14c46ca7f2': 'پیگیری پرداخت',
    '5f9786ac-5398-4a18-930b-a002dc0947a7': 'تاییدیه مالی',
    '817cacb3-dce2-4870-9488-58efbd9d9004': 'اطلاع‌رسانی / وچر',
    '342b87fc-741d-4d5e-ab81-c967f70354e6': 'مذاکره اولیه',
    '13f970a7-58b4-437d-999c-707736c5c49c': 'پیگیری',
    '6ea4b752-162e-473e-b2b9-d404966cb515': 'قیمت و نهایی کردن',
    '4e456956-6940-40b5-bba1-d4f60643f9ec': 'مذاکره/پرداخت (تیم فروش)',
    'f0b971a4-7063-4b72-87d1-3f500f2ad525': 'هماهنگی توزیع/ تحویل به مشتری',
    'e1450768-d71f-4bc1-9e2a-aed93a7546eb': 'مشتری‌های ثابت',
    '300c1df0-3277-4e66-ae6d-e0a3f3f44f90': 'مذاکره اولیه',
    'eb28d07d-f429-49b0-975b-1679a724abd1': 'قیمت و نهایی کردن',
    'dd0fa73e-fdb1-4460-89b2-1a959fcbeb4c': 'فرصت جدید',
    'c1788257-7dff-4fef-a2e8-f7e130039314': 'مذاکره اولیه',
    'ecbf85cd-498c-4889-b8d6-cf05c7a40d18': 'پیگیری',
    '748fa273-414a-448d-af7e-e9de50baef91': 'قیمت و نهایی کردن',
    '208adc77-96f7-4499-b549-1fb78d113894': 'سایر',
    'c276e29a-f45d-4366-89c3-0bf91056ccbd': 'سایر'
}

reversed_cols_map = {
    'Code': "deal_id", # کد دیدار معامله
    'Title': "deal_title", # عنوان معامله
    'Price': "deal_value", # ارزش معامله
    'Status': "deal_status", # وضعیت معامله
    'ChangeToWonTime': "deal_done_date", # تاریخ انجام معامله
    'RegisterTime': "deal_created_date", # تاریخ ایجاد معامله
    'Owner.DisplayName': "deal_owner", # مسئول معامله
    'SourceId': "deal_source", # شیوه آشنایی معامله
    'Contact.Code': "Customer_id", # ای دی مشتری
    'Items.Id': "product_code", # کد محصول
    'Items.Quantity': "product_quantity", # تعداد محصول
    'Items.Price': "product_price", # قیمت
    'Items.UnitPrice': "product_unit_price", # واحد قیمت محصول
    'Items.Discount': "product_discount", # میزان تخفیف محصول
    'Items.PriceAfterDiscount': "final_amount", # مبلغ نهایی
}

# Date conversion functions
def convert_to_iran_time(utc_time):
    if utc_time.endswith('Z'):
        utc_time = utc_time[:-1]
    dt = datetime.strptime(utc_time, "%Y-%m-%dT%H:%M:%S")
    dt += timedelta(hours=3, minutes=30)
    return dt.strftime("%Y-%m-%d %H:%M")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    """
    Preprocess the DataFrame by renaming columns, converting data types, and handling missing values.
    """

    try:
        logger.info("Start preprocessing loaded data")

        columns = df.columns

        # Drop unwanted rows by 'مسئول معامله'

        drop_operators = [
            "S.Hadi Cheheltani",
            "TECH TEAM",
            "آرا رنجبر",
            "امیرحسین جوادی",
            "حسین رشیدی زاده",
            "محمدرضا ایدرم",
            "فرزین سوری"
        ]
        if 'deal_owner' in columns:
            ops_lower = [op.lower() for op in drop_operators]
            df = df[
                ~df['deal_owner']
                    .astype(str).str.strip()
                    .str.lower().isin(ops_lower)
            ]
        else:
            raise KeyError("Column 'deal_owner' not found in the data.")
        
        # Handel date columns

        date_columns = [
            'deal_done_date', 'deal_created_date'
        ]
        for col in date_columns:
            if col in columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                KeyError(f"Column '{col}' not found in the data.")

        # Convert deal_value to numeric
        if 'deal_value' in columns:
            df['deal_value'] = pd.to_numeric(df['deal_value'], errors='coerce')
        else:
            KeyError("Column deal_value not found in the data.")

        logger.info(f"Data preprocessed successfully. Final shape: {df.shape}")

        return df.reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise 


# Get value from json
def get_nested_value(obj, path):
    """
        extract features from the json response row
    """
    keys = path.split('.')
    for key in keys:
        if obj is None:
            return None
        if key.startswith('Items') and isinstance(obj, dict) and obj.get('Items'):
            obj = obj['Items'][0]
            key = key.replace('Items.', '')
        elif '.' in key:
            if not isinstance(obj, dict):
                return None
            part1, part2 = key.split('.')
            obj = obj.get(part1, {})
            if obj is None:
                return None
            obj = obj.get(part2, '')
            return obj
        else:
            if not isinstance(obj, dict):
                return None
            obj = obj.get(key, '')
    return obj



def load_data(start_date: str = None, end_date: str = None, WON: bool = False) -> pd.DataFrame:
    """
    Loads deal data from the Didar API with pagination and retry mechanism.

    Args:
        start_date: Start date for filtering deals (YYYY-MM-DD).
        end_date: End date for filtering deals (YYYY-MM-DD).
        WON: If True, only fetch deals with Status 1 (WON).

    Returns:
        A pandas DataFrame containing the fetched deal data.

    Raises:
        ConnectionError: If a chunk fails to be fetched after multiple retries.
        ValueError: If the API response structure is unexpected or no data is found at all.
    """
    logger.info("Start loading data")
    
    # Get API key from environment variables
    API_KEY = st.secrets.get("DIDAR_API_KEY")['DIDAR_API_KEY'] 
    if not API_KEY:
        raise ValueError("DIDAR_API_KEY not found in environment variables")
        
    url = f"https://app.didar.me/api/deal/search?apikey={API_KEY}"

    all_deals = []
    offset = 0
    limit = 2000  
    delay_between_chunks = 2
    max_retries = 5 
    delay_on_retry = 5  
    request_timeout = 10 

    print(f"Getting data from {start_date} to {end_date}, WON status: {WON}")

    while True:
        payload = {
            'Criteria':{},
            "From": offset,
            "Limit": limit
        }
        if start_date:
            payload["Criteria"]["SearchFromTime"] = start_date

        if end_date:
            payload["Criteria"]["SearchToTime"] = end_date

        if WON:
            payload["Criteria"]["Status"] = 1

        attempt = 0
        while attempt < max_retries:
            try:
                print(f"Extracting data from API {offset} (try {attempt + 1}/{max_retries})...")
                response = requests.post(url, json=payload, timeout=request_timeout)
                response.raise_for_status()

                try:
                    data = response.json()
                except json.JSONDecodeError:
                    print(f"Attempt {attempt + 1}/{max_retries}: Invalid JSON response for offset {offset}. Retrying in {delay_on_retry} seconds...")
                    attempt += 1
                    time.sleep(delay_on_retry)
                    continue

                if not data or not data.get("Response") or not data["Response"].get("List"):
                    print(f"Warning: API returned empty or unexpected response structure for offset {offset}.")
                    current_chunk = []
                    break

                current_chunk = data["Response"]["List"]
                all_deals.extend(current_chunk)
                print(f"Success: Received {len(current_chunk)} deals. Total received so far: {len(all_deals)}")
                break

            except Timeout:
                print(f"Attempt {attempt + 1}/{max_retries}: Request timeout for offset {offset}. Retrying in {delay_on_retry} seconds...")
                attempt += 1
                time.sleep(delay_on_retry)
            except HTTPError as e:
                print(f"Attempt {attempt + 1}/{max_retries}: HTTP error occurred: {e} for offset {offset}. Retrying in {delay_on_retry} seconds...")
                attempt += 1
                time.sleep(delay_on_retry)
            except RequestException as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Network or request error: {e} for offset {offset}. Retrying in {delay_on_retry} seconds...")
                attempt += 1
                time.sleep(delay_on_retry)
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Unexpected error occurred: {e}. Retrying in {delay_on_retry} seconds...")
                attempt += 1
                time.sleep(delay_on_retry)

        if attempt == max_retries:
            print(f"Failed to fetch data chunk at offset {offset} after {max_retries} attempts.")
            raise ConnectionError(f"Unable to fetch data chunk at offset {offset} after {max_retries} attempts.")

        if len(current_chunk) < limit:
            print("Reached end of data.")
            break

        offset += limit
        print(f"Complete chunk received ({limit} records). Waiting {delay_between_chunks} seconds before next request...")
        time.sleep(delay_between_chunks)

    if not all_deals:
        print("No deals found matching the specified criteria.")
        return pd.DataFrame()

    rows = []
    for deal in all_deals:
        row = {}
        for key, header in reversed_cols_map.items():
            value = get_nested_value(deal, key)

            if key == 'SourceId':
                value = source_id_map.get(value, value)
            elif key in ['ChangeToWonTime', 'RegisterTime'] and value:
                value = convert_to_iran_time(value.split('.')[0])

            row[header] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info(f"Data loaded successfully with {len(df)} records.")
    try:
        preprocess(df)
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    from_date = (datetime.today() - timedelta(days=14)).strftime('%Y-%m-%d')
    to_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(load_data(from_date, to_date))