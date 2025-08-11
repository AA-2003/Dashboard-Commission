import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
import numpy as np
from utils.write_sheet import write_df_to_sheet
from utils.load_sheet import load_sheet



# calculate wolfs
def cal_wolfs(df: pd.DataFrame) -> pd.DataFrame:
    """functions for calculate wolfs - filter for current Jalali month"""
    # Convert deal_created_date to datetime
    df['deal_created_date'] = pd.to_datetime(df['deal_created_date'])

    # Merge "محمد آبساران/روز" and "محمد آبساران/شب" as one person
    def normalize_owner(owner):
        if owner in ["محمد آبساران/روز", "محمد آبساران/شب"]:
            return "محمد آبساران"
        return owner

    df['deal_owner'] = df['deal_owner'].apply(normalize_owner)

    # Filter for current Jalali month
    df['jalali_date'] = df['deal_created_date'].apply(lambda x: jdatetime.date.fromgregorian(date=x.date()))
    # Get current Jalali year and month
    today_jalali = jdatetime.date.today()
    current_jalali_year = today_jalali.year
    current_jalali_month = today_jalali.month
    df = df[ 
            (df['jalali_date'].apply(lambda d: d.year) == current_jalali_year) & 
            (df['jalali_date'].apply(lambda d: d.month) == current_jalali_month) &
            (df['deal_type'] == "New Sale")
            ]
    
    df['Date'] = df['deal_created_date'].dt.date

    # Sum deal_value for each person per day
    daily_sum = df.groupby(['Date', 'deal_owner'])['deal_value'].sum().reset_index()

    scores = {}

    all_owners = df['deal_owner'].unique()

    # For each day, rank and assign scores
    for day, group in daily_sum.groupby('Date'):
        group_sorted = group.sort_values('deal_value', ascending=False).reset_index(drop=True)
        owners_today = group_sorted['deal_owner'].tolist()
        for idx, row in group_sorted.iterrows():
            owner = row['deal_owner']
            # Scoring: 1st=3, 2nd=2, 3rd=1, last=-1, others=0
            if idx == 0:
                score = 3
            elif idx == 1:
                score = 2
            elif idx == 2:
                score = 1
            elif idx == len(group_sorted) - 1:
                score = -1
            else:
                score = 0
            scores[owner] = scores.get(owner, 0) + score

    # Build final list of owners and their scores
    wolf_board = [{'deal_owner': owner, 'score': scores.get(owner, 0)} for owner in all_owners]
    
    return pd.DataFrame(wolf_board).sort_values('score', ascending=False).reset_index(drop=True)

# load_sherlock data
def load_sherlock_data() -> pd.DataFrame:
    """"""
    
    sherlock_df = load_sheet('Sherlock')
    sherlock_df['Date'] = pd.to_datetime(sherlock_df['Date'])

    # Add Jalali date column
    sherlock_df['jalali_date'] = sherlock_df['Date'].apply(lambda x: jdatetime.date.fromgregorian(date=x.date()))
    today_jalali = jdatetime.date.today()
    current_jalali_year = today_jalali.year
    current_jalali_month = today_jalali.month

    # Filter for current Jalali month
    sherlock_df = sherlock_df[
        (sherlock_df['jalali_date'].apply(lambda d: d.year) == current_jalali_year) &
        (sherlock_df['jalali_date'].apply(lambda d: d.month) == current_jalali_month)
    ]

    # Prepare scores
    scores = {}

    # For each day, assign points: 1st=10, 2nd=5, 3rd=-3
    for day, group in sherlock_df.groupby('Date'):
        # Get persons for the day
        if group['First person'].values[0] == '':
            continue
        first = group['First person'].dropna().values
        second = group['Second person'].dropna().values
        third = group['Last person'].dropna().values

        # If multiple rows per day, count all
        for p in first:
            scores[p] = scores.get(p, 0) + 10
        for p in second:
            scores[p] = scores.get(p, 0) + 5
        for p in third:
            scores[p] = scores.get(p, 0) - 3
    
    # Build DataFrame
    result = pd.DataFrame([
        {'person': person, 'score': score}
        for person, score in scores.items()
    ]).sort_values('score', ascending=False).reset_index(drop=True)

    return result

def sales():
    """sales team dashboard with optimized metrics and visualizations"""
    st.title("📊 داشبورد تیم Sales")
    
    if not all(key in st.session_state for key in ['username', 'role', 'data', 'team', 'auth']):
        st.error("لطفا ابتدا وارد سیستم شوید")
        return

    # Initialize data and variables
    role = st.session_state.role
    username = st.session_state.username
    name = st.session_state.name

    st.write(f"{name}  عزیز خوش آمدی😃")    

    if role in ["admin", "manager"]:
        data = st.session_state['data']  
        data = data[data['team']=='sales'].copy()  
        wolf_board = cal_wolfs(data.copy())
        sherlock_board = load_sherlock_data()

        tabs = st.tabs(['صفحه اصلی', 'گرگ وال استریت', 'شرلوک', 'تنظیمات'])
        with tabs[0]:
            st.write(data)

        with tabs[1]: 
            st.write(wolf_board)

        with tabs[2]:
            st.write(sherlock_board)

        # setting tabs for set new values for parameters
        with tabs[3]:
            col1, col2 = st.columns(2)
            with col1:
                target = st.number_input(label="تارگت این ماه:", key='target_number', format="%f", step=1.0, value=0.0)
                reward_percent = st.number_input(label="درصد پاداش این ماه:", key='reward_percent', format="%f", step=1.0, max_value=100.0, value=0.0)
                wolf1_percent = st.number_input(label="درصد گرگ اول:", key='wolf1_percent', format="%f", step=1.0, max_value=100.0, value=0.0)
            with col2:
                wolf2_percent = st.number_input(label="درصد گرگ دوم:", key='wolf2_percent', format="%f", step=1.0, max_value=100.0, value=0.0)
                sherlock_percent = st.number_input(label="درصد شرلوک:", key='sherlock_percent', format="%f", step=1.0, max_value=100.0, value=0.0)
                performance_percent = st.number_input(label="درصد عملکرد:", key='performance_percent', format="%f", step=1.0, max_value=100.0, value=0.0)

            if st.button('تنظیم مجدد', key='write'):
                if sum([wolf1_percent, wolf2_percent, sherlock_percent, performance_percent]) != 100:
                    st.warning("جمع درصد ها با 100 برابر نیست!!!")
                else:
                    df = pd.DataFrame([{
                        "Target": target,
                        "Reward percent": reward_percent,
                        "Wolf1": wolf1_percent,
                        "Wolf2": wolf2_percent,
                        "Sherlock": sherlock_percent,
                        "Performance": performance_percent
                    }])
                    success = write_df_to_sheet(df, sheet_name='Sales team parameters')
                    if success:
                        st.info("پارامترها با موفقیت آپدیت شد.")
                    else:
                        st.info("خطا در آپدیت پارامترها!!!")

    else:
        pass
