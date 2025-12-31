import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
from typing import Optional, Dict, List
from utils.sheetConnect import get_sheet_names
from utils.funcs import  handel_errors
from utils.custom_css import apply_custom_css
from utils.sidebar import render_sidebar
from utils.sheetConnect import write_df_to_sheet, authenticate_google_sheets, load_sheet


# --- Constants ---
MONTH_NAMES = {
    '01': 'فروردین', '02': 'اردیبهشت', '03': 'خرداد',
    '04': 'تیر', '05': 'مرداد', '06': 'شهریور',
    '07': 'مهر', '08': 'آبان', '09': 'آذر',
    '10': 'دی', '11': 'بهمن', '12': 'اسفند'
    
}


def safe_to_jalali(gregorian_date):
    y, m, d = gregorian_date.year, gregorian_date.month, gregorian_date.day
    try:
        jalali_date = jdatetime.date.fromgregorian(day=d, month=m, year=y)
        return jalali_date.strftime("%Y/%m/%d")
    except Exception:
        return ""


# ========================================
# =========== Main Application ===========
# ========================================
def sales():
    """Main sales dashboard function."""
    apply_custom_css()
    render_sidebar()
    
    # Check authentication
    if not st.session_state.get('logged_in'):
        st.warning("لطفا ابتدا وارد سیستم شوید")
        return
    
    st.title("📊 داشبورد تیم Sales")
    
    # Check access
    userdata = st.session_state.get('userdata', {})
    teams = [t.strip() for t in userdata.get('team', '').split('|')]
    role = st.session_state.userdata.get('role', '')
    username_in_didar = st.session_state.userdata.get('username_in_didar', '')

    if 'sales' not in teams:
        st.error("شما به این بخش دسترسی ندارید")
        return
    
    # Get team members
    team_users = st.session_state.all_teams_users[
        st.session_state.all_teams_users['team'].apply(
            lambda x: 'Sales' in [t.strip() for t in x.split('|')]
        ) & (st.session_state.all_teams_users['role'] != 'admin')
    ]

    team_members = team_users['pms_name'].tolist()
    team_member_names = team_users['name'].tolist()

    try:
        # pms deals and records
        pms_reservetions = load_sheet(key='PMS_SPREADSHEET_ID', sheet_name='PMS_recent_deals')
        monthly_records = load_sheet(key='PMS_SPREADSHEET_ID', sheet_name='month_records')
        daily_records = load_sheet(key='PMS_SPREADSHEET_ID', sheet_name='Record_Performance')

        daily_records['Date'] = pd.to_datetime(daily_records['Date']).dt.date
        daily_records['Ammount_so_far'] = daily_records['Ammount_so_far'].astype(int)
        daily_records['Target'] = daily_records['Target'].astype(int)
        daily_records['jalali_date'] = daily_records['Date'].apply(safe_to_jalali)
        daily_records['Gap_to_target'] = daily_records['Target'] - daily_records['Ammount_so_far']
        daily_records['Gap_to_target'] = daily_records['Gap_to_target'].apply(lambda x: 0 if x > 0 else abs(x)).astype(int)
        daily_records['reward'] = daily_records.apply(
            lambda row: (row['Ammount_so_far'] - row['Target']) * 0.2 if row['Ammount_so_far'] >= row['Target'] else 0,
            axis=1
        ).astype(int)

        monthly_records['date'] = pd.to_datetime(monthly_records['first_date']).dt.date
        monthly_records['total_revenue'] = monthly_records['total_revenue'].astype(int)
        monthly_records['target'] = monthly_records['target'].astype(int)


        pms_reservetions['created_at'] = pd.to_datetime(pms_reservetions['created_at'], utc=True).dt.tz_convert('Asia/Tehran')
        pms_reservetions['total_nights'] = pd.to_numeric(pms_reservetions['total_nights'], errors='coerce').fillna(0).astype(int)
        pms_reservetions['last_status'] = pd.to_numeric(pms_reservetions['last_status'], errors='coerce').fillna(0).astype(int)

    except Exception as e:
        handel_errors(e, "خطا در بارگذاری داده‌های PMS", show_error=True, raise_exception=True)


    pms_reservetions = pms_reservetions[
        pms_reservetions['expert_name'].isin(team_members)
    ]

    # this and last month status
    this_month_name = MONTH_NAMES[f"{jdatetime.date.today().month:02d}"]
    last_month_name = MONTH_NAMES[f"{(jdatetime.date.today().month -1) or 12:02d}"]

    tabs = st.tabs([f"وضعیت {this_month_name}", f"وضعیت {last_month_name}"])

    with tabs[0]:
        this_month_first_date = jdatetime.date.today().replace(day=1).togregorian()
        st.subheader(f"وضعیت عملکرد روزانه در ماه {this_month_name}")

        last_month_row = monthly_records.sort_values('date', ascending=False).head(1)

        if last_month_row['target'].values[0] > last_month_row['total_revenue'].values[0]:
            this_month_target_value = last_month_row['target'].values[0]
        else:
            this_month_target_value = last_month_row['total_revenue'].values[0]
        this_month_recordes = daily_records[
            daily_records['Date'] >= this_month_first_date
        ].reset_index(drop=True)
        this_month_recordes['target_achieved'] = (
            this_month_recordes['Ammount_so_far'] >= this_month_recordes['Target']
        )
        
        cols = st.columns(2)

        with cols[0]:
            cols_ = st.columns(2)

            with cols_[0]:
                st.metric(
                    label="تعداد روزهای که که تارگت زده شده",
                    value=f"{this_month_recordes['target_achieved'].sum()} روز"
                )
                st.metric(
                    label="درآمد کل تا امروز",
                    value=f"{this_month_recordes['Ammount_so_far'].sum():,} تومان"
                )

                # reward if they achieve the monthly target
                # 5% of difference between target and ammount_so_far if target is achieved
                if this_month_recordes['Ammount_so_far'].sum() >= this_month_target_value:
                    monthly_reward = (this_month_recordes['Ammount_so_far'].sum() - this_month_target_value) * 0.05
                    
                    st.metric(
                            label="میزان پاداش ماهانه",
                            value=f"{monthly_reward:0,.0f} تومان"
                        )
                    
            with cols_[1]:
                # reward of the daily target achievement
                # 20% of the difference between target and ammount_so_far if target is achieved
                total_reward = 0
                for _, row in this_month_recordes.iterrows():
                    if row['target_achieved']:
                        diff = row['Ammount_so_far'] - row['Target']
                        reward = diff * 0.2
                        total_reward += reward
                
                st.metric(
                    label="جمع پاداش روزانه",
                    value=f"{total_reward:0,.0f} تومان"
                )
                
                st.metric(
                    label= "تارگت این ماه",
                    value=f"{this_month_target_value:,} تومان"        
                )
            
            # pie plot for progress monthly target
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = this_month_recordes['Ammount_so_far'].sum(),
                delta = {'reference': this_month_target_value, 'valueformat':',', 'relative': False, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, this_month_target_value]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, this_month_target_value*0.5], 'color': "lightgray"},
                        {'range': [this_month_target_value*0.5, this_month_target_value], 'color': "gray"}
                    ],
                    'threshold' : {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': this_month_target_value
                    }
                },
                title = {'text': "درصد پیشرفت تارگت ماهانه", 'font': {'size': 16}}
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # daily records
        with cols[1]:
            st.dataframe(
                this_month_recordes[['jalali_date', 'Target', 'Ammount_so_far', 'target_achieved', 'Gap_to_target', 'reward']].rename(columns={
                    'jalali_date': 'تاریخ',
                    'Target': 'تارگت روزانه',
                    'Ammount_so_far': 'درآمد',
                    'target_achieved': 'تارگت رو زدن؟',
                    'Gap_to_target': 'فاصله از تارگت',
                    'reward': 'پاداش'
                }))
            
        st.markdown("---")
        st.subheader("رزرو های ماهانه")
    
        this_month_monthly_reservetions = pms_reservetions[
                (pms_reservetions['created_at'] >= pd.to_datetime(this_month_first_date).tz_localize('Asia/Tehran')) &
            (pms_reservetions['expert_name'].isin(team_members)) &
            (pms_reservetions['total_nights'] >= 15) &
            (pms_reservetions['last_status'].isin([2, 4]))
        ]
        st.dataframe(this_month_monthly_reservetions)
    with tabs[1]:
        last_month_first_date = (jdatetime.date.today().replace(day=1) - jdatetime.timedelta(days=1)).replace(day=1).togregorian()
        last_month_last_date = (jdatetime.date.today().replace(day=1) - jdatetime.timedelta(days=1)).togregorian()
        
        st.subheader(f"وضعیت عملکرد تیم در ماه {last_month_name}")
        last_month_recordes = daily_records[
            (daily_records['Date'] >= last_month_first_date) &
            (daily_records['Date'] <= last_month_last_date)
        ].reset_index(drop=True)
        last_month_recordes['target_achieved'] = (
            last_month_recordes['Ammount_so_far'] >= last_month_recordes['Target']
        )

        last_month_row = monthly_records.sort_values('date', ascending=False).head(1)

        cols = st.columns(2)
        with cols[0]:

            cols_ = st.columns(2)
            with cols_[0]:
                st.metric(
                    label="تعداد روزهای که که تارگت زده شده",
                    value=f"{last_month_recordes['target_achieved'].sum()} روز"
                )
                st.metric(
                    label="درآمد کل ماه گذشته",
                    value=f"{last_month_recordes['Ammount_so_far'].sum():,} تومان"
                )
            
            # reward if they achieve the monthly target
            # 5% of difference between target and ammount_so_far if target is achieved
            if last_month_recordes['Ammount_so_far'].sum() >= last_month_row['target'].values[0]:
                monthly_reward = (last_month_recordes['Ammount_so_far'].sum() - last_month_row['target'].values[0]) * 0.05
                st.metric(
                    label="میزان پاداش ماهانه",
                    value=f"{monthly_reward:0,.0f} تومان"
                )

            with cols_[1]:
                # reward of the daily target achievement
                # 20% of the difference between target and ammount_so_far if target is achieved
                total_reward = 0
                for _, row in last_month_recordes.iterrows():
                    if row['target_achieved']:
                        diff = row['Ammount_so_far'] - row['Target']
                        reward = diff * 0.2
                        total_reward += reward
                
                st.metric(
                    label="جمع پاداش روزانه",
                    value=f"{total_reward:0,.0f} تومان"
                )   

                st.metric(
                    label= "تارگت ماه گذشته",
                    value=f"{last_month_row['target'].values[0]:,} تومان"        
                )

            # pie plot for progress monthly target
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = last_month_recordes['Ammount_so_far'].sum(),
                delta = {'reference': last_month_row['target'].values[0], 'valueformat':',', 'relative': False, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, last_month_row['target'].values[0]]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, last_month_row['target'].values[0]*0.5], 'color': "lightgray"},
                        {'range': [last_month_row['target'].values[0]*0.5, last_month_row['target'].values[0]], 'color': "gray"}
                    ],
                    'threshold' : {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': last_month_row['target'].values[0]
                    }
                },
                title = {'text': "درصد پیشرفت تارگت ماهانه", 'font': {'size': 16}}
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # daily records
        with cols[1]:
            st.dataframe(
                last_month_recordes[['jalali_date', 'Target', 'Ammount_so_far', 'target_achieved', 'Gap_to_target', 'reward']].rename(columns={
                    'jalali_date': 'تاریخ',
                    'Target': 'تارگت روزانه',
                    'Ammount_so_far': 'درآمد',
                    'target_achieved': 'تارگت رو زدن؟',
                    'Gap_to_target': 'فاصله از تارگت',
                    'reward': 'پاداش'
                }), height=600)
        st.markdown("---")

    return 
    
if __name__ == "__main__":
    sales()
