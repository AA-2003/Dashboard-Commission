import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
import datetime as dt
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
        print(f"Error converting date: {gregorian_date}")
        return ""

def month_tab(
        first_date_of_month_gregorian: dt.date,
        last_date_of_month_gregorian: dt.date,
        monthly_records: pd.DataFrame,
        daily_records: pd.DataFrame,
        didar_deals: pd.DataFrame
        ):
    """Render the tab for a specific month."""

    st.subheader("وضعیت تارگت روزانه")
    
    month_row = monthly_records[
        (monthly_records['date'] == first_date_of_month_gregorian)
    ]
    if month_row.empty:
        # if it is empty, its mean this month is not finished yet
        # so we need to get the data from another sheet
        this_month = load_sheet('PMS_SPREADSHEET_ID', 'this_month')
        this_month = this_month.rename(columns={
            'date': 'today',
            'amount_so_far': 'total_revenue',
            'target': 'target'
        })
        this_month['total_revenue'] = this_month['total_revenue'].str.replace(',', '').astype(float)
        this_month['target'] = this_month['target'].str.replace(',', '').astype(float)
        month_row = this_month.copy()
        
    month_target_value = month_row['target'].values[0]
    month_recordes = daily_records[
        (daily_records['Date'] >= first_date_of_month_gregorian) &
        (daily_records['Date'] <= last_date_of_month_gregorian)
    ].reset_index(drop=True)
    month_recordes['target_achieved'] = (
        month_recordes['Ammount_so_far'] >= month_recordes['Target']
    )
    cols = st.columns(2)

    with cols[0]:
        cols_ = st.columns(2)

        with cols_[0]:
            st.metric(
                label="تعداد روزهای که که تارگت زده شده",
                value=f"{month_recordes['target_achieved'].sum()} روز"
            )
            st.metric(
                label="درآمد کل",
                value=f"{month_recordes['Ammount_so_far'].sum():,} تومان"
            )
            
            # test monthly reward
            x = month_recordes['Ammount_so_far'].sum() * 1.2

            # reward if they achieve the monthly target
            # 5% of difference between target and ammount_so_far if target is achieved
            # if month_recordes['Ammount_so_far'].sum() >= month_target_value:
            if x >= month_target_value:
                # monthly_reward = (month_recordes['Ammount_so_far'].sum() - month_target_value) * 0.05
                monthly_reward = (x - month_target_value) * 0.05
                st.metric(
                        label="میزان پاداش ماهانه",
                        value=f"{monthly_reward:0,.0f} تومان"
                    )
                
        with cols_[1]:
            # reward of the daily target achievement
            # 20% of the difference between target and ammount_so_far if target is achieved
            total_reward = 0
            for _, row in month_recordes.iterrows():
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
                value=f"{month_target_value:,} تومان"        
            )
        
        # pie plot for progress monthly target
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = month_recordes['Ammount_so_far'].sum(),
            delta = {'reference': month_target_value, 'valueformat':',', 'relative': False, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, month_target_value]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, month_target_value*0.5], 'color': "lightgray"},
                    {'range': [month_target_value*0.5, month_target_value], 'color': "gray"}
                ],
                'threshold' : {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': month_target_value
                }
            },
            title = {'text': "درصد پیشرفت تارگت ماهانه", 'font': {'size': 16}}
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # daily records
    with cols[1]:
        st.dataframe(
            month_recordes[['jalali_date', 'Target', 'Ammount_so_far', 'target_achieved', 'Gap_to_target', 'reward']].rename(columns={
                'jalali_date': 'تاریخ',
                'Target': 'تارگت روزانه',
                'Ammount_so_far': 'درآمد',
                'target_achieved': 'تارگت رو زدن؟',
                'Gap_to_target': 'فاصله از تارگت',
                'reward': 'پاداش'
            }))
        
    st.markdown("---")
    st.subheader("رزرو های ماهانه")
    # deals that checkout is in this month and total nights >= 15 and only new sells
    didar_deals['checkout'] = pd.to_datetime(didar_deals['checkout'])
    didar_deals['product_quantity'] = didar_deals['product_quantity'].astype(float)
    monthly_deals = didar_deals[
        (didar_deals['checkout'].dt.date >= first_date_of_month_gregorian) &
        (didar_deals['checkout'].dt.date <= last_date_of_month_gregorian) &
        (didar_deals['product_quantity'] >= 15) &
        (didar_deals['deal_type']=="فروش جدید") &
        (~didar_deals['deal_source'].isin([
            'پلت‌فرم', "مهمان واسطه", "تلگرام(سوشال)", "واتساپ(سوشال)", "فرودگاه", "دایرکت اینستاگرام"
        ]))
    ].reset_index(drop=True)

    # 3 percent of total deal value as monthly reservation commission for each person
    monthly_deals['deal_value'] = pd.to_numeric(monthly_deals['deal_value'], errors='coerce').fillna(0) / 10
    monthly_reward = monthly_deals.groupby('deal_owner')['deal_value'].sum().reset_index()
    monthly_reward['monthly_reservation_commission'] = (monthly_reward['deal_value'] * 0.03).astype(int)

    # each expert tab
    tabs = st.tabs(monthly_deals.groupby('deal_owner')['deal_value'].sum().sort_values(ascending=False).index.tolist())
    for tab_name, tab in zip(monthly_deals.groupby('deal_owner')['deal_value'].sum().sort_values(ascending=False).index.tolist(), tabs):
        with tab:
            
            # some monthly metrics
            cols = st.columns(2)

            with cols[0]:
                # total deals
                st.metric(
                    label="تعداد رزروها",
                    value=monthly_deals[monthly_deals['deal_owner'] == tab_name].shape[0]
                )
                # total nights
                st.metric(
                    label="مجموع تعداد شب رزروها",
                    value=monthly_deals[monthly_deals['deal_owner'] == tab_name]['product_quantity'].sum()
                )

            with cols[1]:
                # total value
                st.metric(
                    label="مجموع ارزش رزروها (تومان)",
                    value=f"{monthly_deals[monthly_deals['deal_owner'] == tab_name]['deal_value'].sum():,}"
                )
                # total commission
                st.metric(
                    label="کمیسیون رزرو ماهانه (3%) (تومان)",
                    value=f"{(monthly_deals[monthly_deals['deal_owner'] == tab_name]['deal_value'].sum() * 0.03):,.0f}"
                )


            with st.expander(f"جزئیات رزروهای {tab_name}"):
                expert_deals = monthly_deals[monthly_deals['deal_owner'] == tab_name].reset_index(drop=True)
                st.dataframe(
                    expert_deals[[
                        'deal_id', 'deal_title', 'deal_value', 'deal_type',
                        'deal_source', 'contact_name', 'product_name', 'product_quantity' 
                    ]].rename(columns={
                        'deal_id': 'شناسه رزرو',
                        'deal_title': 'عنوان رزرو',
                        'deal_value': 'ارزش رزرو (تومان)',
                        'deal_type': 'نوع رزرو',
                        'deal_source': 'چنل رزرو',
                        'contact_name': 'نام مشتری',
                        'product_name': 'تیپ',
                        'product_quantity': 'تعداد شب'
                    })
                )

    st.dataframe(
        monthly_reward.rename(columns={
            'deal_owner': 'نام فروشنده',
            'deal_value': 'مجموع ارزش قراردادها(تومان)',
            'monthly_reservation_commission': 'کمیسیون رزرو ماهانه (3%)'
        })
    )

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

    team_member_names = team_users['didar_name'].tolist()
    try:
        # pms deals 
        # pms_reservetions = load_sheet(key='PMS_SPREADSHEET_ID', sheet_name='PMS_recent_deals')
        # pms_reservetions['created_at'] = pd.to_datetime(pms_reservetions['created_at'], utc=True).dt.tz_convert('Asia/Tehran')
        # pms_reservetions['total_nights'] = pd.to_numeric(pms_reservetions['total_nights'], errors='coerce').fillna(0).astype(int)
        # pms_reservetions['last_status'] = pd.to_numeric(pms_reservetions['last_status'], errors='coerce').fillna(0).astype(int)
        
        # records data from pms deals
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

        # didar deals for monthly reservation
        didar_deals = load_sheet(key='DEALS_SPREADSHEET_ID', sheet_name='Didar Deals')
        
    except Exception as e:
        handel_errors(e, "خطا در بارگذاری داده‌های PMS", show_error=True, raise_exception=True)

    didar_deals = didar_deals[
        (didar_deals['deal_owner'].isin(team_member_names)) &
        (didar_deals['deal_status']=="Won")
    ].reset_index(drop=True)

    # this and last month status
    this_month_name = MONTH_NAMES[f"{jdatetime.date.today().month:02d}"]
    last_month_name = MONTH_NAMES[f"{(jdatetime.date.today().month -1) or 12:02d}"]

    tabs = st.tabs([f"وضعیت {this_month_name}", f"وضعیت {last_month_name}"])

    with tabs[0]:
        this_month_first_date = jdatetime.date.today().replace(day=1).togregorian() 
        this_month_last_date = (jdatetime.date.today().replace(day=1) + jdatetime.timedelta(days=32)).replace(day=1) - jdatetime.timedelta(days=1)
        month_tab(this_month_first_date, this_month_last_date.togregorian(),
                  monthly_records, daily_records, didar_deals)
        
    with tabs[1]:
        last_month_first_date = (jdatetime.date.today().replace(day=1) - jdatetime.timedelta(days=1)).replace(day=1).togregorian()
        last_month_last_date = (jdatetime.date.today().replace(day=1) - jdatetime.timedelta(days=1)).togregorian()
        
        month_tab(last_month_first_date, last_month_last_date,
                  monthly_records, daily_records, didar_deals)
    
if __name__ == "__main__":
    sales()