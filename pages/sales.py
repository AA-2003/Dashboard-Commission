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
from utils.sheetConnect import load_sheet


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
    

def performance_evaluation_sheet_load(month: str) -> Optional[str]:
    """Load performance evaluation sheet for a given month."""
    try:
        year, month_num = month.split('-')
        year = str(year)
        month_name = MONTH_NAMES[month_num]

        # get all sheet names
        sheet_names = get_sheet_names('EVAL_SPREADSHEET_ID')
        # if first char is آ replace it with ا
        month_name_alt = month_name[0].replace('آ', 'ا') + month_name[1:]

        for sheet_name in sheet_names:
            # Split sheet name into words to avoid partial matches (e.g., 'دی' in 'اردیبهشت')
            sheet_name_words = sheet_name.replace('-', ' ').replace('_', ' ').split()
            if (month_name in sheet_name_words or month_name_alt in sheet_name_words) and \
                (year in sheet_name or year[1:] in sheet_name or year[-2:] in str(sheet_name)):
                return sheet_name
        return None
    except Exception as e:
        handel_errors(e, "Error loading performance evaluation sheet")

def month_tab(
        first_date_of_month_gregorian: dt.date,
        last_date_of_month_gregorian: dt.date,
        monthly_records: pd.DataFrame,
        daily_records: pd.DataFrame,
        didar_deals: pd.DataFrame,
        team_members: List[str]
        ):
    """Render the tab for a specific month."""

    st.subheader("وضعیت تارگت روزانه")

    total_daily_traget_commission = 0
    total_month_target_commission = 0

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

            # reward if they achieve the monthly target
            # 5% of difference between target and ammount_so_far if target is achieved
            if month_recordes['Ammount_so_far'].sum() >= month_target_value:
                total_month_target_commission += (month_recordes['Ammount_so_far'].sum() - month_target_value) * 0.05
                st.metric(
                        label="میزان پاداش ماهانه",
                        value=f"{total_month_target_commission:0,.0f} تومان"
                    )
                
        with cols_[1]:
            # reward of the daily target achievement
            # 20% of the difference between target and ammount_so_far if target is achieved
            for _, row in month_recordes.iterrows():
                if row['target_achieved']:
                    diff = row['Ammount_so_far'] - row['Target']
                    reward = diff * 0.2
                    total_daily_traget_commission += reward
                
            st.metric(
                label="جمع پاداش روزانه",
                value=f"{total_daily_traget_commission:0,.0f} تومان"
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

    # filler valid deals

    filterd_didar_deals = didar_deals[
        (didar_deals['deal_created_time'].dt.date >= first_date_of_month_gregorian) &
        (didar_deals['deal_created_time'].dt.date <= last_date_of_month_gregorian) &
        (~didar_deals['deal_source'].isin([
            'پلت‌فرم', "مهمان واسطه", "تلگرام(سوشال)", "واتساپ(سوشال)", "فرودگاه", "دایرکت اینستاگرام"
        ]))
    ].reset_index(drop=True)

    # deals that checkout is in this month and total nights >= 15 and only new sells
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
    monthly_reward = monthly_deals.groupby('deal_owner')['deal_value'].sum().reset_index()
    monthly_reward['monthly_reservation_commission'] = (monthly_reward['deal_value'] * 0.03).astype(int)

    # map the monthly reservation commission to team members table
    team_members = team_members.merge(
        monthly_reward[['deal_owner', 'monthly_reservation_commission']],
        left_on='didar_name',
        right_on='deal_owner',
        how='left'
    )
    team_members['monthly_reservation_commission'] = team_members['monthly_reservation_commission'].fillna(0)

    st.write('---')
    st.subheader("ارزیابی عملکرد و کمیسیون نهایی")
    first_date_of_month_jalali = jdatetime.date.fromgregorian(date=first_date_of_month_gregorian)
    performance_evaluation = performance_evaluation_sheet_load(
        month=f"{first_date_of_month_jalali.year}-{first_date_of_month_jalali.month:02d}"
    )
    if performance_evaluation:
        try:
            eval_sheet = load_sheet('EVAL_SPREADSHEET_ID', performance_evaluation)
            percent_row = eval_sheet.loc[65].to_dict()
            team_members['performance_percent'] = 0.0
            for member in team_members['evaluation_sheet_name'].tolist():
                if member in percent_row:
                    team_members.loc[team_members['evaluation_sheet_name'] == member, 'performance_percent'] = float(percent_row[member].replace('%', ''))

            team_members['daily_target_commission'] = (total_daily_traget_commission * (team_members['performance_percent'] / 100)).round(0)
            team_members['month_target_commission'] = (total_month_target_commission * (team_members['performance_percent'] / 100)).round(0)
            team_members['total_commission'] = (
                team_members['monthly_reservation_commission'] +
                team_members['daily_target_commission'] +
                team_members['month_target_commission']
            ).astype(int)

            st.dataframe(
                team_members[[
                    'didar_name', 'performance_percent', 'monthly_reservation_commission',
                    'daily_target_commission', 'month_target_commission', 'total_commission'
                ]].sort_values(by='total_commission', ascending=False).rename(columns={
                    'didar_name': 'نام فروشنده',
                    'performance_percent': 'درصد ارزیابی عملکرد',
                    'monthly_reservation_commission': 'کمیسیون رزرو ماهانه (3%)',
                    'daily_target_commission': 'کمیسیون پاداش روزانه',
                    'month_target_commission': 'کمیسیون پاداش ماهانه',
                    'total_commission': 'جمع کل کمیسیون‌ها'
                })
            )
        except Exception as e:
            handel_errors(e, "خطا در بارگذاری داده‌های ارزیابی عملکرد")
    else:
        st.info("هنوز ارزیابی عملکرد این ماه انجام نشده است.")
        st.dataframe(
            team_members[[
                'didar_name',
                'monthly_reservation_commission'
            ]].sort_values(by='monthly_reservation_commission', ascending=False).rename(columns={
                'didar_name': 'نام فروشنده',
                'monthly_reservation_commission': 'کمیسیون رزرو ماهانه (3%)'
            })
        )

    st.markdown("---")
    st.subheader("جزئیات عملکرد هر کارشناس فروش")

    # each expert tab
    tabs = st.tabs(filterd_didar_deals.groupby('deal_owner')['deal_value'].sum().sort_values(ascending=False).index.tolist())
    for tab_name, tab in zip(filterd_didar_deals.groupby('deal_owner')['deal_value'].sum().sort_values(ascending=False).index.tolist(), tabs):
        with tab:
            st.markdown(f"### عملکرد {tab_name} در ماه جاری")

            new_sales = filterd_didar_deals[filterd_didar_deals['deal_type']=="فروش جدید"].reset_index(drop=True)
            renewal_sales = filterd_didar_deals[filterd_didar_deals['deal_type']=="تمدید"].reset_index(drop=True)

            cols = st.columns(2)
            with cols[0]:
                # total deals
                st.metric(
                    label="تعداد رزروهای جدید",
                    value=new_sales[new_sales['deal_owner'] == tab_name].shape[0]
                )
                # total nights
                st.metric(
                    label="مجموع تعداد شب رزروهای جدید",
                    value=new_sales[new_sales['deal_owner'] == tab_name]['product_quantity'].sum()
                )
                # total value
                st.metric(
                    label="مجموع ارزش رزروهای جدید (تومان)",
                    value=f"{new_sales[new_sales['deal_owner'] == tab_name]['deal_value'].sum():,}"
                )
                # avg night per deal
                st.metric(
                    label="میانگین تعداد شب هر رزرو جدید",
                    value=f"{new_sales[new_sales['deal_owner'] == tab_name]['product_quantity'].mean():.2f}"
                )

            with cols[1]:
                # total deals
                st.metric(
                    label="تعداد رزروهای تمدید",
                    value=renewal_sales[renewal_sales['deal_owner'] == tab_name].shape[0]
                )
                # total nights
                st.metric(
                    label="مجموع تعداد شب رزروهای تمدید",
                    value=renewal_sales[renewal_sales['deal_owner'] == tab_name]['product_quantity'].sum()
                )
                # total value
                st.metric(
                    label="مجموع ارزش رزروهای تمدید (تومان)",
                    value=f"{renewal_sales[renewal_sales['deal_owner'] == tab_name]['deal_value'].sum():,}"
                )
                # avg night per deal
                st.metric(
                    label="میانگین تعداد شب هر رزرو تمدید",
                    value=f"{renewal_sales[renewal_sales['deal_owner'] == tab_name]['product_quantity'].mean():.2f}"
                )

            with st.expander(f"جزئیات رزروهای {tab_name}"):
                expert_deals = filterd_didar_deals[filterd_didar_deals['deal_owner'] == tab_name].reset_index(drop=True)
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
            
            st.markdown("#### رزروهای ماهانه")
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

            with st.expander(f"جزئیات رزروهای ماهانه {tab_name}"):
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
    
    if 'sales' not in teams:
        st.error("شما به این بخش دسترسی ندارید")
        return
    
    if 'all_teams_users' not in st.session_state or st.session_state.all_teams_users is None:
        try:
            all_teams_users = load_sheet(key='QC_SPREADSHEET_ID', sheet_name='Users') 
            st.session_state.all_teams_users = all_teams_users
        except Exception as e:
            handel_errors(e, "Error loading all teams users data")
    print(st.session_state.all_teams_users)
    # Get team members
    team_members = st.session_state.all_teams_users[
        st.session_state.all_teams_users['team'].apply(
            lambda x: 'Sales' in [t.strip() for t in x.split('|')]
        ) & (~st.session_state.all_teams_users['role'].isin(['Admin', 'Team Manager']))
    ]
    team_member_names = team_members['didar_name'].tolist()
    
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
        if 'deals_data' not in st.session_state or st.session_state.deals_data is None:
            deals_data = load_sheet(key='DEALS_SPREADSHEET_ID', sheet_name='Didar Deals')
            st.session_state.deals_data = deals_data
        didar_deals = st.session_state.deals_data.copy()
        didar_deals = didar_deals[
            (didar_deals['deal_owner'].isin(team_member_names)) &
            (didar_deals['deal_status']=="Won")
        ].reset_index(drop=True)
        didar_deals['checkout'] = pd.to_datetime(didar_deals['checkout'])
        didar_deals['deal_created_time'] = pd.to_datetime(didar_deals['deal_created_time'])
        didar_deals['product_quantity'] = didar_deals['product_quantity'].astype(float)
        didar_deals['deal_value'] = pd.to_numeric(didar_deals['deal_value'], errors='coerce').fillna(0) / 10
    except Exception as e:
        handel_errors(e, "خطا در بارگذاری داده‌های PMS")

    # number of tabs: month from azar 1404 to this month
    first_month = jdatetime.date(1404, 9, 1)
    current_month = jdatetime.date.today().replace(day=1)
    month_list = []
    temp_month = first_month
    while temp_month <= current_month:
        month_list.append(temp_month)
        if temp_month.month == 12:
            temp_month = jdatetime.date(temp_month.year + 1, 1, 1)
        else:
            temp_month = jdatetime.date(temp_month.year, temp_month.month + 1, 1)
    month_list.reverse()

    tabs = st.tabs([f"وضعیت {MONTH_NAMES[f'{m.month:02d}']} {m.year}" for m in month_list])

    for tab_name, tab, month in zip(
        [f"وضعیت {MONTH_NAMES[f'{m.month:02d}']} {m.year}" for m in month_list],
        tabs,
        month_list
    ):
        with tab:
            first_date_of_month_gregorian = month.replace(day=1).togregorian()
            last_date_of_month_jalali = (month.replace(day=1) + jdatetime.timedelta(days=32)).replace(day=1) - jdatetime.timedelta(days=1)
            last_date_of_month_gregorian = last_date_of_month_jalali.togregorian()
            month_tab(
                first_date_of_month_gregorian,
                last_date_of_month_gregorian,
                monthly_records,
                daily_records,
                didar_deals,
                team_members
            )
    
if __name__ == "__main__":
    sales()