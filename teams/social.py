import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import jdatetime

from utils.write_sheet import write_df_to_sheet
from utils.load_sheet import load_sheet, load_sheet_uncache


def normalize_owner(owner):
    if owner in ["محمد آبساران/روز", "محمد آبساران/شب"]:
        return "محمد آبساران"
    elif owner == "حسین  طاهری":
        return "حسین  طاهری"
    elif owner == "فرشته فرج نژاد":
        return "فرشته فرج نژاد"
    elif owner == "پوریا کیوانی":
        return "پوریا کیوانی"
    elif owner == "حافظ قاسمی":
        return "حافظ قاسمی"
    elif owner == "پویا  ژیانی":
        return "پویا  ژیانی"
    elif owner == "بابک  مسعودی":
        return "بابک  مسعودی"
    elif owner == "پویا وزیری":
        return "پویا وزیری"
    elif owner == "Sara Malekzadeh":
        return "سارا ملک زاده"
    return owner


def social():
    """Social team dashboard with metrics and visualizations (English comments)"""
    st.title("📊 داشبورد تیم Social")
    
    # Check user authentication and required session keys
    if not all(key in st.session_state for key in ['username', 'role', 'team', 'auth']):
        st.error("لطفا ابتدا وارد سیستم شوید")
        return

    # Get user info from session
    role = st.session_state.role
    username = st.session_state.username
    name = st.session_state.name

    st.write(f"{name}  عزیز خوش آمدی😃")  

    if role in ["admin", "manager"]:
        # Load and filter data for social team
        data = st.session_state['data']
        data = data[data['team'] == 'social'].reset_index(drop=True).copy()
        data['deal_owner'] = data['deal_owner'].apply(normalize_owner)

        # Load reward parameters from sheet
        parametrs_df = load_sheet_uncache('Social team parameters')
        # shift sheet
        shift_sheet = load_sheet('Social shift') 


        tabs = st.tabs(['صفحه اصلی', 'تنظیمات'])
        with tabs[0]:
            month_choose = st.selectbox(
                label='ماه مورد نظر را انتخاب کنید:',
                options=['این ماه', 'ماه پیش'],
                key='month_select_box'
            )

            month_map = {
                'این ماه': 0,
                'ماه پیش': 1
            }

            month_index = month_map.get(month_choose, 0)

            # Convert 'deal_created_date' to Jalali date
            @st.cache_data(ttl=600)
            def safe_to_jalali(x):
                return jdatetime.date.fromgregorian(date=pd.to_datetime(x).date())
            
            data['deal_created_date'] = pd.to_datetime(data['deal_created_date']).dt.date
            data['jalali_date'] = data['deal_created_date'].apply(safe_to_jalali)
            data['jalali_year_month'] = data['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}")
            
            shift_sheet['تاریخ میلادی'] = pd.to_datetime(shift_sheet['تاریخ میلادی'])
            shift_sheet['jalali_date'] = shift_sheet['تاریخ میلادی'].apply(safe_to_jalali)
            shift_sheet['jalali_year_month'] = shift_sheet['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}")

            # Get month 
            today = jdatetime.date.today()
            year = today.year
            month = today.month
            if month_index == 1:
                # Calculate previous month
                if month == 1:
                    prev_month = 12
                    prev_year = year - 1
                else:
                    prev_month = month - 1
                    prev_year = year
                month = f"{prev_year}-{prev_month:02d}"
            else:
                month = f"{year}-{month:02d}"
            

            st.info(f'تاریخ: {month}')
            this_month_deals = data.loc[data['jalali_year_month'] == month]
            this_month_shifts = shift_sheet[shift_sheet['jalali_year_month'] == month]
            
            if this_month_deals.empty:
                st.info('هیچ معامله‌ای برای این ماه ثبت نشده است!!!')
            else:

                # team metrics
                value_sum = this_month_deals['deal_value'].sum()
                number_of_deals = this_month_deals.shape[0]
                number_of_leads = (this_month_shifts['تعداد پیام (اینستاگرام)'].sum() + 
                         this_month_shifts['تعداد پیام (تلگرام)'].sum() + this_month_shifts['تعداد پیام (واتساپ)'].sum()) 
                

                cols = st.columns(2)
                with cols[0]:
                    st.metric('میزان فروش', f'{value_sum:,.0f} ریال')
                with cols[1]:
                    st.metric('تعداد فروش', number_of_deals)
                    if number_of_deals > 0:
                        st.metric(f'تعداد لید تا {str(this_month_shifts['jalali_date'].max())}', number_of_leads)

                # number of deals per day
                daily_deal_count = this_month_deals.groupby('deal_created_date').size().reset_index(name='تعداد')
                daily_deal_count['jalali_date'] = daily_deal_count['deal_created_date'].apply(safe_to_jalali)

                st.subheader('تعداد معامله به ازای روز')
                fig = px.line(
                    daily_deal_count,
                    x='deal_created_date',
                    y='تعداد',
                    title='',
                    labels={
                        'deal_created_date': 'تاریخ'
                    },
                    markers=True,
                    hover_data=['jalali_date']
                )
                st.plotly_chart(fig, use_container_width=True)


                # filters on sellers and channels
                st.write('---')
                channels = this_month_deals['deal_source'].unique().tolist()
                sellers = this_month_deals['deal_owner'].unique().tolist()

                cols = st.columns(2)
                # channel filter 
                with cols[0]:
                    channel_values = st.multiselect(
                        label="انتخاب کانال فروش",
                        options=channels,
                        key='channel_mutliselect'
                    )

                # seller filter
                with cols[1]:
                    seller_values = st.multiselect(
                        label='انتخاب فروشنده:',
                        options=sellers,
                        key='seller_mutliselect'
                    )
                if len(seller_values)==0 or len(channel_values)==0:
                    st.info('حداقل یک کارشناس و یک کانال را انتخاب کنید.')
                else:
                    filtered_deals = this_month_deals[
                        (this_month_deals['deal_owner'].isin(seller_values)) &
                        (this_month_deals['deal_source'].isin(channel_values))
                    ]
                    filtered_shift = this_month_shifts[
                        this_month_shifts['کارشناس'].isin(seller_values)
                    ]
                    if filtered_deals.empty:
                        st.info('هیچ معامله‌ای با این فیلترها وجود ندارد‍‍!!!')
                        return
                    value_sum = filtered_deals['deal_value'].sum()
                    number_of_deals = filtered_deals.shape[0]
                    number_of_deals = (
                        (filtered_shift['تعداد پیام (اینستاگرام)'].sum() if 'دایرکت اینستاگرام' in channel_values else 0) +
                        (filtered_shift['تعداد پیام (تلگرام)'].sum() if 'تلگرام(سوشال)' in channel_values else 0) +
                        (filtered_shift['تعداد پیام (واتساپ)'].sum() if 'واتساپ(سوشال)' in channel_values else 0)
                    )
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric('میزان فروش', f'{value_sum:,.0f} ریال')
                    with cols[1]:
                        st.metric('تعداد فروش', number_of_deals)
                        st.metric('تعداد لید: ', number_of_deals)

                    # number of deals per day
                    daily_deal_count = filtered_deals.groupby('deal_created_date').size().reset_index(name='تعداد')

                    all_days = pd.date_range(
                        daily_deal_count['deal_created_date'].min(), 
                        daily_deal_count['deal_created_date'].max())

                    all_days_df = pd.DataFrame({'deal_created_date': all_days})
                    

                    daily_deal_count['deal_created_date'] = pd.to_datetime(daily_deal_count['deal_created_date'])
                    all_days_df['deal_created_date'] = pd.to_datetime(all_days_df['deal_created_date'])
                    daily_deal_count = all_days_df.merge(daily_deal_count, on='deal_created_date', how='left').fillna(0)
                    
                    daily_deal_count['تعداد'] = daily_deal_count['تعداد'].astype(int)

                    daily_deal_count['تاریخ شمسی'] = daily_deal_count['deal_created_date'].apply(safe_to_jalali)
                    st.subheader('تعداد معامله به ازای روز')
                    fig = px.line(
                        daily_deal_count,
                        x='deal_created_date',
                        y='تعداد',
                        title='',
                        labels={
                            'deal_created_date': 'تاریخ میلادی'
                        },
                        markers=True,
                        hover_data=['تاریخ شمسی']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # number of leads per day
                    daily_lead_count_insta = filtered_shift.groupby('تاریخ میلادی')['تعداد پیام (اینستاگرام)'].sum().reset_index(name="تعداد")
                    daily_lead_count_tele  = filtered_shift.groupby('تاریخ میلادی')['تعداد پیام (تلگرام)'].sum().reset_index(name="تعداد")
                    daily_lead_count_whats = filtered_shift.groupby('تاریخ میلادی')['تعداد پیام (واتساپ)'].sum().reset_index(name="تعداد")

                    dfs = []

                    if 'واتساپ(سوشال)' in channel_values:
                        dfs.append(daily_lead_count_whats)
                    if 'تلگرام(سوشال)' in channel_values:
                        dfs.append(daily_lead_count_tele)
                    if 'دایرکت اینستاگرام' in channel_values:
                        dfs.append(daily_lead_count_insta)
                        
                    daily_lead_count = pd.concat(dfs).groupby('تاریخ میلادی')['تعداد'].sum().reset_index()

                    all_days = pd.date_range(daily_lead_count['تاریخ میلادی'].min(), 
                                            daily_lead_count['تاریخ میلادی'].max())

                    all_days_df = pd.DataFrame({'تاریخ میلادی': all_days})

                    daily_lead_count = all_days_df.merge(daily_lead_count, on='تاریخ میلادی', how='left').fillna(0)

                    daily_lead_count['تعداد'] = daily_lead_count['تعداد'].astype(int)

                    daily_lead_count['تاریخ شمسی'] = daily_lead_count['تاریخ میلادی'].apply(safe_to_jalali)

                    st.subheader('تعداد لید به ازای روز')
                    fig = px.line(
                        daily_lead_count,
                        x='تاریخ میلادی',
                        y='تعداد',
                        title='',
                        labels={
                            'تاریخ میلادی': 'تاریخ میلادی',
                            'تعداد': 'تعداد پیام'
                        },
                        markers=True,
                        hover_data=['تاریخ شمسی']
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # reward section
            # reward calculate base on checkout date
            st.write('---')
            st.subheader('پاداش')
            target = parametrs_df['target'].values[0]
            data['checkout_jalali'] = data['checkout_date'].apply(safe_to_jalali)
            data['checkout_jalali_year_month'] = data['checkout_jalali'].apply(lambda d: f"{d.year}-{d.month:02d}")
            
            deals_for_reward = data[data['checkout_jalali_year_month']==month]
            if deals_for_reward.empty:
                st.info('ّهیچ معامله‌ای با تاریخ خروج در این ماه ثبت نشده است!!!')
            else:
                current_value = deals_for_reward['deal_value'].sum()/10
                # Calculate percent of target (current vs 95% of best previous)
                if target > 0:
                    percent_of_target = (current_value / (0.95 * target)) * 100
                    percent_of_target = min(percent_of_target, 100)
                else:
                    percent_of_target = 0

                # Determine reward percent based on performance
                if target > 0 and current_value >= 0.95 * target:
                    reward_percent = parametrs_df['grow_percent'].values[0]
                else:
                    reward_percent = parametrs_df['normal_percent'].values[0]
                
                st.metric('تارگت', value=f'{target:,.0f}')

                # --- Visualizations ---
                # 1. Gauge: percent of target achieved
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=percent_of_target,
                    delta={'reference': 100, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "royalblue"},
                        'steps': [
                            {'range': [0, 95], 'color': "#ffe0e0"},
                            {'range': [95, 100], 'color': "#e0ffe0"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 95
                        }
                    },
                    title={'text': "میزان پیشرفت"}
                ))


                st.plotly_chart(gauge_fig, use_container_width=True)
                selected_member = st.selectbox(
                    "انتخاب کارشناس:",
                    this_month_deals['deal_owner'].unique().tolist(),
                    index=1
                )
                member_deals = deals_for_reward[deals_for_reward['deal_owner']==selected_member]
                member_value = member_deals['deal_value'].sum()
                member_reward = member_value * reward_percent /100
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric(f'میزان فروش {selected_member}:', value=f'{member_value:,.0f} ریال')
                with cols[1]:
                    st.metric(f'میزان پاداش {selected_member}:', value=f'{member_reward:,.0f} ریال')

                                 

        # --- Settings tab: update reward parameters ---
        with tabs[1]:
            with st.form('social team parameters'):
                col1, col2 = st.columns(2)
                with col1:
                    target =  st.number_input(
                        label="تارگت این ماه",
                        key='target',
                        format='%d',
                        step=10_000_000,
                        min_value=1_000_000,
                        value=parametrs_df['target'].astype(int).values[0]
                    )
                with col2:
                    grow_percent = st.number_input(
                        label="درصد پاداش در صورت رشد:",
                        key='grow_percent',
                        format="%f",
                        step=1.0,
                        max_value=100.0,
                        value=parametrs_df['grow_percent'].astype(float).values[0]
                    )
                    normal_percent = st.number_input(
                        label="درصد در حالت عادی:",
                        key='normal_percent',
                        format="%f",
                        step=1.0,
                        max_value=100.0,
                        value=parametrs_df['normal_percent'].astype(float).values[0]
                    )
                submitted = st.form_submit_button('تنظیم مجدد')
                if submitted:
                    df = pd.DataFrame([{
                        "grow_percent": grow_percent,
                        "normal_percent": normal_percent,
                        "target": target
                    }])
                    success = write_df_to_sheet(df, sheet_name='Social team parameters')
                    if success:
                        st.info("پارامترها با موفقیت آپدیت شد.")
                    else:
                        st.info("خطا در آپدیت پارامترها!!!")
    else:
        pass