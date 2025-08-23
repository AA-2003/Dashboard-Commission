import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
from utils.write_sheet import write_df_to_sheet
from utils.load_sheet import load_sheet, load_sheet_uncache
from utils.func import convert_df, convert_df_to_excel


# --- Data Transformation & Helper Functions ---

def normalize_owner(owner: str) -> str:
    """
    Standardizes owner names by mapping variations to a single, consistent name.
    Uses a dictionary for cleaner and more efficient mapping.
    """
    name_map = {
        "محمد آبساران/روز": "محمد آبساران",
        "محمد آبساران/شب": "محمد آبساران",
        "حسین  طاهری": "حسین  طاهری",
        "فرشته فرج نژاد": "فرشته فرج نژاد",
        "پوریا کیوانی": "پوریا کیوانی",
        "حافظ قاسمی": "حافظ قاسمی",
        "پویا  ژیانی": "پویا  ژیانی",
        "بابک  مسعودی": "بابک  مسعودی",
        "پویا وزیری": "پویا وزیری",
        "Sara Malekzadeh": "سارا ملک زاده"
    }
    return name_map.get(owner, owner)

@st.cache_data(ttl=600)
def safe_to_jalali(gregorian_date):
    """
    Safely converts a Gregorian date to a Jalali date, with caching for performance.
    """
    return jdatetime.date.fromgregorian(date=pd.to_datetime(gregorian_date).date())

def get_month_filter_string(month_choice: str) -> str:
    """
    Determines the target Jalali month string ('YYYY-MM') based on user selection.
    
    Args:
        month_choice: The string from the selectbox ('این ماه' or 'ماه پیش').

    Returns:
        A string representing the target Jalali month, e.g., "1403-05".
    """
    today_jalali = jdatetime.date.today()
    if month_choice == 'ماه پیش':
        # To get the previous month, go to the first day of the current month and subtract one day.
        first_day_of_current_month = today_jalali.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - jdatetime.timedelta(days=1)
        return f"{last_day_of_previous_month.year}-{last_day_of_previous_month.month:02d}"
    elif month_choice == "دو ماه پیش":
        # Go to the first day of the current month, subtract one day to get last month,
        # then go to the first day of that month and subtract one day to get two months ago.
        first_day_of_current_month = today_jalali.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - jdatetime.timedelta(days=1)
        first_day_of_previous_month = last_day_of_previous_month.replace(day=1)
        last_day_of_two_months_ago = first_day_of_previous_month - jdatetime.timedelta(days=1)
        return f"{last_day_of_two_months_ago.year}-{last_day_of_two_months_ago.month:02d}"
    else: # Default to 'این ماه'
        return f"{today_jalali.year}-{today_jalali.month:02d}"

# --- UI Component Functions ---

def display_metrics(deals_df: pd.DataFrame, shifts_df: pd.DataFrame, selected_channels: list = None):
    """
    Calculates and displays the primary KPI metrics in Streamlit columns.
    
    Args:
        deals_df: DataFrame containing deal information for the period.
        shifts_df: DataFrame containing shift and lead information.
        selected_channels: A list of channels to filter lead counts by. If None, all are used.
    """
    if deals_df.empty:
        st.info('هیچ معامله‌ای برای نمایش آمار وجود ندارد.')
        return

    value_sum = deals_df['deal_value'].sum() / 10
    number_of_deals = deals_df.shape[0]

    # Calculate lead count based on selected channels for accurate filtering
    lead_count = 0
    if selected_channels is None or 'دایرکت اینستاگرام' in selected_channels:
        lead_count += shifts_df['تعداد پیام (اینستاگرام)'].sum()
    if selected_channels is None or 'تلگرام(سوشال)' in selected_channels:
        lead_count += shifts_df['تعداد پیام (تلگرام)'].sum()
    if selected_channels is None or 'واتساپ(سوشال)' in selected_channels:
        lead_count += shifts_df['تعداد پیام (واتساپ)'].sum()

    cols = st.columns(3)
    cols[0].metric('💰 میزان فروش', f'{value_sum:,.0f} تومان')
    cols[1].metric('📈 تعداد فروش', f'{number_of_deals:,}')
    cols[2].metric('📞 تعداد لید', f'{int(lead_count):,}')


def plot_daily_trend(df: pd.DataFrame, date_col: str, value_col: str, title: str, labels: dict):
    """
    Generates and displays a line chart for daily trend data.

    Args:
        df: The DataFrame containing the data to plot.
        date_col: The name of the column containing dates.
        value_col: The name of the column containing the values to plot.
        title: The title for the chart.
        labels: A dictionary for customizing axis labels.
    """
    if df.empty:
        # st.info(f"داده‌ای برای نمایش نمودار '{title}' وجود ندارد.")
        return
        
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create a full date range to ensure the chart shows days with zero activity
    if not df.empty:
        all_days_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max())
        all_days_df = pd.DataFrame({date_col: all_days_range})
        df = all_days_df.merge(df, on=date_col, how='left').fillna(0)

    df['تاریخ شمسی'] = df[date_col].apply(safe_to_jalali)
    df[value_col] = df[value_col].astype(int)

    st.subheader(title)
    fig = px.line(
        df,
        x=date_col,
        y=value_col,
        labels=labels,
        markers=True,
        hover_data=['تاریخ شمسی']
    )
    st.plotly_chart(fig, use_container_width=True, key=f'plot-{title}')

def display_reward_section(deals_for_reward: pd.DataFrame, parameters: dict, user_filter: str = None):
    """
    Calculates and displays the reward section, including the progress gauge
    and individual reward metrics.

    Args:
        deals_for_reward: DataFrame filtered for deals with checkout dates in the current month.
        parameters: A dictionary containing the target and reward percentages.
        user_filter: If a username is provided, it shows rewards for only that user.
                     Otherwise, it shows a dropdown for admins.
    """
    st.subheader('🏆 پاداش عملکرد (بر اساس تاریخ خروج)')

    if deals_for_reward.empty:
        st.warning('هیچ معامله‌ای با تاریخ خروج در این ماه برای محاسبه پاداش ثبت نشده است.')
        return

    target = parameters.get('target', 0)
    # The deal value is divided by 10, likely to convert from Rials to Tomans.
    current_value = deals_for_reward['deal_value'].sum() / 10
    
    # --- Reward Logic ---
    # The reward percentage changes based on whether the team's sales (current_value)
    # have reached the target.
    # The progress bar is capped at 100%.
    if target > 0:
        percent_of_target = min((current_value / target) * 100, 100)
    else:
        percent_of_target = 0
    
    # Determine which reward percentage to use (normal vs. growth)
    reward_percent = parameters.get('grow_percent', 0) if target > 0 and current_value >= target else parameters.get('normal_percent', 0)
    deals_count = deals_for_reward.shape[0]
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric('🎯 تارگت فروش ماه', value=f'{target:,.0f} تومان')
    col2.metric('تعداد فروش', value=deals_count)
    col3.metric('میزان فروش', value=f"{current_value:,.0f} تومان")
    col4.metric('میانگین مبلغ معامله', value=f"{current_value/deals_count:,.2f} تومان")

    deals_for_reward['checkout_jalali_str'] = deals_for_reward['checkout_jalali'].apply(lambda x: x.strftime('%Y/%m/%d'))

    # --- Progress Gauge Visualization ---
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent_of_target,
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "royalblue"}},
        title={'text': "درصد تحقق تارگت"}
    ))
    st.plotly_chart(gauge_fig, use_container_width=True, key='gauge_plot')

    # --- Individual Reward Display ---
    if user_filter:
        selected_member = user_filter
        st.markdown(f"#### پاداش شما ({selected_member})")
    else: # Admin view with dropdown
        sellers = deals_for_reward['deal_owner'].unique().tolist()
        selected_member = st.selectbox("انتخاب کارشناس برای مشاهده پاداش:", sellers)

    if selected_member:
        member_deals = deals_for_reward[deals_for_reward['deal_owner'] == selected_member]
        member_value = member_deals['deal_value'].sum() / 10
        member_reward = member_value * reward_percent / 100
        
        cols = st.columns(2)
        cols[0].metric(f'میزان فروش {selected_member}', value=f'{member_value:,.0f} تومان')
        cols[1].metric(f'💰 میزان پاداش {selected_member}', value=f'{member_reward:,.0f} تومان')

        with st.expander('جزئیات معامله ها', False):
            data_to_write = member_deals[[
                'deal_id', 'deal_title', 'deal_value', 'deal_done_date',
                'deal_created_date', 'deal_owner', 'deal_source', 'Customer_id',
                'checkout_date', 'checkout_jalali_str'
                ]].rename(columns={
                'deal_id': 'شناسه معامله',
                'deal_title': 'عنوان معامله',
                'deal_value': 'مبلغ معامله',
                'deal_done_date': 'تاریخ انجام معامله',
                'deal_created_date': 'تاریخ ایجاد معامله',
                'deal_owner': 'کارشناس',
                'deal_source': 'کانال فروش',
                'Customer_id': 'شناسه مشتری',
                'checkout_date': 'تاریخ خروج',
                'checkout_jalali_str': 'تاریخ خروج (شمسی)'
                })
            st.write(data_to_write)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="دانلود داده‌ها به صورت CSV",
                    data=convert_df(data_to_write),
                    file_name=f'{selected_member}-deals.csv',
                    mime='text/csv',
                )
            with col2:
                st.download_button(
                    label="دانلود داده‌ها به صورت اکسل",
                    data=convert_df_to_excel(data_to_write),
                    file_name=f'{selected_member}-deals.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )

# --- Main App Function ---

def social():
    """Main function to render the Social team dashboard Streamlit page."""
    st.title("📊 داشبورد تیم Social")
    
    # --- 1. Authentication and Initialization ---
    if not all(key in st.session_state for key in ['username', 'role', 'data', 'auth']):
        st.error("لطفا ابتدا وارد سیستم شوید")
        st.stop()

    role = st.session_state.role
    username = st.session_state.username
    name = st.session_state.name
    is_manager = role in ["admin", "manager"]
    st.write(f"{name} عزیز خوش آمدی 😃")  

    # --- 2. Data Loading and Pre-processing ---
    data = st.session_state['data']
    data = data[data['team'] == 'social'].copy()
    data['deal_owner'] = data['deal_owner'].apply(normalize_owner)
    
    # For rewards, we need Jalali dates based on the checkout_date
    data['checkout_jalali'] = data['checkout_date'].apply(safe_to_jalali)
    data['checkout_jalali_year_month'] = data['checkout_jalali'].apply(lambda d: f"{d.year}-{d.month:02d}")
    
    # For general stats, we use the deal_done_date
    data['deal_done_date'] = pd.to_datetime(data['deal_done_date']).dt.date
    data['jalali_date'] = data['deal_done_date'].apply(safe_to_jalali)
    data['jalali_year_month'] = data['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}")

    # Load parameters and shift data
    parametrs_df = load_sheet_uncache('Social team parameters')
    parameters = parametrs_df.iloc[0].to_dict() if not parametrs_df.empty else {}
    shift_sheet = load_sheet('Social shift') 
    shift_sheet['تاریخ میلادی'] = pd.to_datetime(shift_sheet['تاریخ میلادی'])
    shift_sheet['jalali_date'] = shift_sheet['تاریخ میلادی'].apply(safe_to_jalali)
    shift_sheet['jalali_year_month'] = shift_sheet['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}")

    # --- 3. UI Rendering ---
    if is_manager:
        # Manager/Admin View with Tabs
        tabs = st.tabs(['داشبورد اصلی', 'تنظیمات پاداش'])
        with tabs[0]:
            render_dashboard(data, shift_sheet, parameters)
        with tabs[1]:
            render_settings_tab(parameters)
    else:
        # Regular User View - Filter data to only this user
        render_dashboard(data, shift_sheet, parameters, user_filter=username)

def render_dashboard(deals_data: pd.DataFrame, shift_data: pd.DataFrame, parameters: dict, user_filter: str = None):
    """
    Renders the main dashboard content. Can be used for both admin and user views.

    Args:
        deals_data: The deals data to display (can be for the whole team or a single user).
        shift_data: The shift data to display.
        parameters: The dictionary of team parameters for rewards.
        user_filter: The username of the logged-in user if this is a user-specific view.
    """
    month_choice = st.selectbox('ماه مورد نظر را انتخاب کنید:', ['این ماه', 'ماه پیش', 'دو ماه پیش'])
    target_month = get_month_filter_string(month_choice)
    st.info(f'آمار ماه: {target_month}')

    # Filter dataframes for the selected month
    monthly_deals = deals_data[
        (deals_data['jalali_year_month'] == target_month)&
        (deals_data['deal_type']=='New Sale')
    ]
    monthly_shifts = shift_data[shift_data['jalali_year_month'] == target_month]

    st.subheader("عملکرد کلی تیم")
    display_metrics(monthly_deals, monthly_shifts)
    plot_daily_trend(
        df=monthly_deals.groupby('deal_done_date').size().reset_index(name='تعداد'),
        date_col='deal_done_date',
        value_col='تعداد',
        title='تعداد معاملات روزانه',
        labels={'deal_done_date': 'تاریخ', 'تعداد': 'تعداد معامله'}
    )
    
    st.divider()

    # The filter section is only shown to managers
    if not user_filter:
        st.subheader("🔍 فیلتر و بررسی جزئیات")
        channels = monthly_deals['deal_source'].unique().tolist()
        sellers = monthly_deals['deal_owner'].unique().tolist()

        cols = st.columns(2)
        channel_values = cols[0].multiselect("انتخاب کانال فروش", options=channels, default=channels)
        seller_values = cols[1].multiselect('انتخاب فروشنده:', options=sellers, default=sellers[0])

        if not seller_values or not channel_values:
            st.warning('حداقل یک کارشناس و یک کانال را انتخاب کنید.')
        else:
            filtered_deals = monthly_deals[
                (monthly_deals['deal_owner'].isin(seller_values)) &
                (monthly_deals['deal_source'].isin(channel_values))
            ]
            filtered_shifts = monthly_shifts[monthly_shifts['کارشناس'].isin(seller_values)]
            
            # Display metrics and charts for the filtered data
            display_metrics(filtered_deals, filtered_shifts, selected_channels=channel_values)
            plot_daily_trend(
                df=filtered_deals.groupby('deal_done_date').size().reset_index(name='تعداد'),
                date_col='deal_done_date', value_col='تعداد', title='تعداد معاملات روزانه  ',
                labels={'deal_done_date': 'تاریخ', 'تعداد': 'تعداد معامله'}
            )

            # Combine leads from different channels for the leads chart
            lead_dfs = []
            if 'دایرکت اینستاگرام' in channel_values:
                lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (اینستاگرام)'].sum().reset_index(name="تعداد"))
            if 'تلگرام(سوشال)' in channel_values:
                lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (تلگرام)'].sum().reset_index(name="تعداد"))
            if 'واتساپ(سوشال)' in channel_values:
                lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (واتساپ)'].sum().reset_index(name="تعداد"))
            
            if lead_dfs:
                daily_lead_count = pd.concat(lead_dfs).groupby('تاریخ میلادی')['تعداد'].sum().reset_index()
                plot_daily_trend(
                    df=daily_lead_count, date_col='تاریخ میلادی', value_col='تعداد', title='تعداد لیدهای روزانه  ',
                    labels={'تاریخ میلادی': 'تاریخ', 'تعداد': 'تعداد لید'}
                )
        if not filtered_shifts.empty:
            with st.expander(f'شیفت های {', '.join(str(i) for i in seller_values)}', False):
                st.write(filtered_shifts)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="دانلود داده‌ها به صورت CSV",
                        data=convert_df(filtered_shifts),
                        file_name='shifts.csv',
                        mime='text/csv',
                    )
                with col2:
                    st.download_button(
                        label="دانلود داده‌ها به صورت اکسل",
                        data=convert_df_to_excel(filtered_shifts),
                        file_name='shifts.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )

    else:
        st.subheader("🔍 عملکرد شما")
        channels = monthly_deals['deal_source'].unique().tolist()
        sellers = monthly_deals['deal_owner'].unique().tolist()

        cols = st.columns(2)
        channel_values = cols[0].multiselect("انتخاب کانال فروش", options=channels, default=channels)

        if not channel_values:
            st.warning('حداقل یک کانال را انتخاب کنید.')
        else:
            filtered_deals = monthly_deals[
                (monthly_deals['deal_owner'] == user_filter) &
                (monthly_deals['deal_source'].isin(channel_values))
            ]
            filtered_shifts = monthly_shifts[monthly_shifts['کارشناس'] == user_filter]
            
            # Display metrics and charts for the filtered data
            display_metrics(filtered_deals, filtered_shifts, selected_channels=channel_values)
            plot_daily_trend(
                df=filtered_deals.groupby('deal_done_date').size().reset_index(name='تعداد'),
                date_col='deal_done_date', value_col='تعداد', title='تعداد معاملات روزانه  ',
                labels={'deal_done_date': 'تاریخ', 'تعداد': 'تعداد معامله'}
            )

            # Combine leads from different channels for the leads chart
            lead_dfs = []
            if 'دایرکت اینستاگرام' in channel_values:
                lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (اینستاگرام)'].sum().reset_index(name="تعداد"))
            if 'تلگرام(سوشال)' in channel_values:
                lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (تلگرام)'].sum().reset_index(name="تعداد"))
            if 'واتساپ(سوشال)' in channel_values:
                lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (واتساپ)'].sum().reset_index(name="تعداد"))
            
            if lead_dfs:
                daily_lead_count = pd.concat(lead_dfs).groupby('تاریخ میلادی')['تعداد'].sum().reset_index()
                plot_daily_trend(
                    df=daily_lead_count, date_col='تاریخ میلادی', value_col='تعداد', title='تعداد لیدهای روزانه  ',
                    labels={'تاریخ میلادی': 'تاریخ', 'تعداد': 'تعداد لید'}
                )

        with st.expander('شیفت های شما'):
            st.write(filtered_shifts)

    st.divider()
    
    # Display the reward section for the chosen month's CHECKOUT dates
    deals_for_reward = deals_data[
        (deals_data['checkout_jalali_year_month'] == target_month)&
        (deals_data['deal_type']=='New Sale')
    ].reset_index(drop=True)

    display_reward_section(deals_for_reward, parameters, user_filter=user_filter)

def render_settings_tab(parameters: dict):
    """Renders the settings form for updating reward parameters."""
    with st.form('social_team_parameters_form'):
        st.subheader("⚙️ تنظیم پارامترهای پاداش")
        
        target = st.number_input(
            label="🎯 تارگت فروش ماه (بر اساس تاریخ خروج و به تومان)",
            step=1_000_000,
            value=int(parameters.get('target', 0))
        )
        grow_percent = st.number_input(
            label="📈 درصد پاداش در صورت رسیدن به تارگت",
            help="این درصد زمانی اعمال می‌شود که فروش تیم به ۹۵٪ تارگت یا بیشتر برسد.",
            step=0.1, format="%.1f",
            value=float(parameters.get('grow_percent', 0.0))
        )
        normal_percent = st.number_input(
            label="📉 درصد پاداش در حالت عادی",
            help="این درصد زمانی اعمال می‌شود که فروش تیم کمتر از ۹۵٪ تارگت باشد.",
            step=0.1, format="%.1f",
            value=float(parameters.get('normal_percent', 0.0))
        )
        
        if st.form_submit_button('ذخیره تغییرات'):
            df = pd.DataFrame([{"target": target, "grow_percent": grow_percent, "normal_percent": normal_percent}])
            if write_df_to_sheet(df, sheet_name='Social team parameters'):
                st.success("پارامترها با موفقیت به‌روزرسانی شد.")
                st.rerun()
            else:
                st.error("خطا در هنگام به‌روزرسانی پارامترها!")