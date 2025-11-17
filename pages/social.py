import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
from utils.write_data import write_df_to_sheet
from utils.load_data import load_sheet, load_sheet_uncache
from utils.func import convert_df, convert_df_to_excel
from utils.logger import log_event
from utils.custom_css import apply_custom_css
from utils.sidebar import render_sidebar

def _get_username():
    """Helper to get current user for logging."""
    try:
        return st.session_state.get('userdata', {}).get('name', 'unknown')
    except Exception:
        return 'unknown'

# --- Data Transformation & Helper Functions ---
def normalize_owner(owner: str) -> str:
    """
    Standardize the expert's name to a canonical format using a mapping dictionary.
    Reduces errors and improves readability by mapping known string variants to a fixed name.
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
    try:
        return name_map.get(owner, owner)
    except Exception as e:
        log_event(_get_username(), 'error', f"Error normalizing owner '{owner}': {e}")
        return owner

@st.cache_data(ttl=600)
def safe_to_jalali(gregorian_date):
    """
    Safely converts Gregorian date to Jalali (Shamsi) date. Uses Streamlit cache for performance.
    """
    try:
        return jdatetime.date.fromgregorian(date=pd.to_datetime(gregorian_date).date())
    except Exception as e:
        log_event(_get_username(), 'error', f"safe_to_jalali({gregorian_date}) {e}")
        return None

def get_month_filter_string(month_choice: str) -> str:
    """
    Get the Jalali (Shamsi) year-month string (YYYY-MM) based on user's dropdown selection.

    Args:
        month_choice: The string value selected by the user (such as 'این ماه', 'ماه پیش', 'دو ماه پیش')

    Returns:
        Target Jalali year-month string, e.g. "1403-05"
    """
    try:
        today_jalali = jdatetime.date.today()
        if month_choice == 'ماه پیش':
            # Get last day of last month
            first_day_of_current_month = today_jalali.replace(day=1)
            last_day_of_previous_month = first_day_of_current_month - jdatetime.timedelta(days=1)
            return f"{last_day_of_previous_month.year}-{last_day_of_previous_month.month:02d}"
        elif month_choice == "دو ماه پیش":
            # Go back two months
            first_day_of_current_month = today_jalali.replace(day=1)
            last_day_of_previous_month = first_day_of_current_month - jdatetime.timedelta(days=1)
            first_day_of_previous_month = last_day_of_previous_month.replace(day=1)
            last_day_of_two_months_ago = first_day_of_previous_month - jdatetime.timedelta(days=1)
            return f"{last_day_of_two_months_ago.year}-{last_day_of_two_months_ago.month:02d}"
        else:  # Default: this month
            return f"{today_jalali.year}-{today_jalali.month:02d}"
    except Exception as e:
        log_event(_get_username(), 'error', f"Error in get_month_filter_string('{month_choice}') {e}")
        return ""

# --- UI Component Functions ---

def display_metrics(deals_df: pd.DataFrame, shifts_df: pd.DataFrame, selected_channels: list = None):
    """
    Calculate and display main KPIs in Streamlit columns.
    Args:
        deals_df: DataFrame of sale deals for the selected period.
        shifts_df: DataFrame of shifts and lead counts for the period.
        selected_channels: List of channels for lead counting, or None for all channels.
    """
    try:
        if deals_df.empty:
            st.info('هیچ معامله‌ای برای نمایش آمار وجود ندارد.')
            return

        value_sum = deals_df['deal_value'].sum() / 10
        number_of_deals = deals_df.shape[0]

        # Calculate lead count depending on selected channels
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
    except Exception as e:
        log_event(_get_username(), 'error', f"display_metrics error: {e}")

def plot_daily_trend(df: pd.DataFrame, date_col: str, value_col: str, title: str, labels: dict):
    """
    Generate and display a daily trend line chart using Plotly.

    Args:
        df: DataFrame with the relevant data.
        date_col: Name of the date column.
        value_col: Name of the metric column.
        title: Title of the chart.
        labels: Dict for custom axis labels in the chart.
    """
    try:
        if df.empty:
            # No data to display chart for 'title'
            return

        df[date_col] = pd.to_datetime(df[date_col])

        # Ensure the full daily range appears, even for missing days
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
    except Exception as e:
        log_event(_get_username(), 'error', f"plot_daily_trend error: {e}")

def display_reward_section(deals_for_reward: pd.DataFrame, parameters: dict, user_filter: str = None):
    """
    Compute and display the reward section, including progress pie and individual metrics.

    Args:
        deals_for_reward: Deals filtered by checkout date in current month.
        parameters: Dict containing target and reward percentages.
        user_filter: If provided, show only that user's rewards; otherwise show team rewards.
    """
    st.subheader('🏆 پاداش عملکرد (بر اساس تاریخ خروج)')
    try:
        if deals_for_reward.empty:
            st.warning('هیچ معامله‌ای با تاریخ خروج در این ماه برای محاسبه پاداش ثبت نشده است.')
            return

        target = parameters.get('record', 0)
        # Values are in Toman (divided by 10)
        current_value = deals_for_reward['deal_value'].sum() / 10

        # Reward progress percentage (max 100%)
        if target > 0:
            display_percentage = min((current_value / target) * 100, 100.0)
        else:
            display_percentage = 0

        # Grow reward percent if above 95% of target
        reward_percent = (
            parameters.get('grow_percent', 0)
            if target > 0 and current_value >= target * 0.95
            else parameters.get('normal_percent', 0)
        )
        deals_count = deals_for_reward.shape[0]
        col1, col2, col3, col4 = st.columns(4)

        col1.metric('🎯 تارگت فروش', value=f'{target:,.0f} تومان')
        col2.metric('تعداد فروش', value=deals_count)
        col3.metric('میزان فروش', value=f"{current_value:,.0f} تومان")
        col4.metric('میانگین مبلغ معامله', value=f"{(current_value / deals_count):,.2f} تومان" if deals_count > 0 else "۰ تومان")

        # Add checkout Jalali date string for display
        try:
            deals_for_reward['checkout_jalali_str'] = deals_for_reward['checkout_jalali'].apply(
                lambda x: x.strftime('%Y/%m/%d') if x else ""
            )
        except Exception as e:
            log_event(_get_username(), 'error', f"Error generating checkout_jalali_str: {e}")

        # --- Progress Pie Visualization ---
        st.subheader("میزان پیشرفت ")
        try:
            fig = go.Figure()
            fig.add_trace(
                go.Pie(
                    values=[display_percentage, 100 - display_percentage],
                    hole=.8,
                    marker_colors=[
                        '#00FF00' if display_percentage >= 100 else '#00FF00',
                        '#E5ECF6'
                    ],
                    showlegend=False,
                    textinfo='none',
                    rotation=90,
                    pull=[0.1, 0],
                )
            )
            fig.update_layout(
                annotations=[
                    dict(
                        text=f'{display_percentage:.1f}%', x=0.5, y=0.5,
                        font_size=24, font_color='#2F4053', showarrow=False
                    ),
                    dict(
                        text='تکمیل شده' if display_percentage >= 100 else 'در حال پیشرفت',
                        x=0.5, y=0.35, font_size=14, font_color='#2E4053', showarrow=False
                    )
                ],
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            log_event(_get_username(), 'error', f"Error drawing reward progress pie: {e}")

        # --- Team Member Reward Table ---
        if not deals_for_reward.empty and user_filter is None:
            try:
                member_stats = (
                    deals_for_reward.groupby('deal_owner')
                    .agg(
                        تعداد_معامله=('deal_id', 'count'),
                        میزان_فروش=('deal_value', lambda x: x.sum() / 10)
                    )
                    .reset_index()
                )
                member_stats['پاداش'] = member_stats['میزان_فروش'] * reward_percent / 100
                member_stats = member_stats.rename(
                    columns={'deal_owner': 'کارشناس'}
                ).sort_values(by='تعداد_معامله', ascending=False)
                st.markdown("#### جدول پاداش اعضای تیم")
                st.dataframe(member_stats.style.format({'میزان_فروش': '{:,.0f}', 'پاداش': '{:,.0f}'}), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="دانلود داده‌ها به صورت CSV",
                        data=convert_df(member_stats),
                        file_name='team-reward.csv',
                        mime='text/csv',
                    )
                with col2:
                    st.download_button(
                        label="دانلود داده‌ها به صورت اکسل",
                        data=convert_df_to_excel(member_stats),
                        file_name='team-reward.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )

                with st.expander('جزئیات معاملات برای پاداش'):
                    st.dataframe(
                        deals_for_reward[
                            [
                                'deal_id', 'deal_title', 'deal_value', 'deal_created_time',
                                'deal_owner', 'deal_source', 'contact_id',
                                'checkout', 'checkout_jalali_str'
                            ]
                        ].rename(
                            columns={
                                'deal_id': 'شناسه معامله',
                                'deal_title': 'عنوان معامله',
                                'deal_value': 'مبلغ معامله',
                                'deal_created_time': 'تاریخ ایجاد معامله',
                                'deal_owner': 'کارشناس',
                                'deal_source': 'کانال فروش',
                                'contact_id': 'شناسه مشتری',
                                'checkout': 'تاریخ خروج',
                                'checkout_jalali_str': 'تاریخ خروج (شمسی)'
                            }
                        ),
                        use_container_width=True
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="دانلود داده‌ها به صورت CSV",
                            data=convert_df(deals_for_reward),
                            file_name='deals-for-reward.csv',
                            mime='text/csv',
                        )
                    with col2:
                        st.download_button(
                            label="دانلود داده‌ها به صورت اکسل",
                            data=convert_df_to_excel(deals_for_reward),
                            file_name='deals-for-reward.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        )
            except Exception as e:
                log_event(_get_username(), 'error', f"display_reward_section team member reward table error: {e}")

        # --- Individual Reward Display ---
        if user_filter:
            selected_member = user_filter
            st.markdown(f"#### پاداش شما ({selected_member})")
        else:
            sellers = deals_for_reward['deal_owner'].unique().tolist()
            # FIX bug: Do not pass 'label' as kwarg twice!
            selected_member = st.selectbox("انتخاب کارشناس برای مشاهده پاداش:", sellers, key="select_expert_reward")

        if selected_member:
            try:
                member_deals = deals_for_reward[deals_for_reward['deal_owner'] == selected_member]
                member_value = member_deals['deal_value'].sum() / 10
                member_reward = member_value * reward_percent / 100

                cols = st.columns(2)
                cols[0].metric(f'میزان فروش {selected_member}', value=f'{member_value:,.0f} تومان')
                cols[1].metric(f'💰 میزان پاداش {selected_member}', value=f'{member_reward:,.0f} تومان')

                with st.expander(f'جزئیات معامله های {selected_member}', False):
                    data_to_write = member_deals[
                        [
                            'deal_id', 'deal_title', 'deal_value', 'deal_created_time',
                            'deal_owner', 'deal_source', 'contact_id',
                            'checkout', 'checkout_jalali_str'
                        ]
                    ].rename(
                        columns={
                            'deal_id': 'شناسه معامله',
                            'deal_title': 'عنوان معامله',
                            'deal_value': 'مبلغ معامله',
                            'deal_created_time': 'تاریخ ایجاد معامله',
                            'deal_owner': 'کارشناس',
                            'deal_source': 'کانال فروش',
                            'contact_id': 'شناسه مشتری',
                            'checkout': 'تاریخ خروج',
                            'checkout_jalali_str': 'تاریخ خروج (شمسی)'
                        }
                    ).reset_index(drop=True)
                    st.dataframe(data_to_write, use_container_width=True)
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
            except Exception as e:
                log_event(_get_username(), 'error', f"display_reward_section individual member error: {selected_member} - {e}")

    except Exception as e:
        log_event(_get_username(), 'error', f"display_reward_section: {e}")

# ----------- Main App Function -----------
def social():
    """
    Main entry function to render the Social team dashboard in Streamlit. Handles user authentication, loading data, and providing manager or user dashboards.
    """
    apply_custom_css()
    render_sidebar()

    st.title("📊 داشبورد تیم Social")
    try:
        # --- 1. Authentication and Initialization ---
        if 'logged_in' not in st.session_state or not st.session_state.logged_in:
            st.warning("لطفا ابتدا وارد سیستم شوید")
            return

        role = st.session_state.userdata.get('role', '')
        teams = st.session_state.userdata.get('team', '')
        teams_list = [team.strip() for team in teams.split('|')]
        name = st.session_state.userdata.get('name', '')

        if 'social' not in teams_list:
            st.error("شما به این بخش دسترسی ندارید")
            return

        is_manager = role in ["admin", "manager"]

        # --- 2. Data Loading and Pre-processing ---
        data = st.session_state.deals_data.copy()
        data = data[data['deal_source'].isin(['دایرکت اینستاگرام', 'تلگرام(سوشال)', 'واتساپ(سوشال)'])].copy()

        data['deal_owner'] = data['deal_owner'].apply(normalize_owner)
        data = data[~data['deal_owner'].isin(['', 'نامشخص', 'Han Rez', 'بابک  مسعودی'])]
        # For rewards, we need Jalali dates based on the checkout
        data['checkout_jalali'] = data['checkout'].apply(safe_to_jalali)
        data['checkout_jalali_year_month'] = data['checkout_jalali'].apply(lambda d: f"{d.year}-{d.month:02d}" if d else "")

        # For general stats, we use the deal_created_time
        data['deal_created_time'] = pd.to_datetime(data['deal_created_time']).dt.date
        data['jalali_date'] = data['deal_created_time'].apply(safe_to_jalali)
        data['jalali_year_month'] = data['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}" if d else "")

        # Load parameters and shift data
        parametrs_df = pd.DataFrame()
        try:
            parametrs_df = load_sheet_uncache('Social team parameters')
        except Exception as e:
            log_event(_get_username(), 'error', f"Error loading Social team parameters: {e}")
        parameters = parametrs_df.iloc[0].to_dict() if not parametrs_df.empty else {}

        shift_sheet = pd.DataFrame()
        try:
            shift_sheet = load_sheet('Social shift')
            shift_sheet['تاریخ میلادی'] = pd.to_datetime(shift_sheet['تاریخ میلادی'])
            shift_sheet['jalali_date'] = shift_sheet['تاریخ میلادی'].apply(safe_to_jalali)
            shift_sheet['jalali_year_month'] = shift_sheet['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}" if d else "")
        except Exception as e:
            log_event(_get_username(), 'error', f"Error loading Social shift sheet: {e}")

        # --- 3. UI Rendering ---
        if is_manager:
            # Show admin/manager tabs
            tabs = st.tabs(['داشبورد اصلی', 'تنظیمات پاداش'])
            with tabs[0]:
                render_dashboard(data, shift_sheet, parameters)
            with tabs[1]:
                render_settings_tab(parameters)
        else:
            # Non-manager user: show only their dashboard
            render_dashboard(data, shift_sheet, parameters, user_filter=name)
    except Exception as e:
        log_event(_get_username(), 'error', f"social()main error: {e}")
        st.error("خطای پیش‌بینی نشده‌ای رخ داده است.")

def render_dashboard(deals_data: pd.DataFrame, shift_data: pd.DataFrame, parameters: dict, user_filter: str = None):
    """
    Render the main dashboard. Can produce admin view or regular user view.

    Args:
        deals_data: DataFrame of all deals (team or individual).
        shift_data: DataFrame of shift logs.
        parameters: Dict of reward config.
        user_filter: If set, restricts to a specific expert.
    """
    try:
        month_choice = st.selectbox('ماه مورد نظر را انتخاب کنید:', ['این ماه', 'ماه پیش', 'دو ماه پیش'], key="select_month_dashboard")
        target_month = get_month_filter_string(month_choice)
        st.info(f'آمار ماه: {target_month}')

        # Filter deals by selected month and new sale type
        monthly_deals = deals_data[
            (deals_data['jalali_year_month'] == target_month) &
            (deals_data['deal_type'] == 'New Sale')
        ]
        monthly_shifts = shift_data[shift_data['jalali_year_month'] == target_month]

        # Prepare deals for reward, based on checkout month
        deals_for_reward = deals_data[
            (deals_data['checkout_jalali_year_month'] == target_month) &
            (deals_data['deal_type'] == 'New Sale')
        ].reset_index(drop=True)

        display_reward_section(deals_for_reward, parameters, user_filter=user_filter)

        st.divider()
        st.subheader("عملکرد کلی تیم")
        display_metrics(monthly_deals, monthly_shifts)
        plot_daily_trend(
            df=monthly_deals.groupby('deal_created_time').size().reset_index(name='تعداد'),
            date_col='deal_created_time',
            value_col='تعداد',
            title='تعداد معاملات روزانه',
            labels={'deal_created_time': 'تاریخ', 'تعداد': 'تعداد معامله'}
        )

        st.divider()

        # Filters are for manager only
        if not user_filter:
            st.subheader("🔍 فیلتر و بررسی جزئیات")
            channels = monthly_deals['deal_source'].unique().tolist()
            sellers = monthly_deals['deal_owner'].unique().tolist()

            cols = st.columns(2)
            channel_values = cols[0].multiselect("انتخاب کانال فروش", options=channels, default=channels)
            seller_values = cols[1].multiselect('انتخاب فروشنده:', options=sellers, default=[sellers[0]] if sellers else None)

            if not seller_values or not channel_values:
                st.warning('حداقل یک کارشناس و یک کانال را انتخاب کنید.')
            else:
                filtered_deals = monthly_deals[
                    (monthly_deals['deal_owner'].isin(seller_values)) &
                    (monthly_deals['deal_source'].isin(channel_values))
                ]
                filtered_shifts = monthly_shifts[monthly_shifts['کارشناس'].isin(seller_values)]

                # Display filtered metrics and trend charts
                display_metrics(filtered_deals, filtered_shifts, selected_channels=channel_values)
                plot_daily_trend(
                    df=filtered_deals.groupby('deal_created_time').size().reset_index(name='تعداد'),
                    date_col='deal_created_time', value_col='تعداد', title='تعداد معاملات روزانه  ',
                    labels={'deal_created_time': 'تاریخ', 'تعداد': 'تعداد معامله'}
                )

                # Build daily lead chart combining channels
                lead_dfs = []
                if 'دایرکت اینستاگرام' in channel_values:
                    lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (اینستاگرام)'].sum().reset_index(name="تعداد"))
                if 'تلگرام(سوشال)' in channel_values:
                    lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (تلگرام)'].sum().reset_index(name="تعداد"))
                if 'واتساپ(سوشال)' in channel_values:
                    lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (واتساپ)'].sum().reset_index(name="تعداد"))

                if lead_dfs:
                    try:
                        daily_lead_count = pd.concat(lead_dfs).groupby('تاریخ میلادی')['تعداد'].sum().reset_index()
                        plot_daily_trend(
                            df=daily_lead_count, date_col='تاریخ میلادی', value_col='تعداد', title='تعداد لیدهای روزانه  ',
                            labels={'تاریخ میلادی': 'تاریخ', 'تعداد': 'تعداد لید'}
                        )
                    except Exception as e:
                        log_event(_get_username(), 'error', f"Error plotting daily_lead_count: {e}")
            if not filtered_shifts.empty:
                try:
                    with st.expander(f'شیفت های {", ".join(str(i) for i in seller_values)}', False):
                        st.dataframe(filtered_shifts, use_container_width=True)
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
                except Exception as e:
                    log_event(_get_username(), 'error', f"Error displaying filtered_shifts: {e}")

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

                # Display filtered metrics and trend charts
                display_metrics(filtered_deals, filtered_shifts, selected_channels=channel_values)
                plot_daily_trend(
                    df=filtered_deals.groupby('deal_created_time').size().reset_index(name='تعداد'),
                    date_col='deal_created_time', value_col='تعداد', title='تعداد معاملات روزانه  ',
                    labels={'deal_created_time': 'تاریخ', 'تعداد': 'تعداد معامله'}
                )

                # Build combined daily leads chart
                lead_dfs = []
                if 'دایرکت اینستاگرام' in channel_values:
                    lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (اینستاگرام)'].sum().reset_index(name="تعداد"))
                if 'تلگرام(سوشال)' in channel_values:
                    lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (تلگرام)'].sum().reset_index(name="تعداد"))
                if 'واتساپ(سوشال)' in channel_values:
                    lead_dfs.append(filtered_shifts.groupby('تاریخ میلادی')['تعداد پیام (واتساپ)'].sum().reset_index(name="تعداد"))

                if lead_dfs:
                    try:
                        daily_lead_count = pd.concat(lead_dfs).groupby('تاریخ میلادی')['تعداد'].sum().reset_index()
                        plot_daily_trend(
                            df=daily_lead_count, date_col='تاریخ میلادی', value_col='تعداد', title='تعداد لیدهای روزانه  ',
                            labels={'تاریخ میلادی': 'تاریخ', 'تعداد': 'تعداد لید'}
                        )
                    except Exception as e:
                        log_event(_get_username(), 'error', f"Error plotting user daily_lead_count: {e}")

            with st.expander('شیفت های شما'):
                st.dataframe(filtered_shifts, use_container_width=True)
    except Exception as e:
        log_event(_get_username(), 'error', f"render_dashboard error: {e}")

def render_settings_tab(parameters: dict):
    """
    Render the form for editing reward parameters, visible to managers only.
    """
    with st.form('social_team_parameters_form'):
        st.subheader("⚙️ تنظیم پارامترهای پاداش")
        try:
            st.metric('تارگت فعلی:', f"{int(parameters.get('target', 0)):,.0f} تومان")

            record = st.number_input(
                label="🏅 رکورد فروش ماه (بر اساس تاریخ خروج و به تومان)",
                step=1_000_000,
                value=int(parameters.get('record', 0))
            )
            record_month = st.text_input(
                label="📅 ماه رکورد",
                value=parameters.get('record_month', 0)
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
                df = pd.DataFrame([{
                    "target": int(parameters.get('target', 0)),
                    "grow_percent": grow_percent,
                    "normal_percent": normal_percent,
                    "record": record,
                    "record_month": record_month
                    }])
                success = False
                try:
                    success = write_df_to_sheet(df, sheet_name='Social team parameters')
                except Exception as e:
                    log_event(_get_username(), 'error', f"Error writing params to sheet: {e}")
                if success:
                    st.success("پارامترها با موفقیت به‌روزرسانی شد.")
                    st.rerun()
                else:
                    log_event(_get_username(), 'error', "Failed to update team parameters!")
                    st.error("خطا در هنگام به‌روزرسانی پارامترها!")
        except Exception as e:
            log_event(_get_username(), 'error', f"render_settings_tab error: {e}")

if __name__ == "__main__":
    social()