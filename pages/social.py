import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
from typing import Optional
from utils.funcs import load_data_cached, handel_errors, download_buttons
from utils.custom_css import apply_custom_css
from utils.sidebar import render_sidebar 
from utils.sheetConnect import write_df_to_sheet, authenticate_google_sheets, load_sheet

# --- Utility Functions ---
def get_username() -> str:
    """Get current username for logging."""
    try:
        return st.session_state.get('userdata', {}).get('name', 'unknown')
    except Exception:
        return 'unknown'

@st.cache_data(ttl=600, show_spinner=False)
def safe_to_jalali(date_value) -> Optional[jdatetime.date]:
    """Convert Gregorian date to Jalali date safely."""
    try:
        if date_value is None or pd.isna(date_value):
            return None
        return jdatetime.date.fromgregorian(date=pd.to_datetime(date_value).date())
    except Exception as e:
        st.write(date_value is None or pd.isna(date_value))
        handel_errors(e, "safe_to_jalali conversion error")
        return None

def get_jalali_month_string(date_obj: jdatetime.date) -> str:
    """Get year-month string from Jalali date."""
    return f"{date_obj.year}-{date_obj.month:02d}"

def get_target_month(month_choice: str) -> str:
    """Get target month string based on user's selection."""
    try:
        today = jdatetime.date.today()
        if month_choice == 'ماه پیش':
            last_month = (today.replace(day=1) - jdatetime.timedelta(days=1))
            return get_jalali_month_string(last_month)
        elif month_choice == 'دو ماه پیش':
            first_of_this_month = today.replace(day=1)
            last_month = first_of_this_month - jdatetime.timedelta(days=1)
            two_months_ago = (last_month.replace(day=1) - jdatetime.timedelta(days=1))
            return get_jalali_month_string(two_months_ago)
        else:
            return get_jalali_month_string(today)

    except Exception as e:
        handel_errors(e, "Error in get_target_month")

# --- Display Functions ---
def display_metrics(deals_df: pd.DataFrame, selected_channels: list = None):
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

        value_sum = deals_df['deal_value'].astype(float).sum() / 10
        number_of_deals = deals_df.shape[0]

        cols = st.columns(2)
        cols[0].metric('💰 میزان فروش', f'{value_sum:,.0f} تومان')
        cols[1].metric('📈 تعداد فروش', f'{number_of_deals:,}')
        
    except Exception as e:
        handel_errors(e, "display_metrics error", show_error=False)

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
        handel_errors(e, "plot_daily_trend error")

def display_reward_section(deals_for_reward: pd.DataFrame, hagh_services_for_reward: pd.DataFrame, parameters: dict, user_filter: str = None):
    """
        Compute and display the reward section, including progress pie and individual metrics.

    Args:
        deals_for_reward: Deals filtered by checkout date in current month.
        hagh_services_for_reward: Hagh services filtered by checkout date in current month.
        parameters: Dict containing target and reward percentages.
        user_filter: If provided, show only that user's rewards; otherwise show team rewards.
    """
    st.subheader('🏆 پاداش عملکرد (بر اساس تاریخ خروج)')
    try:
        if deals_for_reward.empty:
            st.warning('هیچ معامله‌ای با تاریخ خروج در این ماه برای محاسبه پاداش ثبت نشده است.')
            return

        target = int(parameters.get('record', 0))
        # Values are in Toman (divided by 10)
        current_value = deals_for_reward['deal_value'].astype(int).sum() / 10

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
            hagh_services_for_reward['checkout_jalali_str'] = hagh_services_for_reward['checkout_jalali'].apply(
                lambda x: x.strftime('%Y/%m/%d') if x else ""
            )
        except Exception as e:
            handel_errors(e, "Error generating checkout_jalali_str")

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
            st.plotly_chart(fig, config={'responsive': True})
        except Exception as e:
            handel_errors(e, "Error drawing reward progress pie")

        # --- Team Member Reward Table ---
        if not deals_for_reward.empty and user_filter is None:
            try:
                deals_for_reward['deal_value'] = deals_for_reward['deal_value'].astype(float)
                member_stats = (
                    deals_for_reward.groupby('deal_owner')
                    .agg(
                        تعداد_معامله=('deal_id', 'count'),
                        میزان_فروش=('deal_value', lambda x: x.sum() / 10)
                    )
                    .reset_index()
                )
                # calculate hagh service reward for each member
                member_hagh_service =( 
                    hagh_services_for_reward.groupby('deal_owner')
                   .agg(
                        تعداد_حق_سرویس=('final_amount', 'count'),
                        مجموع_حق_سرویس=('final_amount', lambda x: x.sum()/10)
                   ).reset_index()
                )
                # map member_hagh_service  to member stats
                member_stats = member_stats.merge(
                    member_hagh_service, left_on='deal_owner', right_on='deal_owner', how='left',
                ).fillna(0)

                member_stats['پاداش'] = (member_stats['میزان_فروش'] * float(reward_percent) / 100) + (member_stats['مجموع_حق_سرویس'] * 0.1)                
                
                member_stats = member_stats.rename(
                    columns={'deal_owner': 'کارشناس'}
                ).sort_values(by='تعداد_معامله', ascending=False)
                st.markdown("#### جدول پاداش اعضای تیم")
                st.dataframe(member_stats.style.format({'میزان_فروش': '{:,.0f}', 'پاداش': '{:,.0f}', 'مجموع_حق_سرویس': '{:,.0f}', 'تعداد_حق_سرویس': '{:,.0f}'}), width='stretch')

                download_buttons(member_stats, 'team_reward')

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
                                'deal_id': 'کد معامله',
                                'deal_title': 'عنوان معامله',
                                'deal_value': 'مبلغ معامله',
                                'deal_created_time': 'تاریخ ایجاد معامله',
                                'deal_owner': 'کارشناس',
                                'deal_source': 'کانال فروش',
                                'contact_id': 'کد مشتری',
                                'checkout': 'تاریخ خروج',
                                'checkout_jalali_str': 'تاریخ خروج (شمسی)'
                            }
                        ),
                        width='stretch'
                    )
                    download_buttons(deals_for_reward, 'team_reward_deals')
                with st.expander('جزئیات حق سرویس برای پاداش'):
                    st.dataframe(
                        hagh_services_for_reward[
                            [
                                'deal_id', 'final_amount', 'deal_owner', 'checkout_jalali_str'
                            ]
                        ].rename(
                            columns={
                                'deal_id': 'کد معامله',
                                'final_amount': 'مبلغ حق سرویس',
                                'deal_owner': 'کارشناس',
                                'checkout_jalali_str': 'تاریخ خروج (شمسی)'
                            }
                        ),
                        width='stretch'
                    )
                    download_buttons(hagh_services_for_reward, 'team_hagh_services')
            except Exception as e:
                handel_errors(e, "display_reward_section team member reward table error")

        # --- Individual Reward Display ---
        if user_filter:
            selected_member = user_filter
            st.markdown(f"#### پاداش شما ({selected_member})")
        else:
            sellers = deals_for_reward['deal_owner'].unique().tolist()
            selected_member = st.selectbox("انتخاب کارشناس برای مشاهده پاداش:", sellers, key="select_expert_reward")

        if selected_member:
            try:
                
                member_deals = deals_for_reward[deals_for_reward['deal_owner'] == selected_member]
                member_value = float(member_deals['deal_value'].astype(float).sum() / 10)
                member_reward = member_value * float(reward_percent) / 100

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
                            'deal_id': 'کد معامله',
                            'deal_title': 'عنوان معامله',
                            'deal_value': 'مبلغ معامله',
                            'deal_created_time': 'تاریخ ایجاد معامله',
                            'deal_owner': 'کارشناس',
                            'deal_source': 'کانال فروش',
                            'contact_id': 'کد مشتری',
                            'checkout': 'تاریخ خروج',
                            'checkout_jalali_str': 'تاریخ خروج (شمسی)'
                        }
                    ).reset_index(drop=True)
                    st.dataframe(data_to_write, width='stretch')
                    download_buttons(data_to_write, f'{selected_member}-deals')
            except Exception as e:
                handel_errors(e, f"display_reward_section individual member error: {selected_member}")
    except Exception as e:
        handel_errors(e, "display_reward_section: general error")

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
        # load hagh services
        if 'hagh_services' not in st.session_state:
            try:
                hagh_services = load_sheet(key='DEALS_SPREADSHEET_ID', sheet_name='حق سرویس')
                st.session_state.hagh_services = hagh_services
            except Exception as e:
                handel_errors(e, "Error loading Hagh services data")
        
        hagh_services = st.session_state.get('hagh_services', pd.DataFrame())

        data = data[
            (data['deal_source'].isin(['دایرکت اینستاگرام', 'تلگرام(سوشال)', 'واتساپ(سوشال)'])) &
            (data['deal_type'].isin(['فروش جدید', 'فروش تمدید'])) &
            (data['deal_status'] == 'Won')
        ].copy()

        # For rewards, we need Jalali dates based on the checkout
        data['checkout_jalali'] = data['checkout'].apply(safe_to_jalali)
        hagh_services['checkout_jalali'] = hagh_services['checkout'].apply(safe_to_jalali)
        data['checkout_jalali_year_month'] = data['checkout_jalali'].apply(lambda d: f"{d.year}-{d.month:02d}" if d else "")

        # For general stats, we use the deal_created_time
        data['deal_created_time'] = pd.to_datetime(data['deal_created_time']).dt.date
        data['jalali_date'] = data['deal_created_time'].apply(safe_to_jalali)
        data['jalali_year_month'] = data['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}" if d else "")

        # Load parameters and shift data
        parametrs_df = pd.DataFrame()
        try:
            parametrs_df = load_data_cached(spreadsheet_key='MAIN_SPREADSHEET_ID', sheet_name='Social team parameters')
        except Exception as e:
            handel_errors(e, "Error loading Social team parameters")
        parameters = parametrs_df.iloc[0].to_dict() if not parametrs_df.empty else {}

        # --- 3. UI Rendering ---
        if is_manager:
            # Show admin/manager tabs
            tabs = st.tabs(['داشبورد اصلی', 'تنظیمات پاداش'])
            with tabs[0]:
                render_dashboard(data, hagh_services, parameters)
            with tabs[1]:
                render_settings_tab(parameters, data)
        else:
            # Non-manager user: show only their dashboard
            render_dashboard(data, hagh_services, parameters, user_filter=name)
            
    except Exception as e:
        handel_errors(e, "social main function error")

def render_dashboard(deals_data: pd.DataFrame, hagh_services, parameters: dict, user_filter: str = None):
    """
    Render the main dashboard. Can produce admin view or regular user view.

    Args:
        deals_data: DataFrame of all deals (team or individual).
        hagh_services: DataFrame of hagh services.
        parameters: Dict of reward config.
        user_filter: If set, restricts to a specific expert.
    """
    
    try:
        month_choice = st.selectbox('ماه مورد نظر را انتخاب کنید:', ['این ماه', 'ماه پیش', 'دو ماه پیش'], key="select_month_dashboard")
        target_month = get_target_month(month_choice)
        st.info(f'آمار ماه: {target_month}')

        # Filter deals by selected month and new sale type
        monthly_deals = deals_data[
            (deals_data['jalali_year_month'] == target_month) &
            (deals_data['deal_type'] == 'فروش جدید')
        ]

        # Prepare deals for reward, based on checkout month
        deals_for_reward = deals_data[
            (deals_data['checkout_jalali_year_month'] == target_month) &
            (deals_data['deal_type'] == 'فروش جدید')
        ].reset_index(drop=True)

        hagh_services_for_reward = hagh_services[
            (hagh_services['deal_id'].isin(deals_for_reward['deal_id'])) 
        ].reset_index(drop=True)

        hagh_services_for_reward['final_amount'] = hagh_services_for_reward['final_amount'].astype(int)

        display_reward_section(deals_for_reward, hagh_services_for_reward, parameters, user_filter=user_filter)

        st.divider()
        st.subheader("عملکرد کلی تیم")
        display_metrics(monthly_deals)
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

                # Display filtered metrics and trend charts
                display_metrics(filtered_deals, selected_channels=channel_values)
                plot_daily_trend(
                    df=filtered_deals.groupby('deal_created_time').size().reset_index(name='تعداد'),
                    date_col='deal_created_time', value_col='تعداد', title='تعداد معاملات روزانه  ',
                    labels={'deal_created_time': 'تاریخ', 'تعداد': 'تعداد معامله'}
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
                # Display filtered metrics and trend charts
                display_metrics(filtered_deals, selected_channels=channel_values)
                plot_daily_trend(
                    df=filtered_deals.groupby('deal_created_time').size().reset_index(name='تعداد'),
                    date_col='deal_created_time', value_col='تعداد', title='تعداد معاملات روزانه  ',
                    labels={'deal_created_time': 'تاریخ', 'تعداد': 'تعداد معامله'}
                )
    except Exception as e:
        handel_errors(e, "render_dashboard error")

def render_settings_tab(parameters: dict, deals_data: pd.DataFrame):
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
            monthes = set([
                f"{year}-{month:02d}" for year in range(1404, 1406) for month in range(1, 13)
            ])
            record_month = st.selectbox(
                label="📅 ماه رکورد",
                options=sorted(monthes),
                index=sorted(monthes).index(parameters.get('record_month', 0)) if parameters.get('record_month', 0) in monthes else 0
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
            # check recent 3 month if record sales is changed or not if it changed update the row
            deals_data['deal_value'] = deals_data['deal_value'].astype(float)/10
            month_records = deals_data.groupby('checkout_jalali_year_month')['deal_value'].sum().reset_index(name='deal_value_sum')
            
            updated_record = month_records[month_records['deal_value_sum'] > record]

            if not updated_record.empty:
                record = updated_record['deal_value_sum'].max()
                record_month = updated_record[updated_record['deal_value_sum'] == record]['checkout_jalali_year_month'].values[0]
                st.info(f"رکورد فروش ماه به‌روزرسانی خواهد شد: {record:,.0f} تومان در ماه {record_month}")

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
                    success = write_df_to_sheet(authenticate_google_sheets(), 'MAIN_SPREADSHEET_ID', 'Social team parameters', df, clear_existing=True)

                except Exception as e:
                    handel_errors(e, "Failed to write Social team parameters")
                if success:
                    st.success("پارامترها با موفقیت به‌روزرسانی شد.")
                    st.rerun()
                else:
                    handel_errors(Exception("Failed to update team parameters"), "Failed to update team parameters")

        except Exception as e:
            handel_errors(e, "render_settings_tab error")

if __name__ == "__main__":
    social()