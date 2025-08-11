import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_nej_datepicker import datepicker_component, Config
import jdatetime

from utils.write_sheet import write_df_to_sheet
from utils.load_sheet import load_sheet
from utils.load_bq import exacute_query

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

        tabs = st.tabs(['صفحه اصلی', 'تنظیمات'])
        with tabs[0]:
            # Load reward parameters from sheet
            target_config = load_sheet('Social team parameters')

            # Convert 'checkout_date' to Jalali date
            @st.cache_data(ttl=600)
            def safe_to_jalali(x):
                return jdatetime.date.fromgregorian(date=pd.to_datetime(x).date())

            data['jalali_date'] = data['checkout_date'].apply(safe_to_jalali)
            # Create year-month columns for grouping and labeling
            data['jalali_year_month'] = data['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}")
            data['jalali_year_month_label'] = data['jalali_date'].apply(lambda d: f"{d.year}/{d.month:02d}")

            # Aggregate deal_value by Jalali year-month
            monthly_sum = data.groupby(['jalali_year_month', 'jalali_year_month_label'])['deal_value'].sum().reset_index()
            monthly_sum = monthly_sum.sort_values('jalali_year_month')

            # Bar chart: monthly deal values
            fig = px.bar(
                monthly_sum,
                x='jalali_year_month_label',
                y='deal_value',
                labels={'jalali_year_month_label': 'ماه جلالی', 'deal_value': 'مجموع ارزش معاملات'},
                title='مجموع ارزش معاملات به تفکیک ماه جلالی'
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Reward calculation and visualizations ---
            if not monthly_sum.empty:
                # Get current (latest) month and its value
                current_month = monthly_sum['jalali_year_month'].max()
                current_value = monthly_sum.loc[monthly_sum['jalali_year_month'] == current_month, 'deal_value'].sum()

                # Get previous months (exclude current)
                previous_months = monthly_sum[monthly_sum['jalali_year_month'] != current_month]
                if not previous_months.empty:
                    # Find best previous month (max deal_value)
                    idx_max = previous_months['deal_value'].idxmax()
                    best_month = previous_months.loc[idx_max, 'jalali_year_month']
                    best_value = previous_months.loc[idx_max, 'deal_value']

                    # Calculate percent of target (current vs 95% of best previous)
                    if best_value > 0:
                        percent_of_target = (current_value / (0.95 * best_value)) * 100
                        percent_of_target = min(percent_of_target, 100)
                        percent_needed = max(0, 100 - percent_of_target)
                    else:
                        percent_of_target = 0
                        percent_needed = 100

                    # Determine reward percent based on performance
                    if best_value > 0 and current_value >= 0.95 * best_value:
                        reward_percent = 0.06
                    else:
                        reward_percent = 0.015
                    reward = current_value * reward_percent

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
                        title={'text': "درصد تحقق هدف (بر اساس بهترین ماه قبلی)"}
                    ))
                    st.plotly_chart(gauge_fig, use_container_width=True)

                    # 2. Bar: compare current and best previous month
                    bar_df = pd.DataFrame({
                        "ماه": [
                            f"{int(best_month.split('-')[0])}/{int(best_month.split('-')[1]):02d}",
                            f"{int(current_month.split('-')[0])}/{int(current_month.split('-')[1]):02d}"
                        ],
                        "ارزش معاملات": [best_value, current_value]
                    })
                    bar_fig = px.bar(
                        bar_df,
                        x="ماه",
                        y="ارزش معاملات",
                        text="ارزش معاملات",
                        color="ماه",
                        color_discrete_sequence=["#b0b0b0", "#4b9cff"],
                        title="مقایسه ارزش معاملات ماه جاری و بهترین ماه قبلی"
                    )
                    bar_fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    bar_fig.update_layout(showlegend=False, yaxis_title="ارزش معاملات (ریال)")
                    st.plotly_chart(bar_fig, use_container_width=True)

                    # 3. Show reward and info
                    st.info(
                        f"پاداش این ماه: {reward:,.0f} ریال "
                        f"({reward_percent*100:.1f}% از مجموع ارزش معاملات {current_value:,.0f} ریال)"
                    )
                    if best_value > 0 and current_value < 0.95 * best_value:
                        st.warning(
                            f"برای رسیدن به هدف بهترین ماه قبلی ({0.95*best_value:,.0f} ریال)، "
                            f"{percent_needed:.1f}% دیگر (حدود {(0.95*best_value-current_value):,.0f} ریال) باقی مانده است."
                        )
                else:
                    st.warning("برای محاسبه پاداش حداقل دو ماه داده لازم است.")
            else:
                st.warning("داده‌ای برای محاسبه پاداش وجود ندارد.")

            st.write('---')

            # --- Filters for deal_source and deal_owner ---
            deal_sources = data['deal_source'].dropna().unique().tolist()
            deal_owners = data['deal_owner'].dropna().unique().tolist()

            selected_deal_source = st.selectbox("انتخاب منبع معامله (deal_source):", options=["همه"] + deal_sources, index=0)
            selected_deal_owner = st.selectbox("انتخاب مالک معامله (deal_owner):", options=["همه"] + deal_owners, index=0)

            # Filter data based on user selection
            filtered_data = data.copy()
            if selected_deal_source != "همه":
                filtered_data = filtered_data[filtered_data['deal_source'] == selected_deal_source]
            if selected_deal_owner != "همه":
                filtered_data = filtered_data[filtered_data['deal_owner'] == selected_deal_owner]

            # --- Calculate total and daily sales for the last week ---
            # Ensure 'deal_done_date' is datetime
            if not pd.api.types.is_datetime64_any_dtype(filtered_data['deal_done_date']):
                filtered_data['deal_done_date'] = pd.to_datetime(filtered_data['deal_done_date'], errors='coerce')

            today = pd.Timestamp.today().normalize()
            week_ago = today - pd.Timedelta(days=6)

            # Filter for last 7 days
            week_data = filtered_data[
                (filtered_data['deal_done_date'] >= week_ago) & (filtered_data['deal_done_date'] <= today)
            ]

            total_sales_week = week_data['deal_value'].sum()

            # Aggregate daily sales sum and count for the week
            daily_agg = (
                week_data.groupby(week_data['deal_done_date'].dt.date)
                .agg(sales_sum=('deal_value', 'sum'), sales_count=('deal_value', 'count'))
                .reindex(
                    [(today - pd.Timedelta(days=i)).date() for i in reversed(range(7))],
                    fill_value=0
                )
            )

            st.subheader("آمار فروش بازه انتخابی")

            # Let user select a date range for statistics using two date_input boxes
            min_date = filtered_data['deal_done_date'].min()
            max_date = filtered_data['deal_done_date'].max()
            if pd.isnull(min_date) or pd.isnull(max_date):
                st.warning("داده‌ای برای انتخاب بازه زمانی وجود ندارد.")
            else:
                min_date_greg = min_date.date()
                max_date_greg = max_date.date()
                # Ensure default_start is not after max_date_greg and not before min_date_greg
                default_start_candidate = max(min_date_greg, (max_date_greg - pd.Timedelta(days=6)))
                default_start = min(default_start_candidate, max_date_greg)
                # Defensive: if min_date_greg > max_date_greg, set both to min_date_greg
                if min_date_greg > max_date_greg:
                    min_date_greg = max_date_greg = default_start = min_date_greg

                # Use session state to store filter values and apply state
                if 'social_filters' not in st.session_state:
                    st.session_state['social_filters'] = {
                        'start_date': default_start,
                        'end_date': max_date_greg,
                        'applied_start_date': default_start,
                        'applied_end_date': max_date_greg,
                        'applied': False
                    }
                # Clamp session state values to valid range
                for key in ['start_date', 'end_date', 'applied_start_date', 'applied_end_date']:
                    val = st.session_state['social_filters'][key]
                    if val < min_date_greg:
                        st.session_state['social_filters'][key] = min_date_greg
                    if val > max_date_greg:
                        st.session_state['social_filters'][key] = max_date_greg

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    start_date = st.date_input(
                        "تاریخ شروع",
                        value=st.session_state['social_filters']['start_date'],
                        min_value=min_date_greg,
                        max_value=max_date_greg,
                        key="start_date_input"
                    )
                with col2:
                    end_date = st.date_input(
                        "تاریخ پایان",
                        value=st.session_state['social_filters']['end_date'],
                        min_value=min_date_greg,
                        max_value=max_date_greg,
                        key="end_date_input"
                    )
                with col3:
                    apply_clicked = st.button("اعمال فیلتر", key="apply_social_filter")

                # Update session state with current selections, clamped to valid range
                st.session_state['social_filters']['start_date'] = min(max(start_date, min_date_greg), max_date_greg)
                st.session_state['social_filters']['end_date'] = min(max(end_date, min_date_greg), max_date_greg)

                # Only update applied filters when button is pressed
                if apply_clicked:
                    st.session_state['social_filters']['applied_start_date'] = st.session_state['social_filters']['start_date']
                    st.session_state['social_filters']['applied_end_date'] = st.session_state['social_filters']['end_date']
                    st.session_state['social_filters']['applied'] = True

                # Use applied filters for calculations
                applied_start_date = st.session_state['social_filters']['applied_start_date']
                applied_end_date = st.session_state['social_filters']['applied_end_date']

                # Ensure start_date <= end_date
                if applied_start_date > applied_end_date:
                    st.warning("تاریخ شروع نباید بعد از تاریخ پایان باشد.")
                    range_data = pd.DataFrame()
                    total_sales_range = 0
                    all_days = pd.date_range(start=applied_start_date, end=applied_end_date, freq='D')
                    daily_agg = pd.DataFrame(
                        0, index=all_days.date, columns=['sales_sum', 'sales_count']
                    )
                else:
                    # Filter for selected date range
                    mask = (
                        (filtered_data['deal_done_date'] >= pd.Timestamp(applied_start_date)) &
                        (filtered_data['deal_done_date'] <= pd.Timestamp(applied_end_date))
                    )
                    range_data = filtered_data[mask]

                    total_sales_range = range_data['deal_value'].sum()

                    # Aggregate daily sales sum and count for the selected range
                    if not range_data.empty:
                        all_days = pd.date_range(start=applied_start_date, end=applied_end_date, freq='D')
                        daily_agg = (
                            range_data.groupby(range_data['deal_done_date'].dt.date)
                            .agg(sales_sum=('deal_value', 'sum'), sales_count=('deal_value', 'count'))
                            .reindex(all_days.date, fill_value=0)
                        )
                    else:
                        all_days = pd.date_range(start=applied_start_date, end=applied_end_date, freq='D')
                        daily_agg = pd.DataFrame(
                            0, index=all_days.date, columns=['sales_sum', 'sales_count']
                        )

                st.metric("مجموع فروش بازه انتخابی (ریال)", f"{total_sales_range:,.0f}")

                # Line chart: daily sales in the selected range
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_agg.index,
                    y=daily_agg['sales_sum'].values,
                    mode='lines+markers',
                    name='ارزش معاملات',
                    hovertemplate=
                        'تاریخ: %{x}<br>' +
                        'ارزش معاملات: %{y:,.0f} ریال<br>' +
                        'تعداد معاملات: %{customdata[0]}<extra></extra>',
                    customdata=daily_agg[['sales_count']].values
                ))
                fig.update_layout(
                    xaxis_title="تاریخ",
                    yaxis_title="ارزش معاملات (ریال)",
                    title="ارزش معاملات در بازه انتخابی",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Settings tab: update reward parameters ---
        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('تاریخ مبدا را مشخص کنید: ')
                config = Config(
                    always_open=False,
                    dark_mode=True,
                    locale="fa",
                    maximum_date=jdatetime.date.today(),
                    color_primary="#ff4b4b",
                    color_primary_light="#ff9494",
                    selection_mode="single",
                    placement="bottom",
                    disabled=True
                )
                selected_date = datepicker_component(config=config)
                res = selected_date.togregorian() if selected_date is not None else None
            with col2:
                grow_percent = st.number_input(
                    label="درصد پاداش در صورت رشد:",
                    key='grow_percent',
                    format="%f",
                    step=1.0,
                    max_value=100.0,
                    value=0.0
                )
                normal_percent = st.number_input(
                    label="درصد در حالت عادی:",
                    key='normal_percent',
                    format="%f",
                    step=1.0,
                    max_value=100.0,
                    value=0.0
                )

            # Save new parameters to sheet
            if st.button('تنظیم مجدد', key='write'):
                df = pd.DataFrame([{
                    "grow_percent": grow_percent,
                    "normal_percent": normal_percent,
                    "Start date": res
                }])
                success = write_df_to_sheet(df, sheet_name='Social team parameters')
                if success:
                    st.info("پارامترها با موفقیت آپدیت شد.")
                else:
                    st.info("خطا در آپدیت پارامترها!!!")

    else:
        pass