import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
import numpy as np
from utils.func import convert_df, convert_df_to_excel
from utils.logger import logger



def ensure_datetime_col(df, col):
    """Ensure a Series is in datetime64 format, return converted col."""
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            return pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            logger.error(f"Failed to convert column '{col}' to datetime: {e}")
            return df[col]
    return df[col]

def calculate_weekly_metrics(data, start_date, end_date):
    """Calculate weekly metrics for given data and date range (fix .dt error)."""
    try:
        deal_created = ensure_datetime_col(data, 'deal_created_date')
        mask = (deal_created.dt.date >= start_date) & (deal_created.dt.date <= end_date)
        count = data[mask].shape[0]
        value = data[mask]['deal_value'].sum()
        avg = value / count if count > 0 else 0
        return count, value, avg
    except Exception as e:
        logger.error(f"Error in calculate_weekly_metrics: {e}")
        return 0, 0, 0

def create_weekly_chart(df, x_col, y_col, title, color_col=None):
    """Create a standardized weekly chart"""
    try:
        fig = px.bar(df, x=x_col, y=y_col, hover_data=['بازه زمانی'], title=title)
        fig.update_layout(
            title_x=0.5,  
            title_font=dict(size=20, family='Tahoma'),
            xaxis_title="بازه زمانی",
            yaxis_title=y_col,
            height=400
        )
        fig.update_layout(title={'x':0.1}) 
        if color_col is not None:
            fig.update_traces(marker_color=['#90EE90' if i == color_col else 'gray' for i in range(len(df))])
        return fig
    except Exception as e:
        logger.error(f"Error creating weekly chart: {e}")
        return go.Figure()

def display_metrics(col, metrics):
    """Display metrics in a standardized format"""
    for label, value, suffix in metrics:
        if value is np.nan:
            value = 0
        try:
            st.metric(label, f"{value:,.0f}{suffix}")
        except Exception as e:
            logger.error(f"Error displaying metric '{label}': {e}")

def normalize_owner(owner: str) -> str:
    """
    Merges different names for the same person into a single, consistent name. 
    Specifically handles day/night shifts for 'محمد آبساران'.
    """
    try:
        if owner in ["محمد آبساران/روز"]:
            return "محمد آبساران"
    except Exception as e:
        logger.error(f"Error normalizing owner '{owner}': {e}")
    return owner

def b2b():
    """B2B team dashboard with optimized metrics and visualizations"""
    try:
        st.title("📊 داشبورد تیم B2B")

        if not all(key in st.session_state for key in ['username', 'role', 'data', 'team', 'auth']):
            st.error("لطفا ابتدا وارد سیستم شوید")
            return

        role = st.session_state.role
        username = st.session_state.username
        name = st.session_state.name
        st.write(f"{name} عزیز خوش آمدی😃")

        # filter data
        data = st.session_state.data.copy()
        filter_data = data[
            data['deal_owner'].isin(['محمد آبساران/روز'])
        ].copy()
        filter_data['deal_created_date'] = ensure_datetime_col(filter_data, 'deal_created_date')
        try:
            filter_data['deal_value'] = pd.to_numeric(filter_data['deal_value'], errors='coerce') / 10
        except Exception as e:
            logger.error(f"Error converting deal_value to numeric: {e}")
            filter_data['deal_value'] = 0

        # Calculate date ranges
        today = datetime.today().date()
        try:
            start_date = jdatetime.date(1404, 2, 28).togregorian()
        except Exception as e:
            logger.error(f"Error converting jdatetime to gregorian: {e}")
            start_date = today

        try:
            filter_data = filter_data[filter_data['deal_created_date'] >= pd.to_datetime(start_date, )]
            filter_data['deal_created_date'] = ensure_datetime_col(filter_data, 'deal_created_date') # Defensive
        except Exception as e:
            logger.error(f"Error filtering/filtering date column: {e}")

        # Defensive for date mask: always use a helper to get date
        def get_date_series(col):
            # Ensures we get a date Series for mask filters
            if not pd.api.types.is_datetime64_any_dtype(col):
                try:
                    col = pd.to_datetime(col, errors='coerce')
                except Exception as e:
                    logger.error(f"Error converting to datetime in get_date_series: {e}")
            return col.dt.date

        weeks_passed = (today - start_date).days // 7
        current_week_start = start_date + timedelta(weeks=weeks_passed)
        week_ranges = [(current_week_start - timedelta(weeks=i), 
                       current_week_start - timedelta(weeks=i-1) - timedelta(days=1)) 
                       for i in range(4, 0, -1)]

        # This week
        try:
            jalali_start = jdatetime.date.fromgregorian(date=current_week_start)
            jalali_end = jdatetime.date.fromgregorian(date=today)
            end_week = jdatetime.date.fromgregorian(date=current_week_start + timedelta(6))
        except Exception as e:
            logger.error(f"Error converting current_week_start/today to jdatetime: {e}")
            jalali_start = jalali_end = end_week = None
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.info(f"شروع هفته: {jalali_start.strftime('%Y/%m/%d')} \n پایان هفته: {end_week.strftime('%Y/%m/%d')}")
            except Exception as e:
                logger.error(f"Error in jalali_start/end_week strftime: {e}")
        with col2:
            try:
                st.info(f"امروز: {jalali_end.strftime('%Y/%m/%d')}")
            except Exception as e:
                logger.error(f"Error in jalali_end strftime: {e}")

        # Calculate team metrics
        try:
            weekly_metrics = [calculate_weekly_metrics(filter_data, start, end) for start, end in week_ranges]
            weekly_counts, weekly_values, weekly_avgs = zip(*weekly_metrics)
        except Exception as e:
            logger.error(f"Error calculating weekly_metrics: {e}")
            weekly_counts, weekly_values, weekly_avgs = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

        # For .dt.date masks, always use get_date_series()
        try:
            date_series = get_date_series(filter_data['deal_created_date'])
        except Exception as e:
            logger.error(f"Error getting date_series: {e}")
            date_series = pd.Series(today for _ in range(len(filter_data)))

        # Calculate current week metrics
        try:
            this_week_mask = (date_series >= current_week_start) & (date_series <= today)
            this_week_count = filter_data[this_week_mask].shape[0]
            this_week_value = filter_data[this_week_mask]['deal_value'].sum()
            this_week_avg = this_week_value / this_week_count if this_week_count > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating this_week metrics: {e}")
            this_week_count, this_week_value, this_week_avg = 0, 0, 0

        max_count_week = weekly_counts.index(max(weekly_counts)) if weekly_counts else 0
        max_value_week = weekly_values.index(max(weekly_values)) if weekly_values else 0
        max_avg_week = weekly_avgs.index(max(weekly_avgs)) if weekly_avgs else 0

        # Display team overview
        st.subheader("📈 آمار کلی تیم")
        col1, col2 = st.columns(2)

        with col1:
            try:
                st_today_count = filter_data[get_date_series(filter_data['deal_created_date']) == today].shape[0]
                display_metrics(col1, [
                    ("تعداد فروش امروز", st_today_count, ""),
                    ("تعداد فروش این هفته", this_week_count, ""),
                    ("بیشترین تعداد فروش هفتگی", max(weekly_counts), f" ({4 - max_count_week} هفته پیش) "),
                ])
                start = week_ranges[weekly_counts.index(max(weekly_counts))][0]
                end = week_ranges[weekly_counts.index(max(weekly_counts))][1]
                st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
            except Exception as e:
                logger.error(f"Error in team overview left column: {e}")

        with col2:
            try:
                today_mask = get_date_series(filter_data['deal_created_date']) == today
                today_value_sum = filter_data[today_mask]['deal_value'].sum()
                display_metrics(col2, [
                    ("مقدار فروش امروز", today_value_sum, " تومان"),
                    ("مقدار فروش این هفته", this_week_value, " تومان"),
                    ("بیشترین مقدار فروش هفتگی", max(weekly_values), f" تومان ({4 - max_value_week}هفته پیش) "),
                ])
                start = week_ranges[weekly_values.index(max(weekly_values))][0]
                end = week_ranges[weekly_values.index(max(weekly_values))][1]
                st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
            except Exception as e:
                logger.error(f"Error in team overview right column: {e}")

        # Team charts
        col1, col2 = st.columns(2)

        with col1:
            try:
                df_counts = pd.DataFrame({
                    'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
                    'تعداد': weekly_counts,
                    'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
                })
                st.plotly_chart(create_weekly_chart(df_counts, 'هفته', 'تعداد', 'تعداد فروش هفتگی تیم', max_count_week))
            except Exception as e:
                logger.error(f"Error plotting counts bar chart: {e}")

        with col2:
            try:
                df_values = pd.DataFrame({
                    'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
                    'مقدار': weekly_values,
                    'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
                })
                st.plotly_chart(create_weekly_chart(df_values, 'هفته', 'مقدار', 'مقدار فروش هفتگی تیم', max_value_week))
            except Exception as e:
                logger.error(f"Error plotting values bar chart: {e}")

        # Team average metrics
        st.subheader("📊 میانگین معاملات تیم")
        col1, col2 = st.columns(2)
        with col1:
            try:
                today_mask = get_date_series(filter_data['deal_created_date']) == today
                today_mean = filter_data[today_mask]['deal_value'].mean()
                display_metrics(col1, [
                    ("میانگین امروز", today_mean, " تومان"),
                    ("میانگین این هفته", this_week_avg, " تومان"),
                    ("بیشترین مقدار فروش هفتگی", max(weekly_values), f" تومان ({4 - max_value_week}هفته پیش) "),
                ])
                start = week_ranges[weekly_values.index(max(weekly_values))][0]
                end = week_ranges[weekly_values.index(max(weekly_values))][1]
                st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
            except Exception as e:
                logger.error(f"Error in team mean left column: {e}")

        with col2:
            try:
                df_avg = pd.DataFrame({
                    'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
                    'میانگین': weekly_avgs,
                    'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
                })
                st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', 'میانگین معامله های تیم', max_avg_week))
            except Exception as e:
                logger.error(f"Error plotting mean bar chart: {e}")

        # Target and reward section
        st.subheader("🎯 تارگت پاداش")

        reward_percentage = 0.05

        # reward
        try:
            target = max(weekly_values) * 0.9
        except Exception as e:
            logger.error(f"Error calculating target: {e}")
            target = 0
        progress_percentage = (this_week_value / target) * 100 if target > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            try:
                st.metric("تارگت هفته", f"{target:,.0f} تومان")
                if this_week_value > target:
                    reward = reward_percentage * (this_week_value - target)
                    st.success(f"🎉 پاداش: {reward:,.0f} تومان")
                else:
                    remaining = target - this_week_value
                    st.warning(f"⏳ باقیمانده: {remaining:,.0f} تومان")
                if progress_percentage < 100:
                    st.info(f"🎯 {100 - progress_percentage:.1f}% تا رسیدن به تارگت ")
            except Exception as e:
                logger.error(f"Error displaying target/reward in left box: {e}")

        with col2:
            try:
                st.subheader("میزان پیشرفت ")
                display_percentage = min(progress_percentage, 100.0)
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    values=[display_percentage, 100 - display_percentage],
                    hole=.8,
                    marker_colors=['#00FF00' if display_percentage >= 100 else '#00FF00', '#E5ECF6'],
                    showlegend=False,
                    textinfo='none',
                    rotation=90,
                    pull=[0.1, 0]
                ))
                fig.update_layout(
                    annotations=[
                        dict(text=f'{display_percentage:.1f}%', x=0.5, y=0.5, font_size=24, font_color='#2F4053', showarrow=False),
                        dict(text='تکمیل شده' if display_percentage >= 100 else 'در حال پیشرفت', x=0.5, y=0.35, font_size=14, font_color='#2E4053', showarrow=False)
                    ],
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, width=True)
            except Exception as e:
                logger.error(f"Error rendering progress pie chart: {e}")

        # Show sales for the last 10 weeks (X axis: start and end day/month of each week)
        st.markdown("---")
        st.subheader("📊 فروش ۱۰ هفته اخیر")

        num_weeks_sales = 10   # Number of weeks to display in the sales chart
        num_weeks_target_reward = 6  # Number of weeks to display target/reward boxes

        # Calculate the last 10 week ranges (ending before the current week)
        all_week_ranges = []
        for i in range(num_weeks_sales, 0, -1):
            week_start = current_week_start - timedelta(weeks=i)
            week_end = week_start + timedelta(days=6)
            all_week_ranges.append((week_start, week_end))

        # Calculate weekly sales for each of the last 10 weeks
        all_weekly_sales = []
        try:
            date_series = get_date_series(filter_data['deal_created_date']) # Defensive
            for start, end in all_week_ranges:
                mask = (date_series >= start) & (date_series <= end)
                value = filter_data[mask]['deal_value'].sum()
                all_weekly_sales.append(value)
        except Exception as e:
            logger.error(f"Error calculating all_weekly_sales: {e}")
            all_weekly_sales = [0]*num_weeks_sales

        # Prepare X axis labels: day/month for start and end of week in Jalali (e.g., 01/03 - 07/03)
        def to_jalali_label(start, end):
            try:
                start_j = jdatetime.date.fromgregorian(date=start)
                end_j = jdatetime.date.fromgregorian(date=end)
                return f"{start_j.day:02d}/{start_j.month:02d} - {end_j.day:02d}/{end_j.month:02d}"
            except Exception as e:
                logger.error(f"Error generating Jalali label from {start} to {end}: {e}")
                return str(start) + "-" + str(end)

        weeks_labels = [
            to_jalali_label(start, end)
            for start, end in all_week_ranges
        ]

        # Plot sales for last 10 weeks
        try:
            fig_sales = go.Figure()
            fig_sales.add_trace(go.Bar(
                x=weeks_labels,
                y=all_weekly_sales,
                name="Sales",
                marker_color="#0984e3",
                text=[f"{v:,.0f}" for v in all_weekly_sales],
                textposition='outside'
            ))
            fig_sales.update_layout(
                title="",
                xaxis_title="هفته (شروع - پایان)",
                yaxis_title="فروش (تومان)",
                font=dict(family="Tahoma", size=16),
                height=450,
                margin=dict(l=20, r=20, t=60, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_sales, width=True)
        except Exception as e:
            logger.error(f"Error plotting last 10 weeks sales chart: {e}")

        # Calculate target and reward for the last 6 weeks (ending before current week)
        st.markdown("---")
        st.subheader("🎯 تارگت & 💰 پاداش (6 هفته اخیر)")

        # For each of the last 6 weeks, calculate target and reward
        target_reward_boxes = []
        for i in range(num_weeks_sales - num_weeks_target_reward, num_weeks_sales):
            # Look back at the 4 weeks before the current week
            start_idx = max(0, i - 4)
            end_idx = i  # up to but not including current week
            prev_weeks = all_weekly_sales[start_idx:end_idx]
            if prev_weeks:
                past_target = max(prev_weeks) * 0.9
            else:
                past_target = 0
            week_value = all_weekly_sales[i]
            if past_target > 0 and week_value > past_target:
                reward = reward_percentage * (week_value - past_target)
            else:
                reward = 0
            week_start, week_end = all_week_ranges[i]
            week_label = to_jalali_label(week_start, week_end)
            target_reward_boxes.append({
                "week": 6 - (i - 4),
                "label": week_label,
                "target": past_target,
                "reward": reward,
                "sales": week_value
            })

        # Display target and reward for each of the last 6 weeks in separate boxes
        cols = st.columns(num_weeks_target_reward)
        for idx, box in enumerate(target_reward_boxes):
            with cols[idx]:
                try:
                    st.markdown(f"<div style='text-align:center; font-size:1.1em; font-weight:bold; margin-bottom:8px;'>{box['label']}</div>", unsafe_allow_html=True)
                    st.markdown(f"{box['week']:,.0f} هفته پیش")
                    st.metric("تارگت", f"{box['target']:,.0f} تومان")
                    st.metric("فروش", f"{box['sales']:,.0f} تومان")
                    if box['reward'] > 0:
                        st.success(f"پاداش: +{box['reward']:,.0f} تومان")
                    else:
                        st.warning("No Reward")
                except Exception as e:
                    logger.error(f"Error displaying target/reward box {idx}: {e}")

        st.markdown("---")
        try:
            filter_data['deal_owner'] = filter_data['deal_owner'].apply(normalize_owner)
        except Exception as e:
            logger.error(f"Error normalizing owners: {e}")

        # Member specific section
        if role in ['member', 'manager']:
            try:
                display_member_metrics(filter_data, username, week_ranges, today, current_week_start, show_name_as_you=True)
            except Exception as e:
                logger.error(f"Error displaying member metrics for {username}: {e}")

        # Manager view of team members with slide navigation
        if role in ['manager', 'admin']:
            try:
                user_list = [user for user in st.secrets['user_lists']['b2b'] 
                            if user != username and st.secrets['roles'][user] != 'admin']
                if user_list:
                    selected_member = st.selectbox(
                        "انتخاب عضو تیم برای مشاهده آمار",
                        user_list,
                        format_func=lambda u: st.secrets['names'][u],
                        key="b2b_member_select"
                    )
                    if selected_member:
                        display_member_metrics(filter_data, selected_member, week_ranges, today, current_week_start)
            except Exception as e:
                logger.error(f"Error in manager/admin member select: {e}")
    except Exception as e:
        logger.error(f"b2b() function failed: {e}")

def display_member_metrics(data, member, week_ranges, today, current_week_start, show_name_as_you=False):
    """Display metrics and charts for a specific team member"""
    try:
        # Defensive ensure datetime column for proper .dt.date access
        member_data = data[data['deal_owner'] == member].copy()
        member_data['deal_created_date'] = ensure_datetime_col(member_data, 'deal_created_date')
        date_series = member_data['deal_created_date'].dt.date
        member_name = st.secrets['names'].get(member, member)

        # Calculate member metrics
        try:
            member_metrics = [calculate_weekly_metrics(member_data, start, end) for start, end in week_ranges]
            member_counts, member_values, member_avgs = zip(*member_metrics)
        except Exception as e:
            logger.error(f"Error calculating member metrics for {member}: {e}")
            member_counts, member_values, member_avgs = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

        # Calculate current week metrics
        try:
            member_this_week_mask = (date_series >= current_week_start) & (date_series <= today)
            member_this_week_count = member_data[member_this_week_mask].shape[0]
            member_this_week_value = member_data[member_this_week_mask]['deal_value'].sum()
            member_this_week_avg = member_this_week_value / member_this_week_count if member_this_week_count > 0 else 0
            member_this_week_data = member_data[member_this_week_mask].reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error calculating member's this week metrics: {e}")
            member_this_week_count, member_this_week_value, member_this_week_avg = 0, 0, 0
            member_this_week_data = pd.DataFrame()

        max_count_week = member_counts.index(max(member_counts)) if member_counts else 0
        max_value_week = member_values.index(max(member_values)) if member_values else 0
        max_avg_week = member_avgs.index(max(member_avgs)) if member_avgs else 0

        if show_name_as_you:
            st.subheader("👤 آمار شما")
        else:
            st.subheader(f"👤 آمار {member_name}")
        col1, col2 = st.columns(2)

        with col1:
            try:
                today_count = member_data[date_series == today].shape[0]
                display_metrics(col1, [
                    ("تعداد فروش امروز", today_count, ""),
                    ("تعداد فروش این هفته", member_this_week_count, ""),
                    ("بیشترین تعداد فروش هفتگی", max(member_counts), f" ({4 - max_count_week} هفته پیش)"),
                ])
                start = week_ranges[member_counts.index(max(member_counts))][0]
                end = week_ranges[member_counts.index(max(member_counts))][1]
                st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
            except Exception as e:
                logger.error(f"Error in member left stats column: {e}")

        with col2:
            try:
                today_sum = member_data[date_series == today]['deal_value'].sum()
                display_metrics(col2, [
                    ("مقدار فروش امروز", today_sum, " تومان"),
                    ("مقدار فروش این هفته", member_this_week_value, " تومان"),
                    ("بیشترین مقدار فروش هفتگی", max(member_values), f" تومان ({4 - max_value_week} هفته پیش)"),
                ])
                start = week_ranges[member_values.index(max(member_values))][0]
                end = week_ranges[member_values.index(max(member_values))][1]
                st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
            except Exception as e:
                logger.error(f"Error in member right stats column: {e}")

        try:
            with st.expander('📋 لیست معاملات شما' if show_name_as_you else f'📋 لیست معاملات {member_name}', expanded=False):
                st.dataframe(member_this_week_data, width=True, hide_index=True)
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        st.download_button(
                            label="دانلود داده‌ها به صورت CSV",
                            data=convert_df(member_this_week_data),
                            file_name=f'deals{member}.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        logger.error(f"Error in member CSV download: {e}")
                with col2:
                    try:
                        st.download_button(
                            label="دانلود داده‌ها به صورت اکسل",
                            data=convert_df_to_excel(member_this_week_data),
                            file_name=f'deals{member}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        )
                    except Exception as e:
                        logger.error(f"Error in member Excel download: {e}")
        except Exception as e:
            logger.error(f"Error in member expander/download: {e}")

        # Member charts
        col1, col2 = st.columns(2)

        with col1:
            try:
                df_member_counts = pd.DataFrame({
                    'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
                    'تعداد': member_counts,
                    'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
                })
                st.plotly_chart(create_weekly_chart(df_member_counts, 'هفته', 'تعداد', 'تعداد فروش هفتگی', max_count_week))
            except Exception as e:
                logger.error(f"Error plotting member counts bar: {e}")

        with col2:
            try:
                df_member_values = pd.DataFrame({
                    'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
                    'مقدار': member_values,
                    'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
                })
                st.plotly_chart(create_weekly_chart(df_member_values, 'هفته', 'مقدار', 'مقدار فروش هفتگی', max_value_week))
            except Exception as e:
                logger.error(f"Error plotting member values bar: {e}")

        # Average deal size
        if show_name_as_you:
            st.subheader("📊 میانگین معاملات شما")
        else:
            st.subheader(f"📊 میانگین معاملات {member_name}")
        col1, col2 = st.columns(2)

        with col1:
            try:
                today_mean = member_data[date_series == today]['deal_value'].mean()
                display_metrics(col1, [
                    ("میانگین امروز", today_mean, " تومان"),
                    ("میانگین این هفته", member_this_week_avg, " تومان"),
                    ("بیشترین میانگین هفتگی", max(member_avgs), f" تومان ({4 - max_avg_week} هفته پیش)"),
                ])
                start = week_ranges[member_avgs.index(max(member_avgs))][0]
                end = week_ranges[member_avgs.index(max(member_avgs))][1]
                st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
            except Exception as e:
                logger.error(f"Error in member mean left column: {e}")

        with col2:
            try:
                df_avg = pd.DataFrame({
                    'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
                    'میانگین': member_avgs,
                    'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
                })
                st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', 'میانگین معامله ها', max_avg_week))
            except Exception as e:
                logger.error(f"Error plotting member mean bar: {e}")
    except Exception as e:
        logger.error(f"display_member_metrics() failed for {member}: {e}")