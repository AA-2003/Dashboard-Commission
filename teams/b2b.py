import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
import numpy as np

def calculate_weekly_metrics(data, start_date, end_date):
    """Calculate weekly metrics for given data and date range"""
    mask = (data['deal_done_date'].dt.date >= start_date) & (data['deal_done_date'].dt.date <= end_date)
    count = data[mask].shape[0]
    value = data[mask]['deal_value'].sum()
    avg = value / count if count > 0 else 0
    return count, value, avg

def create_weekly_chart(df, x_col, y_col, title, color_col=None):
    """Create a standardized weekly chart"""
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

def display_metrics(col, metrics):
    """Display metrics in a standardized format"""
    for label, value, suffix in metrics:
        if value is np.nan:
            value = 0
        st.metric(label, f"{value:,.0f}{suffix}")

def b2b():
    """B2B team dashboard with optimized metrics and visualizations"""
    st.title("📊 داشبورد تیم B2B")
    
    if not all(key in st.session_state for key in ['username', 'role', 'data', 'team', 'auth']):
        st.error("لطفا ابتدا وارد سیستم شوید")
        return

    # Initialize data and variables
    role = st.session_state.role
    username = st.session_state.username
    name = st.session_state.name
    st.write(f"{name} عزیز خوش آمدی😃")
    
    # Process data
    filter_data = st.session_state.data[st.session_state.data['team'] == 'b2b'].copy()
    filter_data['deal_done_date'] = pd.to_datetime(filter_data['deal_done_date'])
    filter_data['deal_value'] = pd.to_numeric(filter_data['deal_value'], errors='coerce') / 10

    # Calculate date ranges
    today = datetime.today().date()
    start_date = jdatetime.date(1404, 2, 28).togregorian()

    # ask about this
    filter_data = filter_data[filter_data['deal_done_date'] >= pd.to_datetime(start_date, )]

    weeks_passed = (today - start_date).days // 7
    current_week_start = start_date + timedelta(weeks=weeks_passed)
    week_ranges = [(current_week_start - timedelta(weeks=i), 
                   current_week_start - timedelta(weeks=i-1) - timedelta(days=1)) 
                   for i in range(4, 0, -1)]
    
    # This week
    jalali_start = jdatetime.date.fromgregorian(date=current_week_start)
    jalali_end = jdatetime.date.fromgregorian(date=today)
    end_week = jdatetime.date.fromgregorian(date=current_week_start + timedelta(6))
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"شروع هفته: {jalali_start.strftime('%Y/%m/%d')} \n پایان هفته: {end_week.strftime('%Y/%m/%d')}")
    with col2:
        st.info(f"امروز: {jalali_end.strftime('%Y/%m/%d')}")

    # Calculate team metrics
    weekly_metrics = [calculate_weekly_metrics(filter_data, start, end) for start, end in week_ranges]
    weekly_counts, weekly_values, weekly_avgs = zip(*weekly_metrics)
    
    # Calculate current week metrics
    this_week_mask = (filter_data['deal_done_date'].dt.date >= current_week_start) & \
                    (filter_data['deal_done_date'].dt.date <= today)
    this_week_count = filter_data[this_week_mask].shape[0]
    this_week_value = filter_data[this_week_mask]['deal_value'].sum()
    this_week_avg = this_week_value / this_week_count if this_week_count > 0 else 0
    
    max_count_week = weekly_counts.index(max(weekly_counts))
    max_value_week = weekly_values.index(max(weekly_values))
    max_avg_week = weekly_avgs.index(max(weekly_avgs))

    # Display team overview
    st.subheader("📈 آمار کلی تیم")
    col1, col2 = st.columns(2)

    with col1:
        display_metrics(col1, [
            ("تعداد فروش امروز", filter_data[filter_data['deal_done_date'].dt.date == today].shape[0], ""),
            ("تعداد فروش این هفته", this_week_count, ""),
            ("بیشترین تعداد فروش هفتگی", max(weekly_counts), f" ({4-max_count_week} هفته پیش) "),
        ])
        start = week_ranges[weekly_counts.index(max(weekly_counts))][0]
        end = week_ranges[weekly_counts.index(max(weekly_counts))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
        
    with col2:
        display_metrics(col2, [
            ("مقدار فروش امروز", filter_data[filter_data['deal_done_date'].dt.date == today]['deal_value'].sum(), " تومان"),
            ("مقدار فروش این هفته", this_week_value, " تومان"),
            ("بیشترین مقدار فروش هفتگی", max(weekly_values), f" تومان ({4-max_value_week}هفته پیش) "),
        ])
        start = week_ranges[weekly_values.index(max(weekly_values))][0]
        end = week_ranges[weekly_values.index(max(weekly_values))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
        

    # Team charts
    col1, col2 = st.columns(2)
    
    with col1:
        df_counts = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'تعداد': weekly_counts,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_counts, 'هفته', 'تعداد', 'تعداد فروش هفتگی تیم', max_count_week))

    with col2:
        df_values = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'مقدار': weekly_values,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_values, 'هفته', 'مقدار', 'مقدار فروش هفتگی تیم', max_value_week))

    # Team average metrics
    st.subheader("📊 میانگین معاملات تیم")
    col1, col2 = st.columns(2)
    
    with col1:
        display_metrics(col1, [
            ("میانگین امروز", filter_data[filter_data['deal_done_date'].dt.date == today]['deal_value'].mean(), " تومان"),
            ("میانگین این هفته", this_week_avg, " تومان"),
              ("بیشترین مقدار فروش هفتگی", max(weekly_values), f" تومان ({4-max_value_week}هفته پیش) "),
        ])
        start = week_ranges[weekly_values.index(max(weekly_values))][0]
        end = week_ranges[weekly_values.index(max(weekly_values))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
        

    with col2:
        df_avg = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'میانگین': weekly_avgs,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', 'میانگین معامله های تیم', max_avg_week))

    
    # Target and reward section
    st.subheader("🎯 تارگت پاداش")

    reward_percentage = 0.05

    # reward
    target = max(weekly_values) * 0.9
    progress_percentage = (this_week_value / target) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("تارگت هفته", f"{target:,.0f} تومان")
        if this_week_value > target:
            reward = reward_percentage * (this_week_value - target)
            st.success(f"🎉 پاداش: {reward:,.0f} تومان")

        else:
            remaining = target - this_week_value
            st.warning(f"⏳ باقیمانده: {remaining:,.0f} تومان")

        if progress_percentage < 100:
            st.info(f"🎯 {100 - progress_percentage:.1f}% تا رسیدن به تارگت ")

    with col2:
        st.subheader("میزان پیشرفت ")
        display_percentage = min(progress_percentage, 100.0)
        fig = go.Figure()
        fig.add_trace(go.Pie(
            values=[display_percentage, 100-display_percentage],
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
        st.plotly_chart(fig, use_container_width=True)
    
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
    for start, end in all_week_ranges:
        mask = (filter_data['deal_done_date'].dt.date >= start) & (filter_data['deal_done_date'].dt.date <= end)
        value = filter_data[mask]['deal_value'].sum()
        all_weekly_sales.append(value)

    # Prepare X axis labels: day/month for start and end of week in Jalali (e.g., 01/03 - 07/03)
    def to_jalali_label(start, end):
        start_j = jdatetime.date.fromgregorian(date=start)
        end_j = jdatetime.date.fromgregorian(date=end)
        return f"{start_j.day:02d}/{start_j.month:02d} - {end_j.day:02d}/{end_j.month:02d}"

    weeks_labels = [
        to_jalali_label(start, end)
        for start, end in all_week_ranges
    ]

    # Plot sales for last 10 weeks
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
    st.plotly_chart(fig_sales, use_container_width=True)

    # Calculate target and reward for the last 6 weeks (ending before current week)
    st.markdown("---")
    st.subheader("🎯 تارگت & 💰 پاداش (6 هفته اخیر)")

    # For each of the last 6 weeks, calculate target and reward
    target_reward_boxes = []
    for i in range(num_weeks_sales - num_weeks_target_reward, num_weeks_sales):
        # Look back at the 4 weeks before the current week
        start_idx = max(0, i-4)
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
            "week": 6-(i-4),
            "label": week_label,
            "target": past_target,
            "reward": reward,
            "sales": week_value
        })

    # Display target and reward for each of the last 6 weeks in separate boxes
    cols = st.columns(num_weeks_target_reward)
    for idx, box in enumerate(target_reward_boxes):
        with cols[idx]:
            st.markdown(f"<div style='text-align:center; font-size:1.1em; font-weight:bold; margin-bottom:8px;'>{box['label']}</div>", unsafe_allow_html=True)
            st.markdown(f"{box['week']:,.0f} هفته پیش")
            st.metric("تارگت", f"{box['target']:,.0f} تومان")
            st.metric("فروش", f"{box['sales']:,.0f} تومان")
            if box['reward'] > 0:
                st.success(f"پاداش: +{box['reward']:,.0f} تومان")
            else:
                st.warning("No Reward")

    st.markdown("---")

    # Member specific section
    if role in ['member', 'manager']:
        display_member_metrics(filter_data, username, week_ranges, today, current_week_start, show_name_as_you=True)

    # Manager view of team members with slide navigation
    if role in ['manager', 'admin']:
        user_list = [user for user in st.secrets['user_lists']['b2b'] 
                    if user != username and st.secrets['roles'][user] != 'admin']
        if user_list:
            if 'member_slide_idx' not in st.session_state:
                st.session_state.member_slide_idx = 0

            col_left, col_center, col_right = st.columns([1, 7, 1])
            with col_left:
                if st.button("➡️ نفر قبلی", key="slide_left"):
                    st.session_state.member_slide_idx = (st.session_state.member_slide_idx - 1) % len(user_list)
            with col_right:
                if st.button("نفر بعدی ⬅️", key="slide_right"):
                    st.session_state.member_slide_idx = (st.session_state.member_slide_idx + 1) % len(user_list)
            with col_center:
                member = user_list[st.session_state.member_slide_idx]
                display_member_metrics(filter_data, member, week_ranges, today, current_week_start)

def display_member_metrics(data, member, week_ranges, today, current_week_start, show_name_as_you=False):
    """Display metrics and charts for a specific team member"""
    member_data = data[data['deal_owner'].str.contains(member)]
    member = st.secrets['names'][member]
    # Calculate member metrics
    member_metrics = [calculate_weekly_metrics(member_data, start, end) for start, end in week_ranges]
    member_counts, member_values, member_avgs = zip(*member_metrics)
    
    # Calculate current week metrics
    member_this_week_mask = (member_data['deal_done_date'].dt.date >= current_week_start) & \
                (member_data['deal_done_date'].dt.date <= today)
    member_this_week_count = member_data[member_this_week_mask].shape[0]
    member_this_week_value = member_data[member_this_week_mask]['deal_value'].sum()
    member_this_week_avg = member_this_week_value / member_this_week_count if member_this_week_count > 0 else 0
    
    max_count_week = member_counts.index(max(member_counts)) if member_counts else 0
    max_value_week = member_values.index(max(member_values)) if member_values else 0
    max_avg_week = member_avgs.index(max(member_avgs)) if member_avgs else 0

    if show_name_as_you:
        st.subheader(f"👤 آمار شما")
    else:
        st.subheader(f"👤 آمار {member}")
    col1, col2 = st.columns(2)
    
    with col1:
        display_metrics(col1, [
            ("تعداد فروش امروز", member_data[member_data['deal_done_date'].dt.date == today].shape[0], ""),
            ("تعداد فروش این هفته", member_this_week_count, ""),
            ("بیشترین تعداد فروش هفتگی", max(member_counts), f" ({4-max_count_week} هفته پیش)"),
        ])
        start = week_ranges[member_counts.index(max(member_counts))][0]
        end = week_ranges[member_counts.index(max(member_counts))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
        

    with col2:
        display_metrics(col2, [
            ("مقدار فروش امروز", member_data[member_data['deal_done_date'].dt.date == today]['deal_value'].sum(), " تومان"),
            ("مقدار فروش این هفته", member_this_week_value, " تومان"),
            ("بیشترین مقدار فروش هفتگی", max(member_values), f" تومان ({4-max_value_week} هفته پیش)"),
        ])
        start = week_ranges[member_values.index(max(member_values))][0]
        end = week_ranges[member_values.index(max(member_values))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
        

    # Member charts
    col1, col2 = st.columns(2)
    
    with col1:
        df_member_counts = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'تعداد': member_counts,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_member_counts, 'هفته', 'تعداد', f'تعداد فروش هفتگی', max_count_week))

    with col2:
        df_member_values = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'مقدار': member_values,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_member_values, 'هفته', 'مقدار', f'مقدار فروش هفتگی', max_value_week))

    # Average deal size
    if show_name_as_you:
        st.subheader(f"📊 میانگین معاملات شما")
    else:
        st.subheader(f"📊 میانگین معاملات {member}")
    col1, col2 = st.columns(2)
    
    with col1:
        display_metrics(col1, [
            ("میانگین امروز", member_data[member_data['deal_done_date'].dt.date == today]['deal_value'].mean(), " تومان"),
            ("میانگین این هفته", member_this_week_avg, " تومان"),
            ("بیشترین میانگین هفتگی", max(member_avgs), f" تومان ({4-max_avg_week} هفته پیش)"),
        ])
        start = week_ranges[member_avgs.index(max(member_avgs))][0]
        end = week_ranges[member_avgs.index(max(member_avgs))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')

    with col2:
        df_avg = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'میانگین': member_avgs,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', f'میانگین معامله ها', max_avg_week))