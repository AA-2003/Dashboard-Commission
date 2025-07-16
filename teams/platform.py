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
    if color_col is not None:
        fig.update_traces(marker_color=['#90EE90' if i == color_col else 'gray' for i in range(len(df))])
    return fig

def display_metrics(col, metrics):
    """Display metrics in a standardized format"""
    for label, value, suffix in metrics:
        if value is np.nan:
            value = 0
        st.metric(label, f"{value:,.0f}{suffix}")

def platform():
    """platform team dashboard with optimized metrics and visualizations"""
    st.title("📊 داشبورد تیم platform")
    
    if not all(key in st.session_state for key in ['username', 'role', 'data', 'team', 'auth']):
        st.error("لطفا ابتدا وارد سیستم شوید")
        return

    # Initialize data and variables
    role = st.session_state.role
    username = st.session_state.username
    name = st.session_state.name
    st.write(f"{name} عزیز خوش آمدی😃")
    
    # Process data
    filter_data = st.session_state.data[st.session_state.data['team'] == 'platform'].copy()
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
    st.subheader("📊 نمودارهای تیم")
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
            ("میانگین معامله امروز", filter_data[filter_data['deal_done_date'].dt.date == today]['deal_value'].mean(), " تومان"),
            ("بیشترین میانگین هفتگی", max(weekly_avgs), " تومان"),
            ("میانگین این هفته", this_week_avg, " تومان")
        ])

    with col2:
        df_avg = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'میانگین': weekly_avgs,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', 'میانگین اندازه معامله هفتگی تیم', max_avg_week))

    st.subheader("📊 فروش به تفکیک پلتفرم")
    platform_sales = filter_data[filter_data['platform']!=''].groupby('platform')['deal_value'].sum().sort_values(ascending=False).reset_index()
    platform_sales.columns = ['پلتفرم', 'مقدار فروش']
    
    fig = px.bar(
        platform_sales,
        x='پلتفرم',
        y='مقدار فروش',
        title='مقدار فروش به تفکیک پلتفرم',
        color='پلتفرم',
        text_auto=True,
        category_orders={'پلتفرم': platform_sales['پلتفرم'].tolist()}  # Force order based on sorted data
    )
    
    fig.update_layout(
        xaxis_title='پلتفرم',
        yaxis_title='مقدار فروش',
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    # Target and reward section
    st.subheader("🎯 تارگت پاداش")
    target = max(weekly_values) * 0.9
    reward_percentage = 0.05
    # this_week_value = target + 332000000

    progress_percentage = (this_week_value / target) * 100    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("تارگت هفته", f"{target:,.0f} تومان")
        if this_week_value > target:
            reward = reward_percentage * (this_week_value - target)
            st.success(f"🎉 پاداش: {reward:,.0f} تومان")
            st.balloons()
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

    # Member specific section
    if role in ['member', 'manager']:
        display_member_metrics(filter_data, username, week_ranges, today, current_week_start)

    # Manager view of team members
    if role in ['manager', 'admin']:
        user_list = [user for user in st.secrets['user_lists']['platform'] 
                    if user != username and st.secrets['roles'][user] != 'admin']
        
        for member in user_list:
            display_member_metrics(filter_data, member, week_ranges, today, current_week_start)

def display_member_metrics(data, member, week_ranges, today, current_week_start):
    """Display metrics and charts for a specific team member"""
    member_data = data[data['deal_owner'] == member]
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
        st.plotly_chart(create_weekly_chart(df_member_counts, 'هفته', 'تعداد', f'تعداد فروش هفتگی {member}', max_count_week))

    with col2:
        df_member_values = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'مقدار': member_values,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_member_values, 'هفته', 'مقدار', f'مقدار فروش هفتگی {member}', max_value_week))

    # Average deal size
    st.subheader(f"📊 میانگین معاملات {member}")
    col1, col2 = st.columns(2)
    
    with col1:
        display_metrics(col1, [
            ("میانگین معامله امروز", member_data[member_data['deal_done_date'].dt.date == today]['deal_value'].mean(), " تومان"),
            ("بیشترین میانگین هفتگی", max(member_avgs), " تومان"),
            ("میانگین این هفته", member_this_week_avg, " تومان")
        ])

    with col2:
        df_avg = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'میانگین': member_avgs,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', f'میانگین اندازه معامله هفتگی {member}', max_avg_week))