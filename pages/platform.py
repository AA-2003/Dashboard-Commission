import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
import numpy as np
from utils.funcs import download_buttons, handel_errors
from utils.custom_css import apply_custom_css
from utils.sidebar import render_sidebar

# Prepare X axis labels: day/month for start and end of week in Jalali (e.g., 01/03 - 07/03)
def to_jalali_label(start, end):
    start_j = jdatetime.date.fromgregorian(date=start)
    end_j = jdatetime.date.fromgregorian(date=end)
    return f"{start_j.day:02d}/{start_j.month:02d} - {end_j.day:02d}/{end_j.month:02d}"
    
@st.cache_data(ttl=300, show_spinner=False)
def get_platform_sales_df(filtered_data, mask, label):
    """Helper for aggregating sales by platform filtered by mask and period label."""
    df = filtered_data[mask & (filtered_data['platform'] != '')].groupby('platform')['deal_value'].sum().sort_values(ascending=False).reset_index().copy()
    df.columns = ['پلتفرم', 'مقدار فروش']
    df['بازه'] = label
    return df

def ensure_datetime_col(df, col):
    """
    Ensure a pandas Series is in datetime64 format. 
    Returns the converted column or original if already correct.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            return pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            handel_errors(e, f"Error converting {col} to datetime", show_error=False)
            return df[col]
    return df[col]

def calculate_weekly_metrics(data, start_date, end_date):
    """
    Calculate weekly metrics for given data and date range.
    Ensures .dt accessor is used only on datetimelike columns.
    """
    deal_created_col = ensure_datetime_col(data, 'deal_created_time')
    # Defensive: Get date values safely
    deal_date_values = deal_created_col.dt.date if pd.api.types.is_datetime64_any_dtype(deal_created_col) else deal_created_col
    mask = (deal_date_values >= start_date) & (deal_date_values <= end_date)
    count = data[mask].shape[0]
    value = data[mask]['deal_value'].sum()
    avg = value / count if count > 0 else 0
    return count, value, avg

def create_weekly_chart(df, x_col, y_col, title, color_col=None):
    """Create a standardized weekly chart."""
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
    """Display metrics in a standardized format."""
    for label, value, suffix in metrics:
        if value is np.nan:
            value = 0
        st.metric(label, f"{value:,.0f}{suffix}")

def main():
    """
    platform team dashboard with optimized metrics and visualizations.
    Ensures all .dt usage is behind type checks to avoid AttributeError.
    """
    apply_custom_css()
    render_sidebar()

    st.title("📊 داشبورد تیم Platform")
    
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("لطفا ابتدا وارد سیستم شوید")
        return

    role = st.session_state.userdata.get('role', '')
    teams = st.session_state.userdata.get('team', '')
    teams_list = [team.strip() for team in teams.split('|')]
    name = st.session_state.userdata.get('name', '')

    if 'platform' not in teams_list:
        st.error("شما به این بخش دسترسی ندارید")
        return

    data = st.session_state.deals_data.copy()
    
    # Filter data for Platform team
    filtered_data = data[
        # (data['deal_owner'].isin(team_users['username_in_didar'].values)) &
        (data['deal_source']=='پلت‌فرم')&
        (data['deal_type'].isin(['تمدید', 'فروش جدید'])) &
        (data['deal_status'] == 'Won')
    ].copy()
    
    filtered_data = filtered_data.copy()

    # Always ensure types safely before .dt
    filtered_data.loc[:, 'deal_created_time'] = pd.to_datetime(filtered_data['deal_created_time'], errors='coerce')
    filtered_data.loc[:, 'deal_value'] = pd.to_numeric(filtered_data['deal_value'], errors='coerce') / 10

    # Calculate date ranges
    today = datetime.today().date()
    start_date = jdatetime.date(1404, 2, 31).togregorian()

    filtered_data = filtered_data[filtered_data['deal_created_time'] >= pd.to_datetime(start_date, )]

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
    weekly_metrics = [calculate_weekly_metrics(filtered_data, start, end) for start, end in week_ranges]
    weekly_counts, weekly_values, weekly_avgs = zip(*weekly_metrics)
    
    # Defensive extraction of .dt.date
    deal_created_col = ensure_datetime_col(filtered_data, 'deal_created_time')
    deal_date_values = deal_created_col.dt.date if pd.api.types.is_datetime64_any_dtype(deal_created_col) else deal_created_col

    # Calculate current week metrics safely
    this_week_mask = (deal_date_values >= current_week_start) & (deal_date_values <= today)
    this_week_count = filtered_data[this_week_mask].shape[0]
    this_week_value = filtered_data[this_week_mask]['deal_value'].sum()
    this_week_avg = this_week_value / this_week_count if this_week_count > 0 else 0

    max_count_week = weekly_counts.index(max(weekly_counts))
    max_value_week = weekly_values.index(max(weekly_values))
    max_avg_week = weekly_avgs.index(max(weekly_avgs))

    # Display team overview
    st.subheader("📈 آمار کلی تیم")
    col1, col2 = st.columns(2)

    with col1:
        today_mask = (deal_date_values == today)
        display_metrics(col1, [
            ("تعداد فروش امروز", filtered_data[today_mask].shape[0], ""),
            ("تعداد فروش این هفته", this_week_count, ""),
            ("بیشترین تعداد فروش هفتگی", max(weekly_counts), f" ({4-max_count_week} هفته پیش) "),
        ])
        start = week_ranges[weekly_counts.index(max(weekly_counts))][0]
        end = week_ranges[weekly_counts.index(max(weekly_counts))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
        
    with col2:
        display_metrics(col2, [
            ("مقدار فروش امروز", filtered_data[today_mask]['deal_value'].sum(), " تومان"),
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
            ("میانگین امروز", filtered_data[today_mask]['deal_value'].mean(), " تومان"),
            ("میانگین این هفته", this_week_avg, " تومان"),
            ("بیشترین میانگین هفتگی", max(weekly_avgs), f" تومان ({4-max_avg_week} هفته پیش)"),
        ])
        start = week_ranges[weekly_avgs.index(max(weekly_avgs))][0]
        end = week_ranges[weekly_avgs.index(max(weekly_avgs))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
    
    with col2:
        df_avg = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'میانگین': weekly_avgs,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', 'میانگین معامله های تیم', max_avg_week))

    st.markdown("---")
    # Platform - Split by Day, Week, Month - Drill Down
    st.subheader("📊 فروش به تفکیک پلتفرم")

    week_end = current_week_start + pd.Timedelta(days=6)
    last_month_start = today - pd.Timedelta(days=29)

    today_mask = (deal_date_values == today)
    week_mask = (deal_date_values >= current_week_start) & (deal_date_values <= week_end)
    month_mask = (deal_date_values >= last_month_start) & (deal_date_values <= today)

    df_day = get_platform_sales_df(filtered_data, today_mask, 'امروز')
    df_week = get_platform_sales_df(filtered_data, week_mask, 'این هفته')
    df_month = get_platform_sales_df(filtered_data, month_mask, 'این ماه')

    df_all = pd.concat([df_day, df_week, df_month], ignore_index=True)

    # Drill down UI
    periods = ['امروز', 'این هفته', 'این ماه']

    selected_period = st.radio("بازه زمانی را انتخاب کنید:", periods, horizontal=True)

    period_df = df_all[df_all['بازه'] == selected_period]

    if not period_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=period_df['پلتفرم'],
            y=period_df['مقدار فروش'],
            marker_color='#0984e3',
            text=[f"{v:,.0f}" if v > 0 else "" for v in period_df['مقدار فروش']],
            textposition='outside'
        ))
        fig.update_layout(
            title='',
            xaxis_title='پلتفرم',
            yaxis_title='مقدار فروش (تومان)',
            bargap=0.35,
            margin=dict(l=20, r=20, t=60, b=20),
            height=450,
            font=dict(family="Tahoma", size=10),
        )
        st.plotly_chart(fig, config={'responsive': True})
    else:
        st.info(f"داده‌ای برای نمایش در بازه «{selected_period}» وجود ندارد.")
        
    st.markdown("---")
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
        st.plotly_chart(fig, config={'responsive': True})

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
        mask = (deal_date_values >= start) & (deal_date_values <= end)
        value = filtered_data[mask]['deal_value'].sum()
        all_weekly_sales.append(value)

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
    # set max y 1.1 times the max sales for better visibility
    max_sales = max(all_weekly_sales) if all_weekly_sales else 0
    fig_sales.update_layout(
        title="",
        yaxis_range=[0, max_sales * 1.1],
        xaxis_title="هفته (شروع - پایان)",
        yaxis_title="فروش (تومان)",
        font=dict(family="Tahoma", size=16),
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    st.plotly_chart(fig_sales, config={'responsive': True})

    # Calculate target and reward for the last 6 weeks (ending before current week)
    st.markdown("---")
    st.subheader(f"🎯 تارگت & 💰 پاداش ({num_weeks_target_reward} هفته اخیر)")

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
        display_member_metrics(filtered_data, name, week_ranges, today, current_week_start, show_name_as_you=True)

    # Manager view of team members with slide navigation
    if role in ['manager', 'admin']:
        # team_members = team_users[
        #     (team_users['name'] != name) &
        #     (team_users['username_in_didar'].isin(filtered_data['deal_owner'].unique()))
        # ]['username_in_didar'].unique()

        filtered_data['deal_value'] = filtered_data['deal_value'].astype(float)
        team_members = filtered_data[
            filtered_data['deal_created_time'] > pd.to_datetime(today - timedelta(days=7))
        ].groupby('deal_owner')['deal_value'].sum().sort_values(ascending=False).index
        if len(team_members) > 0:
            tabs = st.tabs(list(team_members))
            for i, tab in enumerate(tabs):
                with tab:
                    display_member_metrics(filtered_data, team_members[i], week_ranges, today, current_week_start, False)


def display_member_metrics(data, member, week_ranges, today, current_week_start, show_name_as_you=False):
    """
    Display metrics and charts for a specific team member.
    Always secures .dt access with checks to prevent AttributeError.
    """
    member_data = data[data['deal_owner'] == member]
    # Prepare member deal dates
    deal_created_col = ensure_datetime_col(member_data, 'deal_created_time')
    deal_date_values = deal_created_col.dt.date if pd.api.types.is_datetime64_any_dtype(deal_created_col) else deal_created_col

    # Calculate member metrics
    member_metrics = [calculate_weekly_metrics(member_data, start, end) for start, end in week_ranges]
    member_counts, member_values, member_avgs = zip(*member_metrics)
    
    # Calculate current week metrics safely
    member_this_week_mask = (deal_date_values >= current_week_start) & (deal_date_values <= today)
    member_this_week_count = member_data[member_this_week_mask].shape[0]
    member_this_week_value = member_data[member_this_week_mask]['deal_value'].sum()
    member_this_week_avg = member_this_week_value / member_this_week_count if member_this_week_count > 0 else 0
    member_this_week_data = member_data[member_this_week_mask].reset_index(drop=True)

    max_count_week = member_counts.index(max(member_counts)) if member_counts else 0
    max_value_week = member_values.index(max(member_values)) if member_values else 0
    max_avg_week = member_avgs.index(max(member_avgs)) if member_avgs else 0

    if show_name_as_you:
        st.subheader("👤 آمار شما")
    else:
        st.subheader(f"👤 آمار {member}")
    col1, col2 = st.columns(2)
    
    with col1:
        today_mask = (deal_date_values == today)
        display_metrics(col1, [
            ("تعداد فروش امروز", member_data[today_mask].shape[0], ""),
            ("تعداد فروش این هفته", member_this_week_count, ""),
            ("بیشترین تعداد فروش هفتگی", max(member_counts), f" ({4-max_count_week} هفته پیش)"),
        ])
        start = week_ranges[member_counts.index(max(member_counts))][0]
        end = week_ranges[member_counts.index(max(member_counts))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
        
    with col2:
        display_metrics(col2, [
            ("مقدار فروش امروز", member_data[today_mask]['deal_value'].sum(), " تومان"),
            ("مقدار فروش این هفته", member_this_week_value, " تومان"),
            ("بیشترین مقدار فروش هفتگی", max(member_values), f" تومان ({4-max_value_week} هفته پیش)"),
        ])
        start = week_ranges[member_values.index(max(member_values))][0]
        end = week_ranges[member_values.index(max(member_values))][1]
        st.write(f'تاریخ: {jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}')
        
    with st.expander('📋 لیست معاملات شما' if show_name_as_you else f'📋 لیست معاملات {member}', expanded=False):
        st.dataframe(member_this_week_data, hide_index=True)
        download_buttons(member_this_week_data, f"{member}_deals_this_week")

    # Member charts
    col1, col2 = st.columns(2)
    
    with col1:
        df_member_counts = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'تعداد': member_counts,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_member_counts, 'هفته', 'تعداد', 'تعداد فروش هفتگی', max_count_week))

    with col2:
        df_member_values = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} - {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges],
            'مقدار': member_values,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=end).strftime("%Y/%m/%d")}' for start, end in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_member_values, 'هفته', 'مقدار', 'مقدار فروش هفتگی', max_value_week))

    # Average deal size
    if show_name_as_you:
        st.subheader("📊 میانگین معاملات شما")
    else:
        st.subheader(f"📊 میانگین معاملات {member}")
    col1, col2 = st.columns(2)
    
    with col1:
        display_metrics(col1, [
            ("میانگین امروز", member_data[today_mask]['deal_value'].mean(), " تومان"),
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
        st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', 'میانگین معامله ها', max_avg_week))

if __name__ == "__main__":
    main()