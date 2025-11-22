import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
from utils.funcs import download_buttons, handel_errors
from utils.custom_css import apply_custom_css
from utils.sidebar import render_sidebar


# Prepare X axis labels: day/month for start and end of week in Jalali (e.g., 01/03 - 07/03)
def to_jalali_label(start, end):
    start_j = jdatetime.date.fromgregorian(date=start)
    end_j = jdatetime.date.fromgregorian(date=end)
    return f"{start_j.day:02d}/{start_j.month:02d} - {end_j.day:02d}/{end_j.month:02d}"


def ensure_datetime_col(df, col):
    """Ensure a column is in datetime64 format."""
    if col not in df.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            return pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            handel_errors(e, f"Error converting column '{col}' to datetime", show_error=False, raise_exception=False)
            return df[col]
    return df[col]

def calculate_weekly_metrics(data, start_date, end_date):
    """Calculate weekly metrics for given data and date range."""
    try:
        deal_created = ensure_datetime_col(data, 'deal_created_time')
        if deal_created is None:
            return 0, 0, 0
        mask = (deal_created.dt.date >= start_date) & (deal_created.dt.date <= end_date)
        count = data[mask].shape[0]
        value = pd.to_numeric(data[mask]['deal_value'], errors='coerce').sum() / 10
        avg = value / count if count > 0 else 0
        return count, value, avg
    except Exception as e:
        handel_errors(e, "Failed to calculate metrics", show_error=False, raise_exception=False)
        return 0, 0, 0

def create_weekly_chart(df, x_col, y_col, title, highlight_idx=None):
    """Create a standardized weekly chart."""
    try:
        fig = px.bar(df, x=x_col, y=y_col, hover_data=['بازه زمانی'], title=title)
        fig.update_layout(
            title_x=0.1,
            title_font=dict(size=20, family='Tahoma'),
            xaxis_title="بازه زمانی",
            yaxis_title=y_col,
            height=400
        )
        if highlight_idx is not None:
            colors = ['#90EE90' if i == highlight_idx else 'gray' for i in range(len(df))]
            fig.update_traces(marker_color=colors)
        return fig
    except Exception as e:
        handel_errors(e, "Error creating chart", show_error=False, raise_exception=False)
        return go.Figure()

def display_metrics(col, metrics):
    """Display metrics in a standardized format."""
    for label, value, suffix in metrics:
        if pd.isna(value):
            value = 0
        try:
            col.metric(label, f"{value:,.0f}{suffix}")
        except Exception as e:
            handel_errors(e, f"Error displaying metric '{label}'", show_error=False, raise_exception=False)

def normalize_owner(owner: str) -> str:
    """Normalize owner names (e.g., merge day/night shifts)."""
    if pd.isna(owner):
        return owner
    if owner in ["محمد آبساران/روز"]:
        return "محمد آبساران"
    return owner

def main():
    """B2B team dashboard."""
    apply_custom_css()
    render_sidebar()

    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("لطفا ابتدا وارد سیستم شوید")
        return

    st.title("📊 داشبورد تیم B2B")

    role = st.session_state.userdata.get('role', '')
    teams = st.session_state.userdata.get('team', '')
    teams_list = [team.strip() for team in teams.split('|')]
    name = st.session_state.userdata.get('name', '')

    if 'b2b' not in teams_list:
        st.error("شما به این بخش دسترسی ندارید")
        return

    # team_users = st.session_state.users[
    #     (st.session_state.users['team'].apply(lambda x: 'b2b' in [team.strip() for team in x.split('|')]))&
    #     (st.session_state.users['role'] != 'admin')
    # ]

    data = st.session_state.deals_data.copy()

    # Filter data for B2B team
    filtered_data = data[
        # (data['deal_owner'].isin(team_users['username_in_didar'].values)) &
        (data['deal_source'].isin(['مهمان واسطه', 'فرودگاه'])) &
        (data['deal_type'].isin(['تمدید', 'فروش جدید'])) &
        (data['deal_status'] == 'Won')
    ].copy()

    # Normalize owner names
    filtered_data['deal_owner'] = filtered_data['deal_owner'].apply(normalize_owner)

    # Ensure datetime column
    filtered_data['deal_created_time'] = ensure_datetime_col(filtered_data, 'deal_created_time')
    if filtered_data['deal_created_time'] is None:
        st.warning("مشکلی به وجود آمده است. لطفا با ادمین تماس بگیرید.")
        return

    # Calculate date ranges
    today = datetime.today().date()
    try:
        start_date = jdatetime.date(1404, 2, 28).togregorian()
    except Exception:
        start_date = today

    # Filter by start date
    filtered_data = filtered_data[filtered_data['deal_created_time'] >= pd.to_datetime(start_date)]

    # Calculate current week
    weeks_passed = (today - start_date).days // 7
    current_week_start = start_date + timedelta(weeks=weeks_passed)
    
    # Last 4 weeks ranges
    week_ranges = [(current_week_start - timedelta(weeks=i), 
                    current_week_start - timedelta(weeks=i-1) - timedelta(days=1)) 
                   for i in range(4, 0, -1)]

    # Display current week info
    try:
        jalali_start = jdatetime.date.fromgregorian(date=current_week_start)
        jalali_end = jdatetime.date.fromgregorian(date=today)
        end_week = jdatetime.date.fromgregorian(date=current_week_start + timedelta(6))
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"شروع هفته: {jalali_start.strftime('%Y/%m/%d')} \n پایان هفته: {end_week.strftime('%Y/%m/%d')}")
        with col2:
            st.info(f"امروز: {jalali_end.strftime('%Y/%m/%d')}")
    except Exception as e:
        handel_errors(e, "Jalali date error", show_error=False, raise_exception=False)

    # Calculate team metrics
    weekly_metrics = [calculate_weekly_metrics(filtered_data, start, end) for start, end in week_ranges]
    weekly_counts, weekly_values, weekly_avgs = zip(*weekly_metrics)

    # This week metrics
    date_series = filtered_data['deal_created_time'].dt.date
    this_week_mask = (date_series >= current_week_start) & (date_series <= today)
    this_week_count = filtered_data[this_week_mask].shape[0]
    this_week_value = pd.to_numeric(filtered_data[this_week_mask]['deal_value'], errors='coerce').sum() / 10
    this_week_avg = this_week_value / this_week_count if this_week_count > 0 else 0

    max_count_week = weekly_counts.index(max(weekly_counts)) if weekly_counts else 0
    max_value_week = weekly_values.index(max(weekly_values)) if weekly_values else 0
    max_avg_week = weekly_avgs.index(max(weekly_avgs)) if weekly_avgs else 0

    # Team overview
    st.subheader("📈 آمار کلی تیم")
    col1, col2 = st.columns(2)

    with col1:
        today_count = filtered_data[date_series == today].shape[0]
        display_metrics(col1, [
            ("تعداد فروش امروز", today_count, ""),
            ("تعداد فروش این هفته", this_week_count, ""),
            ("بیشترین تعداد فروش هفتگی", max(weekly_counts), f" ({4 - max_count_week} هفته پیش)"),
        ])

    with col2:
        today_value = pd.to_numeric(filtered_data[date_series == today]['deal_value'], errors='coerce').sum() / 10
        display_metrics(col2, [
            ("مقدار فروش امروز", today_value, " تومان"),
            ("مقدار فروش این هفته", this_week_value, " تومان"),
            ("بیشترین مقدار فروش هفتگی", max(weekly_values), f" تومان ({4 - max_value_week} هفته پیش)"),
        ])

    # Team charts
    col1, col2 = st.columns(2)

    with col1:
        df_counts = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=s).strftime("%m/%d")} - {jdatetime.date.fromgregorian(date=e).strftime("%m/%d")}' for s, e in week_ranges],
            'تعداد': weekly_counts,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=s).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=e).strftime("%Y/%m/%d")}' for s, e in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_counts, 'هفته', 'تعداد', 'تعداد فروش هفتگی تیم', max_count_week), config={'responsive': True})

    with col2:
        df_values = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=s).strftime("%m/%d")} - {jdatetime.date.fromgregorian(date=e).strftime("%m/%d")}' for s, e in week_ranges],
            'مقدار': weekly_values,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=s).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=e).strftime("%Y/%m/%d")}' for s, e in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_values, 'هفته', 'مقدار', 'مقدار فروش هفتگی تیم', max_value_week), config={'responsive': True})

    # Team average metrics
    st.subheader("📊 میانگین معاملات تیم")
    col1, col2 = st.columns(2)
    
    with col1:
        today_mean = pd.to_numeric(filtered_data[date_series == today]['deal_value'], errors='coerce').mean() / 10
        display_metrics(col1, [
            ("میانگین امروز", today_mean, " تومان"),
            ("میانگین این هفته", this_week_avg, " تومان"),
            ("بیشترین میانگین هفتگی", max(weekly_avgs), f" تومان ({4 - max_avg_week} هفته پیش)"),
        ])

    with col2:
        df_avg = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=s).strftime("%m/%d")} - {jdatetime.date.fromgregorian(date=e).strftime("%m/%d")}' for s, e in week_ranges],
            'میانگین': weekly_avgs,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=s).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=e).strftime("%Y/%m/%d")}' for s, e in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', 'میانگین معامله‌های تیم', max_avg_week), config={'responsive': True})

    # Target and reward section
    st.subheader("🎯 تارگت پاداش")
    reward_percentage = 0.05
    target = max(weekly_values) * 0.9 if weekly_values else 0
    progress_percentage = (this_week_value / target) * 100 if target > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("تارگت هفته", f"{target:,.0f} تومان")
        if this_week_value > target:
            reward = reward_percentage * (this_week_value - target)
            st.success(f"🎉 پاداش: {reward:,.0f} تومان")
        else:
            remaining = target - this_week_value
            st.warning(f"⏳ باقیمانده: {remaining:,.0f} تومان")

    with col2:
        display_percentage = min(progress_percentage, 100.0)
        fig = go.Figure()
        fig.add_trace(go.Pie(
            values=[display_percentage, 100 - display_percentage],
            hole=.8,
            marker_colors=['#00FF00', '#E5ECF6'],
            showlegend=False,
            textinfo='none'
        ))
        fig.update_layout(
            annotations=[dict(text=f'{display_percentage:.1f}%', x=0.5, y=0.5, font_size=24, showarrow=False)],
            height=250,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, config={'responsive': True})

        st.markdown("---")

    # Show sales for the last 10 weeks (X axis: start and end day/month of each week)
    st.markdown("---")
    num_weeks_sales = 10   # Number of weeks to display in the sales chart
    num_weeks_target_reward = 6  # Number of weeks to display target/reward boxes
    st.subheader(f"📊 فروش {num_weeks_sales} هفته اخیر")
    
    # Defensive extraction of .dt.date
    deal_created_col = ensure_datetime_col(filtered_data, 'deal_created_time')
    deal_date_values = deal_created_col.dt.date if pd.api.types.is_datetime64_any_dtype(deal_created_col) else deal_created_col

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
        value = filtered_data[mask]['deal_value'].astype(float).sum() / 10
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
        textposition='outside',
    ))
    # set max y 1.1 times the max sales for better visibility
    max_sales = max(all_weekly_sales) if all_weekly_sales else 0
    fig_sales.update_layout(
        title="",
        yaxis_range=[0, max_sales * 1.1],
        xaxis_title="هفته (شروع - پایان)",
        yaxis_title="فروش (تومان)",
        font=dict(family="Tahoma", size=16),
        height=500,
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

    st.subheader("👥 آمار اعضای تیم")

    # Member metrics
    if role in ['member', 'manager']:
        display_member_metrics(filtered_data, name, week_ranges, today, current_week_start, True)

    # Manager view
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

def display_member_metrics(data, member, week_ranges, today, current_week_start, show_as_you=False):
    """Display metrics for a specific member."""
    member_data = data[data['deal_owner'] == member].copy()
    date_series = member_data['deal_created_time'].dt.date

    # Calculate metrics
    member_metrics = [calculate_weekly_metrics(member_data, start, end) for start, end in week_ranges]
    member_counts, member_values, member_avgs = zip(*member_metrics)

    # This week
    this_week_mask = (date_series >= current_week_start) & (date_series <= today)
    this_week_count = member_data[this_week_mask].shape[0]
    this_week_value = pd.to_numeric(member_data[this_week_mask]['deal_value'], errors='coerce').sum() / 10
    this_week_data = member_data[this_week_mask].reset_index(drop=True)

    max_count_week = member_counts.index(max(member_counts)) if member_counts else 0
    max_value_week = member_values.index(max(member_values)) if member_values else 0

    title = "👤 آمار شما" if show_as_you else f"👤 آمار {member}"
    st.subheader(title)

    col1, col2 = st.columns(2)
    with col1:
        today_count = member_data[date_series == today].shape[0]
        display_metrics(col1, [
            ("تعداد فروش امروز", today_count, ""),
            ("تعداد فروش این هفته", this_week_count, ""),
            ("بیشترین تعداد فروش", max(member_counts), f" ({4 - max_count_week} هفته پیش)"),
        ])

    with col2:
        today_value = pd.to_numeric(member_data[date_series == today]['deal_value'], errors='coerce').sum() / 10
        display_metrics(col2, [
            ("مقدار فروش امروز", today_value, " تومان"),
            ("مقدار فروش این هفته", this_week_value, " تومان"),
            ("بیشترین مقدار فروش", max(member_values), f" تومان ({4 - max_value_week} هفته پیش)"),
        ])

    # Data table
    with st.expander('📋 لیست معاملات' + (' شما' if show_as_you else f' {member}'), expanded=False):
        st.dataframe(member_data , hide_index=True)
        download_buttons(this_week_data, f'deals_{member}')


def get_username():
    """Helper to get current user for logging."""
    try:
        return st.session_state.get('userdata', {}).get('name', 'unknown')
    except Exception:
        return 'unknown'

if __name__ == "__main__":
    main()
