import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
from utils.func import convert_df, convert_df_to_excel
from utils.custom_css import apply_custom_css
from utils.logger import log_event
from utils.sidebar import render_sidebar

def ensure_datetime_col(df, col):
    """Ensure a column is in datetime64 format."""
    if col not in df.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            return pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            log_event(st.session_state.userdata.get('name', ''), "Error", f"Failed to convert '{col}': {e}")
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
        log_event(st.session_state.userdata.get('name', ''), "Error", f"Failed to calculate metrics: {e}")
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
        log_event(st.session_state.userdata.get('name', ''), "Error", f"Error creating chart: {e}")
        return go.Figure()

def display_metrics(col, metrics):
    """Display metrics in a standardized format."""
    for label, value, suffix in metrics:
        if pd.isna(value):
            value = 0
        try:
            col.metric(label, f"{value:,.0f}{suffix}")
        except Exception as e:
            log_event(st.session_state.userdata.get('name', ''), "Error", f"Error displaying '{label}': {e}")

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

    team_users = st.session_state.users[
        (st.session_state.users['team'].apply(lambda x: 'b2b' in [team.strip() for team in x.split('|')]))&
        (st.session_state.users['role'] != 'admin')
    ]

    data = st.session_state.deals_data.copy()

    # Filter data for B2B team
    filtered_data = data[
        (data['deal_source'].isin(['مهمان واسطه', 'فرودگاه'])) &
        (data['deal_owner'].isin(team_users['username_in_didar'].values))
    ].copy()

    # Normalize owner names
    filtered_data['deal_owner'] = filtered_data['deal_owner'].apply(normalize_owner)

    # Ensure datetime column
    filtered_data['deal_created_time'] = ensure_datetime_col(filtered_data, 'deal_created_time')
    if filtered_data['deal_created_time'] is None:
        st.error("خطا در پردازش تاریخ معاملات")
        return

    # Calculate date ranges
    today = datetime.today().date()
    try:
        start_date = jdatetime.date(1404, 2, 28).togregorian()
    except Exception as e:
        log_event(name, "Error", f"Date conversion error: {e}")
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
        log_event(name, "Error", f"Jalali date error: {e}")

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
        st.plotly_chart(create_weekly_chart(df_counts, 'هفته', 'تعداد', 'تعداد فروش هفتگی تیم', max_count_week), use_container_width=True)

    with col2:
        df_values = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=s).strftime("%m/%d")} - {jdatetime.date.fromgregorian(date=e).strftime("%m/%d")}' for s, e in week_ranges],
            'مقدار': weekly_values,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=s).strftime("%Y/%m/%d")} تا {jdatetime.date.fromgregorian(date=e).strftime("%Y/%m/%d")}' for s, e in week_ranges]
        })
        st.plotly_chart(create_weekly_chart(df_values, 'هفته', 'مقدار', 'مقدار فروش هفتگی تیم', max_value_week), use_container_width=True)

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
        st.plotly_chart(create_weekly_chart(df_avg, 'هفته', 'میانگین', 'میانگین معامله‌های تیم', max_avg_week), use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

    # Member metrics
    if role in ['member', 'manager']:
        display_member_metrics(filtered_data, name, week_ranges, today, current_week_start, True)

    # Manager view
    if role in ['manager', 'admin']:
        team_members = team_users[
            (team_users['name'] != name) &
            (team_users['username_in_didar'].isin(filtered_data['deal_owner'].unique()))
        ]['username_in_didar'].unique()

        if len(team_members) > 0:
            selected_member = st.selectbox("انتخاب عضو تیم", team_members, key="b2b_member_select")
            if selected_member:
                display_member_metrics(filtered_data, selected_member, week_ranges, today, current_week_start, False)

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
        st.dataframe(this_week_data, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("دانلود CSV", convert_df(this_week_data), f'deals_{member}.csv', 'text/csv')
        with col2:
            st.download_button("دانلود اکسل", convert_df_to_excel(this_week_data), f'deals_{member}.xlsx', 
                             'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == "__main__":
    main()
