import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
from typing import Optional, Dict, List
from utils.write_data import write_df_to_sheet
from utils.sheetConnect import get_sheet_names
from utils.funcs import download_buttons, load_data_cached, handel_errors
from utils.custom_css import apply_custom_css
from utils.sidebar import render_sidebar


# --- Constants ---
MONTH_NAMES = {
    '01': 'فروردین', '02': 'اردیبهشت', '03': 'خرداد',
    '04': 'تیر', '05': 'مرداد', '06': 'شهریور',
    '07': 'مهر', '08': 'آبان', '09': 'آذر',
    '10': 'دی', '11': 'بهمن', '12': 'اسفند'
}

# --- Utility Functions ---
@st.cache_data(ttl=600)
def safe_to_jalali(date_value) -> Optional[str]:
    """Convert Gregorian date to Jalali date safely."""
    try:
        jalali_date = jdatetime.date.fromgregorian(date=pd.to_datetime(date_value).date())
        return jalali_date.strftime('%Y-%m-%d')
    except Exception as e:
        handel_errors(e, "Date conversion error", show_error=False)
        return None

def get_jalali_month_string(date_obj) -> str:
    """Get year-month string from Jalali date."""
    if isinstance(date_obj, str):
        # Parse string format 'YYYY-MM-DD' to extract year and month
        try:
            year, month = date_obj.split('-')[:2]
            return f"{year}-{month}"
        except Exception:
            return None
    elif isinstance(date_obj, jdatetime.date):
        return f"{date_obj.year}-{date_obj.month:02d}"
    return None

# --- Data Processing Functions ---
def find_eval_sheet(target_month: str, sheet_names: List[str]) -> Optional[str]:
    """Find evaluation sheet name for given month."""
    try:
        year, month = target_month.split('-')
        month_name = MONTH_NAMES.get(month)
        
        if not month_name:
            return None
            
        matching_sheets = [
            sheet for sheet in sheet_names 
            if month_name in sheet and (year in sheet or year[1:] in sheet)
        ]
        
        return matching_sheets[0] if matching_sheets else None
        
    except Exception as e:
        handel_errors(e, "Error finding eval sheet", show_error=False)
        return None

def calculate_wolf_scores(wolf_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Wolf of Wall Street scores for team members."""
    try:
        persons = {}
        
        for _, row in wolf_data.iterrows():
            person = row['name']
            score = int(row['score'])
            
            if person in persons:
                persons[person]['score'] += score
                if score == 3:
                    persons[person]['first_place_count'] += 1
            else:
                persons[person] = {
                    'deal_owner': person,
                    'score': score,
                    'first_place_count': 1 if score == 3 else 0,
                }
        
        return pd.DataFrame.from_dict(persons, orient='index').reset_index(drop=True)
        
    except Exception as e:
        handel_errors(e, "Error calculating wolf scores", show_error=False)
        return pd.DataFrame()
    
def calculate_sherlock_scores(wolf_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Wolf of Wall Street scores for team members."""
    try:
        persons = {}
        
        for _, row in wolf_data.iterrows():
            person = row['name']
            score = int(row['score'])
            
            if person in persons:
                persons[person]['score'] += score
                if score == 10:
                    persons[person]['first_place_count'] += 1
            else:
                persons[person] = {
                    'deal_owner': person,
                    'score': score,
                    'first_place_count': 1 if score == 3 else 0,
                }
        
        return pd.DataFrame.from_dict(persons, orient='index').reset_index(drop=True)
        
    except Exception as e:
        handel_errors(e, "Error calculating sherlock scores", show_error=False)
        return pd.DataFrame()

def calculate_reward_percentages(
    wolf_board: pd.DataFrame,
    sherlock_board: pd.DataFrame,
    parameters: Dict,
    team_members: List[str]
) -> Dict[str, Dict[str, float]]:
    """Calculate reward percentages for each team member."""
    
    # Initialize percentages
    wolf_percentages = {member: 0 for member in team_members}
    sherlock_percentages = {member: 0 for member in team_members}

    try:
        # Calculate Wolf percentages
        if not wolf_board.empty:
            wolf_sorted = wolf_board.sort_values("score", ascending=False).reset_index(drop=True)
            max_score = wolf_sorted.iloc[0]["score"]
            
            first_place = wolf_sorted[wolf_sorted["score"] == max_score]
            first_names = first_place["deal_owner"].tolist()
            
            wolf1_pct = int(parameters['Wolf1'].values[0])
            wolf2_pct = int(parameters['Wolf2'].values[0])
            
            if len(first_names) > 1:
                # Split first and second place rewards
                split_pct = (wolf1_pct + wolf2_pct) / len(first_names)
                for member in first_names:
                    wolf_percentages[member] = split_pct
            else:
                # Award first place
                wolf_percentages[first_names[0]] = wolf1_pct
                
                # Award second place
                second_group = wolf_sorted[wolf_sorted["score"] < max_score]
                if not second_group.empty:
                    second_max = second_group.iloc[0]["score"]
                    second_place = wolf_sorted[wolf_sorted["score"] == second_max]
                    second_names = second_place["deal_owner"].tolist()
                    
                    split_pct = wolf2_pct / len(second_names)
                    for member in second_names:
                        wolf_percentages[member] = split_pct
        
        # Calculate Sherlock percentages
        if not sherlock_board.empty:
            sherlock_sorted = sherlock_board.sort_values("score", ascending=False).reset_index(drop=True)
            max_score = sherlock_sorted.iloc[0]["score"]
            
            first_place = sherlock_sorted[sherlock_sorted["score"] == max_score]
            first_names = first_place["deal_owner"].tolist()
            
            sherlock_pct = int(parameters['Sherlock'].values[0])
            split_pct = sherlock_pct / len(first_names)
            
            for member in first_names:
                sherlock_percentages[member] = split_pct
        
        # Combine all percentages
        result = {}
        performance_base = int(parameters['Performance'].values[0])
        parameters = parameters[['name_in_eval_sheet', 'name', 'precent']]
        for member in team_members:
            if member in parameters['name'].values:
                member_performance = float(parameters[parameters['name']==member]['precent'].values[0].replace('%', ''))
            else:
                member_performance = 0
            performance_pct = member_performance * performance_base / 100
            
            result[member] = {
                'performance_percent': performance_pct,
                'wolf_percent': wolf_percentages.get(member, 0),
                'sherlock_percent': sherlock_percentages.get(member, 0),
                'total_percent': performance_pct + wolf_percentages.get(member, 0) + sherlock_percentages.get(member, 0)
            }
        return result

    except Exception as e:
        handel_errors(e, "Error calculating reward percentages", show_error=False)
        return {}
    
def calculate_reward_details(
    reward_amount: float,
    wolf_board: pd.DataFrame,
    sherlock_board: pd.DataFrame,
    parameters: Dict,
    team_members: List[str]
) -> pd.DataFrame:
    """Calculate detailed reward breakdown for each team member."""
    try:
        if reward_amount <= 0:
            return pd.DataFrame()
        
        percentages = calculate_reward_percentages(
            wolf_board, sherlock_board, parameters, team_members
        )
        
        member_stats = pd.DataFrame.from_dict(percentages, orient='index')
        member_stats['پاداش'] = member_stats['total_percent'] * reward_amount / 100
        
        member_stats = member_stats.rename(columns={
            'total_percent': 'درصد کل',
            'performance_percent': 'درصد عملکرد',
            'wolf_percent': 'درصد گرگ',
            'sherlock_percent': 'درصد شرلوک'
        }).reset_index().rename(columns={'index': 'کارشناس'})
        
        return member_stats
        
    except Exception as e:
        handel_errors(e, "Error calculating rewards", show_error=False)
        return pd.DataFrame()

# --- UI Display Functions ---
def display_sale_type_metrics(deals: pd.DataFrame, sale_type: str, container):
    """Display metrics for a specific sale type."""
    deals_filtered = deals[deals['deal_type'] == sale_type].copy()

    title = 'فروش جدید' if sale_type == 'فروش جدید' else 'تمدید'
    container.subheader(title)
    
    if deals_filtered.empty:
        container.info("داده‌ای برای نمایش وجود ندارد")
        return
    
    total_deals = len(deals_filtered)
    total_nights = deals_filtered['product_quantity'].astype(float).sum()
    total_value = deals_filtered['deal_value'].astype(float).sum() / 10
    avg_nights = round(total_nights / total_deals, 1) if total_deals > 0 else 0
    avg_value = round(total_value / total_deals, 1) if total_deals > 0 else 0
    
    container.metric('مجموع فروش', f"{total_value:,.1f} تومان")
    container.metric('تعداد کل معاملات', f"{total_deals:,}")
    container.metric('میانگین تعداد شب', f"{avg_nights:,}")
    container.metric('میانگین ارزش معامله', f"{avg_value:,.1f} تومان")

def display_progress_chart(progress_pct: float):
    """Display circular progress chart."""
    display_pct = min(progress_pct, 100.0)
    color = '#00FF00' if display_pct >= 100 else '#FFA500'
    
    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=[display_pct, 100 - display_pct],
        hole=.8,
        marker_colors=[color, '#E5ECF6'],
        showlegend=False,
        textinfo='none',
        rotation=90,
        pull=[0.1, 0]
    ))
    
    status_text = 'تکمیل شده' if display_pct >= 100 else 'در حال پیشرفت'
    
    fig.update_layout(
        annotations=[
            dict(text=f'{display_pct:.1f}%', x=0.5, y=0.5, 
                 font_size=24, font_color='#2F4053', showarrow=False),
            dict(text=status_text, x=0.5, y=0.35, 
                 font_size=14, font_color='#2E4053', showarrow=False)
        ],
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, config={'responsive': True})

def display_team_metrics(deals: pd.DataFrame, parameters: Dict) -> float:
    """Display team-wide metrics and return calculated reward."""
    try:
        st.subheader("📊 آمار کلی تیم")
        
        # Display metrics by sale type
        cols = st.columns(2)
        for sale_type in ['فروش جدید', 'تمدید']:
            if sale_type ==  'فروش جدید' :
                container = cols[0]
            else:
                container = cols[1]
            display_sale_type_metrics(deals, sale_type, container)
        
        st.markdown('---')
        
        # Calculate totals and reward
        target = float(parameters.get('Target', 0))
        reward_pct = float(parameters.get('Reward percent', 0))
        total_sales = deals['deal_value'].astype(float).sum() / 10
        
        diff = total_sales - target
        reward_amount = max(0, diff) * reward_pct / 100
        progress_pct = (total_sales / target * 100) if target > 0 else 0
        remaining = target - total_sales
        # Display summary
        col1, col2 = st.columns(2)
        
        with col1:
            if reward_amount > 0:
                st.metric("کل فروش", f"{total_sales:,.0f} تومان")
                st.metric("تارگت", f"{target:,.0f} تومان")
                st.success(f"🏆 پاداش کل تیم: {reward_amount:,.0f} تومان")
            else:
                st.metric("تارگت", f"{target:,.0f} تومان")
                st.warning(f"⏳ باقیمانده: {remaining:,.0f} تومان")
                st.info(f"🎯 {100 - progress_pct:.1f}% تا هدف")
        
        with col2:
            st.subheader("میزان پیشرفت")
            display_progress_chart(progress_pct)
        
        return reward_amount
        
    except Exception as e:
        handel_errors(e, "Error displaying team metrics", show_error=False)
        raise

def display_deals_chart(deals: pd.DataFrame, member_name: Optional[str] = None):
    """Display daily deals chart (count or value)."""
    try:
        if deals.empty:
            return
        
        col1, col2 = st.columns([1, 3])

        with col1:
            plot_type = st.radio(
                '', ['تعداد معاملات', 'میزان فروش'],
                key=f'plot_type_{member_name or "team"}'
            )
        
        with col2:
            title_prefix = f"{member_name}" if member_name else "تیم"
            
            if plot_type == 'میزان فروش':
                st.subheader(f"💹 میزان فروش روزانه {title_prefix}")
                deals['deal_value'] = deals['deal_value'].astype(float) / 10
                daily_values = deals.groupby('deal_created_time')['deal_value'].sum().reset_index()
                daily_values['تاریخ شمسی'] = daily_values['deal_created_time'].apply(safe_to_jalali)
                
                fig = px.line(
                    daily_values, x='deal_created_time', y='deal_value',
                    labels={'deal_created_time': 'تاریخ', 'deal_value': 'مجموع ارزش'},
                    hover_data=['تاریخ شمسی'], markers=True
                )
            else:
                st.subheader(f"📈 تعداد معاملات روزانه {title_prefix}")
                
                daily_counts = deals.groupby('deal_created_time').size().reset_index(name='count')
                daily_counts['تاریخ شمسی'] = daily_counts['deal_created_time'].apply(safe_to_jalali)
                
                y_max = int(daily_counts['count'].max() * 1.15) + 1
                
                fig = px.line(
                    daily_counts, x='deal_created_time', y='count',
                    labels={'deal_created_time': 'تاریخ', 'count': 'تعداد'},
                    hover_data=['تاریخ شمسی'], markers=True
                )
                fig.update_layout(yaxis=dict(range=[0, y_max]))
            
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, config={'responsive': True})
            
    except Exception as e:
        handel_errors(e, "Error displaying chart", show_error=False)

def display_member_details(deals: pd.DataFrame, member_name: str):
    """Display detailed statistics for a team member."""
    try:
        st.subheader(f"👤 آمار عملکرد {member_name}")
        
        member_deals = deals[deals['deal_owner'] == member_name].reset_index(drop=True)
        
        if member_deals.empty:
            st.info(f"داده‌ای برای {member_name} وجود ندارد")
            return
        
        # Display metrics by sale type
        cols = st.columns(2)
        for sale_type in ['فروش جدید', 'تمدید']:
            if sale_type ==  'فروش جدید' :
                container = cols[0]
            else:
                container = cols[1]
            display_sale_type_metrics(member_deals, sale_type, container)
        

        # Display charts
        display_deals_chart(member_deals, member_name)
        
        # Display deals list
        with st.expander(f"📋 لیست معاملات {member_name}", expanded=False):
            st.dataframe(member_deals, width='stretch', hide_index=True)
        
            download_buttons(member_deals, f'deals_{member_name}')
                
    except Exception as e:
        handel_errors(e, "Error displaying member details", show_error=False)

# --- Main Application ---
def sales():
    """Main sales dashboard function."""
    apply_custom_css()
    render_sidebar()
    
    # Check authentication
    if not st.session_state.get('logged_in'):
        st.warning("لطفا ابتدا وارد سیستم شوید")
        return
    
    st.title("📊 داشبورد تیم Sales")
    
    # Check access
    userdata = st.session_state.get('userdata', {})
    teams = [t.strip() for t in userdata.get('team', '').split('|')]
    role = st.session_state.userdata.get('role', '')
    username_in_didar = st.session_state.userdata.get('username_in_didar', '')

    if 'sales' not in teams:
        st.error("شما به این بخش دسترسی ندارید")
        return
    
    # Get team members
    team_users = st.session_state.users[
        st.session_state.users['team'].apply(
            lambda x: 'sales' in [t.strip() for t in x.split('|')]
        ) & (st.session_state.users['role'] != 'admin')
    ]
    team_members = team_users['username_in_didar'].tolist()
    team_member_names = team_users['name'].tolist()
    
    # Prepare deals data
    deals = st.session_state.deals_data.copy()
    deals = deals[
        (deals['deal_type'].isin(['فروش جدید', 'تمدید'])) &
        (deals['deal_status'] == 'Won')
    ]
    deals = deals[deals['deal_owner'].isin(team_members)]
    deals['deal_created_time'] = pd.to_datetime(deals['deal_created_time']).dt.date
    deals['jalali_date'] = deals['deal_created_time'].apply(safe_to_jalali)
    deals['jalali_year_month'] = deals['jalali_date'].apply(get_jalali_month_string)
    
    # Month selection
    month_choice = st.selectbox('ماه مورد نظر:', ['این ماه', 'ماه پیش'])
    today = jdatetime.date.today()
    
    if month_choice == 'ماه پیش':
        last_month = (today.replace(day=1) - jdatetime.timedelta(days=1))
        target_month = get_jalali_month_string(last_month)
    else:
        target_month = get_jalali_month_string(today)
    
    deals_filtered = deals[deals['jalali_year_month'] == target_month]
    st.info(f'نمایش آمار برای ماه: {target_month}')
    
    
    # Load supporting data
    try:
        # Load Wolf of Wall Street data
        wolf_data = load_data_cached(spreadsheet_key='MAIN_SPREADSHEET_ID', sheet_name='Wolf')
        wolf_data = wolf_data[
            (~wolf_data['date'].isna())
            | (~wolf_data['name'].isna())
        ]
        wolf_data['jalali_date'] = wolf_data['date'].apply(safe_to_jalali)
        wolf_data['month'] = wolf_data['jalali_date'].apply(get_jalali_month_string)
        wolf_data = wolf_data[wolf_data['month'] == target_month]
        wolf_board = calculate_wolf_scores(wolf_data)
        
        # Load Sherlock data
        sherlock_data = load_data_cached(spreadsheet_key='CHAMPIONS_SPREADSHEET_ID', sheet_name='Sherlock BI dashboard')
        sherlock_data = sherlock_data[
            (~sherlock_data['name'].isna())
        ][['Date', 'Jalali Date', 'name', 'score']]
        sherlock_data['jalali_date'] = sherlock_data['Date'].apply(safe_to_jalali)
        sherlock_data['month'] = sherlock_data['jalali_date'].apply(get_jalali_month_string)
        sherlock_data = sherlock_data[sherlock_data['month'] == target_month]
        sherlock_board = calculate_sherlock_scores(sherlock_data)  # Same calculation logic

        # Load parameters
        parameters_data = load_data_cached(spreadsheet_key='MAIN_SPREADSHEET_ID', sheet_name='Sales team parameters')
        
    except Exception as e:
        handel_errors(e, "Error loading supporting data", show_error=False)
        st.error("خطا در بارگذاری داده‌ها")
        return
    parameters = parameters_data.loc[0, ['Target', 'Reward percent', 'Wolf1', 'Wolf2', 'Sherlock', 'Performance']].to_dict()
    
    # Display team metrics and calculate reward
    st.markdown('---')
    reward_amount = display_team_metrics(deals_filtered, parameters)
    
    # Display reward breakdown if applicable
    
    if role in ['manager', 'admin']:
        tabs = st.tabs(['جزئیات پاداش', 'بازی گرگ', 'بازی شرلوک', 'تنظیمات'])
        
        with tabs[0]:
            st.subheader("💰 توزیع پاداش")
            if reward_amount > 0:
                reward_details = calculate_reward_details(
                    reward_amount, wolf_board, sherlock_board, parameters_data, team_members
                )
                
                if not reward_details.empty:
                    st.dataframe(
                        reward_details.sort_values('درصد کل', ascending=False).style.format({
                            'پاداش': '{:,.0f}',
                            'درصد کل': '{:.2f}%',
                            'درصد عملکرد': '{:.2f}%',
                            'درصد گرگ': '{:.2f}%',
                            'درصد شرلوک': '{:.2f}%'
                        }),
                        width='stretch',
                        hide_index=True
                    )
        with tabs[1]:
            st.write("### 🐺 بازی گرگ وال استریت")

            if not wolf_board.empty:
                st.dataframe(
                    wolf_board.sort_values(by='score', ascending=False).reset_index(drop=True).style.format({
                        'score': '{:,.0f}',
                        'sales_amount': '{:,.0f}'
                    }),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("داده‌ای برای نمایش وجود ندارد")
        with tabs[2]:
            st.write("### 🕵️ بازی شرلوک")

            if not sherlock_board.empty:
                st.dataframe(
                    sherlock_board.sort_values(by='score', ascending=False).reset_index(drop=True).style.format({
                        'score': '{:,.0f}',
                        'sales_amount': '{:,.0f}'
                    }),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("داده‌ای برای نمایش وجود ندارد")
        
        with tabs[3]:
            st.write("### ⚙️ تنظیمات تیم فروش")
            eval_sheet_names = get_sheet_names('EVAL_SPREADSHEET_ID')
            sheet_name = find_eval_sheet(target_month, eval_sheet_names)
            if sheet_name:
                st.success(f"تب ارزیابی برای ماه انتخاب شده: {sheet_name}")
                sheet_data = load_data_cached(spreadsheet_key='EVAL_SPREADSHEET_ID', sheet_name=sheet_name)
                sheet_data.columns = sheet_data.columns.str.strip()

                parameters_data['name'] = parameters_data['name'].str.strip()
                parameters_data['name_in_eval_sheet'] = parameters_data['name_in_eval_sheet'].str.strip()

                eval_name_map = parameters_data.set_index('name')['name_in_eval_sheet'].to_dict()
                for idx, member_name in enumerate(team_member_names):
                    eval_name = eval_name_map.get(team_members[idx].strip(), None)
                    if eval_name: 
                        precent = sheet_data[sheet_data['KPI']=='پاداش نهایی کل تیم'][eval_name]
                        parameters_data.loc[
                            parameters_data['name'] == team_members[idx], 
                            'precent'
                        ] = precent.values[0]
                parameters_data['update_at'] = None
                parameters_data.loc[0, 'update_at'] = pd.Timestamp.now()

            else:
                st.warning("تب ارزیابی برای ماه انتخاب شده یافت نشد")

            with st.form("update_parameters_form"):
                cols = st.columns(2)
                with cols[0]:
                    # general parameters inputs                
                    st.number_input('تارگت:', value=int(parameters_data.loc[0, 'Target']), key='target_input')
                    st.number_input('درصد پاداش:', value=int(parameters_data.loc[0, 'Reward percent']), key='reward_percent_input')
                    st.number_input('wolf1:', value=int(parameters_data.loc[0, 'Wolf1']), key='wolf1_input')
                    st.number_input('wolf2:', value=int(parameters_data.loc[0, 'Wolf2']), key='wolf2_input')
                    st.number_input('sherlock:', value=int(parameters_data.loc[0, 'Sherlock']), key='sherlock_input')
                    st.number_input('performance:', value=int(parameters_data.loc[0, 'Performance']), key='performance_input')
                with cols[1]:    
                # users precent inputs
                    for idx, member_name in enumerate(team_member_names):
                        percent_value = parameters_data.loc[
                            parameters_data['name'] == team_members[idx], 
                            'precent'
                        ]
                        if not percent_value.empty:
                            percent_str = str(percent_value.values[0]).replace('%', '').strip()
                            percent_float = float(percent_str) if percent_str else 0.0
                        else:
                            percent_float = 0.0
                        
                        st.number_input(
                            f"درصد {member_name}:", 
                            value=percent_float,
                            key=f'percent_input_{team_members[idx]}'
                        )
                
                submitted = st.form_submit_button("ذخیره تنظیمات در شیت اصلی", type='primary')
                
                if submitted:
                    # update parameters_data with new inputs
                    parameters_data.loc[0, 'Target'] = st.session_state['target_input']
                    parameters_data.loc[0, 'Reward percent'] = st.session_state['reward_percent_input']
                    parameters_data.loc[0, 'Wolf1'] = st.session_state['wolf1_input']
                    parameters_data.loc[0, 'Wolf2'] = st.session_state['wolf2_input']
                    parameters_data.loc[0, 'Sherlock'] = st.session_state['sherlock_input']
                    parameters_data.loc[0, 'Performance'] = st.session_state['performance_input']
                    for idx, member_name in enumerate(team_member_names):
                        parameters_data.loc[
                            parameters_data['name'] == team_members[idx], 
                            'precent'
                        ] = st.session_state[f'percent_input_{team_members[idx]}']
                    
                    st.markdown("""
                    **توجه:** با کلیک روی دکمه زیر، تنظیمات فعلی تیم فروش با مقادیر موجود در شیت ارزیابی به‌روزرسانی خواهد شد.
                    لطفاً قبل از انجام این کار از صحت داده‌ها اطمینان حاصل کنید.
                    """)
                    write_df_to_sheet(
                        parameters_data,
                        spreadsheet_key='MAIN_SPREADSHEET_ID',
                        sheet_name='Sales team parameters',
                        clear_sheet=True
                    )
    # Display team-wide charts
    display_deals_chart(deals_filtered)
    
    # Display all deals list
    with st.expander("📋 لیست کل معاملات تیم", expanded=False):
        st.dataframe(deals_filtered, width='stretch', hide_index=True)
        
        download_buttons(deals_filtered, f'team_deals_{target_month}')
    
    if role not in ['manager', 'admin']:
        st.markdown('---')
        st.header("📈 آمار شما")
        display_member_details(deals_filtered, username_in_didar)

    else:
        # Individual member details
        st.markdown('---')
        st.header("📈 آمار تفکیکی اعضای تیم")

        tabs = st.tabs(team_member_names)
        
        for idx, member_name in enumerate(team_member_names):
            with tabs[idx]:
                display_member_details(deals_filtered, team_members[idx])

    
if __name__ == "__main__":
    sales()
