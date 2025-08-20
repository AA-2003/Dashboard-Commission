import streamlit as st
import pandas as pd
import plotly.express as px
import jdatetime
from utils.write_sheet import write_df_to_sheet
from utils.load_sheet import load_sheet, load_sheet_uncache

# --- Data Transformation & Calculation Functions ---

@st.cache_data(ttl=600)
def safe_to_jalali(x):
    """
    Safely convert a Gregorian date object to a Jalali date object.
    Caches the result for performance.
    """
    return jdatetime.date.fromgregorian(date=pd.to_datetime(x).date())

def normalize_owner(owner: str) -> str:
    """
    Merges different names for the same person into a single, consistent name.
    Specifically handles day/night shifts for 'محمد آبساران'.
    """
    if owner in ["محمد آبساران/روز", "محمد آبساران/شب"]:
        return "محمد آبساران"
    return owner

def cal_wolfs(df: pd.DataFrame, target_month: str) -> tuple[pd.DataFrame, dict]:
    """
    Calculates the 'Wolf of Wall Street' scores for each deal owner for the current Jalali month.

    Scoring Logic (per day):
    - 1st place: +3 points
    - 2nd place: +2 points
    - 3rd place: +1 point
    - Last place: -1 point
    - Others: 0 points

    Args:
        df: The main deals DataFrame.

    Returns:
        A tuple containing:
        - wolf_board_df: A DataFrame with 'deal_owner' and 'score', sorted by score.
        - first_place_counts: A dictionary mapping owners to their number of 1st place finishes.
    """
    df['deal_created_date'] = pd.to_datetime(df['deal_created_date'])
    df['jalali_date'] = df['deal_created_date'].apply(lambda x: jdatetime.date.fromgregorian(date=x.date()))
    df['jalali_year_month'] = df['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}")

    # Filter for 'New Sale' deals in target month
    df_current_month = df[
        (df['jalali_year_month']==target_month)&
        (df['deal_type'] == "New Sale")
    ]

    if df_current_month.empty:
        return pd.DataFrame(columns=["deal_owner", "score"]), {}

    # Aggregate sales per owner for each day
    daily_sum = df_current_month.groupby([df_current_month['deal_created_date'].dt.date, 'deal_owner'])['deal_value'].sum().reset_index()
    scores = {}
    first_place_counts = {}
    
    # Iterate through each day to rank owners and assign scores
    for _, group in daily_sum.groupby('deal_created_date'):
        group_sorted = group.sort_values('deal_value', ascending=False).reset_index(drop=True)
        num_participants = len(group_sorted)
        
        for idx, row in group_sorted.iterrows():
            owner = row['deal_owner']
            score = 0
            if idx == 0:  # 1st place
                score = 3
                first_place_counts[owner] = first_place_counts.get(owner, 0) + 1
            elif idx == 1:  # 2nd place
                score = 2
            elif idx == 2:  # 3rd place
                score = 1
            
            # Last place gets a penalty, but only if they are not also in the top 3
            if idx == num_participants - 1 and num_participants > 3:
                score = -1
                
            scores[owner] = scores.get(owner, 0) + score

    # Create a final DataFrame with the results
    all_owners = df_current_month['deal_owner'].unique()
    wolf_board = [{'deal_owner': owner, 'score': scores.get(owner, 0)} for owner in all_owners]
    wolf_board_df = pd.DataFrame(wolf_board).sort_values('score', ascending=False).reset_index(drop=True)
    
    return wolf_board_df, first_place_counts

def load_sherlock_data(target_month: str) -> pd.DataFrame:
    """
    Loads data from the 'Sherlock' sheet and calculates scores for the current Jalali month.

    Scoring Logic (per day):
    - First person: +10 points
    - Second person: +5 points
    - Last person: -3 points

    Also maps various nicknames to their official full names.

    Returns:
        A DataFrame with 'deal_owner' and 'score', sorted by score.
    """
    sherlock_df = load_sheet('Sherlock')
    sherlock_df['Date'] = pd.to_datetime(sherlock_df['Date'])
    sherlock_df['jalali_date'] = sherlock_df['Date'].apply(lambda x: jdatetime.date.fromgregorian(date=x.date()))
    sherlock_df['jalali_year_month'] = sherlock_df['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}")


    # Filter for month
    sherlock_df = sherlock_df[
        sherlock_df['jalali_year_month'] == target_month
    ]

    scores = {}
    # Iterate through each day to assign points
    for _, group in sherlock_df.groupby('Date'):
        if group['First person'].iat[0] == '': continue
        
        for p in group['First person'].dropna(): scores[p] = scores.get(p, 0) + 10
        for p in group['Second person'].dropna(): scores[p] = scores.get(p, 0) + 5
        for p in group['Last person'].dropna(): scores[p] = scores.get(p, 0) - 3

    # Map nicknames to full names for consistency
    name_map = {
        'حافظ': 'حافظ قاسمی', 'پویا(وزیری)': 'پویا وزیری', 'محمد (آبساران)': 'محمد آبساران',
        'بابک': 'بابک مسعودی', 'حسین (طاهری)': 'حسین طاهری', 'پویا (شب)': 'پویا ژیانی',
        'پوریا': 'پوریا کیوانی', 'فرشته': 'فرشته فرج نژاد', 'زینب': 'زینب فلاح نژاد',
        'مربی': 'آرمین مربی', '': None,
    }
    valid_names = [
        'حسین طاهری', 'پوریا کیوانی', 'پویا ژیانی', 'فرشته فرج نژاد', 'محمد آبساران',
        'حافظ قاسمی', 'بابک مسعودی', 'زینب فلاح نژاد', 'پویا وزیری', 'آرمین مربی'
    ]

    # Remap scores to full names, summing scores if multiple nicknames map to the same person
    mapped_scores = {}
    for person, score in scores.items():
        mapped_name = name_map.get(person, person)
        if mapped_name in valid_names:
            mapped_scores[mapped_name] = mapped_scores.get(mapped_name, 0) + score

    result_df = pd.DataFrame(list(mapped_scores.items()), columns=['deal_owner', 'score'])
    return result_df.sort_values('score', ascending=False).reset_index(drop=True)

def calculate_reward_details(reward_amount: float, wolf_board: pd.DataFrame, sherlock_board: pd.DataFrame, parameters: dict, team_members_names: list) -> pd.DataFrame:
    """
    Calculates the detailed reward breakdown for each team member.

    The final reward is based on a combination of:
    1.  Performance: A fixed percentage assigned to each member.
    2.  Wolf Score: A percentage for the 1st and 2nd place winners.
    3.  Sherlock Score: A percentage for the 1st place winner.
    
    This function handles complex scenarios like ties for 1st or 2nd place by splitting
    the reward percentage equally among the winners.
    """
    if reward_amount <= 0:
        return pd.DataFrame()

    detail_percent_dict = {}

    # --- WOLF REWARD CALCULATION ---
    # This logic determines who gets the 'Wolf1' (1st place) and 'Wolf2' (2nd place) rewards.
    wolf_percentages = {member: 0 for member in team_members_names}
    if not wolf_board.empty:
        wolf_sorted = wolf_board.sort_values("score", ascending=False).reset_index(drop=True)
        max_score = wolf_sorted.iloc[0]["score"]
        
        # Find all members who tied for the highest score
        wolf_first_group = wolf_sorted[wolf_sorted["score"] == max_score]
        wolf_first_names = wolf_first_group["deal_owner"].tolist()
        
        wolf1_percent_param = parameters.get('Wolf1', 0)
        wolf2_percent_param = parameters.get('Wolf2', 0)

        if len(wolf_first_names) > 1:
            # If there's a tie for first place, the Wolf1 and Wolf2 rewards are combined
            # and split equally among all first-place winners.
            split_percent = (wolf1_percent_param + wolf2_percent_param) / len(wolf_first_names)
            for member in wolf_first_names:
                wolf_percentages[member] = split_percent
        elif len(wolf_first_names) == 1:
            # If there is a single first-place winner, they get the full Wolf1 reward.
            first_place_winner = wolf_first_names[0]
            wolf_percentages[first_place_winner] = wolf1_percent_param

            # Now, find the second-place winner(s)
            second_score_group = wolf_sorted[wolf_sorted["score"] < max_score]
            if not second_score_group.empty:
                second_max_score = second_score_group.iloc[0]["score"]
                wolf_second_group = wolf_sorted[wolf_sorted["score"] == second_max_score]
                wolf_second_names = wolf_second_group["deal_owner"].tolist()
                
                # The Wolf2 reward is split among all second-place winners.
                if wolf_second_names:
                    split_percent = wolf2_percent_param / len(wolf_second_names)
                    for member in wolf_second_names:
                        wolf_percentages[member] = split_percent

    # --- SHERLOCK REWARD CALCULATION ---
    sherlock_percentages = {member: 0 for member in team_members_names}
    if sherlock_board is not None and not sherlock_board.empty:
        sherlock_sorted = sherlock_board.sort_values("score", ascending=False).reset_index(drop=True)
        max_score = sherlock_sorted.iloc[0]["score"]

        # Find all members tied for the highest Sherlock score
        sherlock_first_group = sherlock_sorted[sherlock_sorted["score"] == max_score]
        sherlock_first_names = sherlock_first_group["deal_owner"].tolist()
        
        sherlock_percent_param = parameters.get('Sherlock', 0)
        if sherlock_first_names:
            # Split the Sherlock reward equally among all first-place winners.
            split_percent = sherlock_percent_param / len(sherlock_first_names)
            for member in sherlock_first_names:
                sherlock_percentages[member] = split_percent

    # --- COMBINE ALL REWARDS ---
    for member in team_members_names:
        # Performance reward is based on a pre-set percentage for each member.
        performance_percent = parameters.get(f'{member}_percent', 0) * parameters.get('Performance', 0) / 100
        wolf_percent = wolf_percentages.get(member, 0)
        sherlock_percent = sherlock_percentages.get(member, 0)

        detail_percent_dict[member] = {
            'performance_percent': performance_percent,
            'wolf_percent': wolf_percent,
            'sherlock_percent': sherlock_percent,
            'total_percent': performance_percent + wolf_percent + sherlock_percent
        }

    # Create the final DataFrame
    member_stats = pd.DataFrame.from_dict(detail_percent_dict, orient='index')
    member_stats['پاداش'] = member_stats['total_percent'] * reward_amount / 100
    member_stats = member_stats.rename(columns={
        'total_percent': 'درصد کل',
        'performance_percent': 'درصد عملکرد',
        'wolf_percent': 'درصد گرگ',
        'sherlock_percent': 'درصد شرلوک'
    }).reset_index().rename(columns={'index': 'کارشناس'})

    return member_stats


# --- UI Display Functions ---

def display_team_metrics(df: pd.DataFrame, parameters: dict) -> float:
    """
    Displays the main team-wide KPI metrics (Target, Total Sales, Deal Count) and progress.

    Returns:
        The calculated total reward amount for the team.
    """
    st.subheader("📊 آمار کلی تیم")
    target = float(parameters.get('Target', 0))
    reward_percent = float(parameters.get('Reward percent', 0))
    
    # The division by 10 might be to convert Rials to Tomans or correct for an extra zero.
    total_sales = df['deal_value'].sum() / 10
    deals_count = df.shape[0]
    
    diff = total_sales - target
    reward_amount = max(0, diff) * reward_percent / 100

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 تارگت این ماه:", f'{target:,.0f} تومان')
    col2.metric("💵 مجموع فروش:", f"{total_sales:,.0f} تومان")
    col3.metric("🔢 تعداد معاملات:", deals_count)

    if reward_amount > 0:
        st.success(f"🏆 پاداش کل تیم: {reward_amount:,.0f} تومان")
    else:
        progress_percentage = (total_sales / target) * 100 if target > 0 else 0
        remaining = target - total_sales
        st.info(f"📈 {progress_percentage:.1f}% از تارگت محقق شده است.")
        st.warning(f"⏳ باقیمانده تا رسیدن به تارگت: {remaining:,.0f} تومان")
    
    return reward_amount

def display_daily_deals_chart(df: pd.DataFrame, title: str):
    """Displays a line chart of the number of deals per day."""
    st.subheader(title)
    if df.empty:
        st.info("داده‌ای برای نمایش نمودار وجود ندارد.")
        return
        
    deals_per_day = df.groupby('deal_created_date').size().reset_index(name='deals_count')
    deals_per_day['تاریخ شمسی'] = deals_per_day['deal_created_date'].apply(safe_to_jalali)

    y_max = deals_per_day['deals_count'].max()
    y_max_limit = 1 if pd.isna(y_max) else int(y_max * 1.15) + 1
    
    fig = px.line(
        deals_per_day, x='deal_created_date', y='deals_count',
        markers=True, labels={'deal_created_date': 'تاریخ', 'deals_count': 'تعداد معاملات'},
        hover_data=['تاریخ شمسی']
    )
    fig.update_layout(
        template='plotly_white', yaxis=dict(range=[0, y_max_limit])
    )
    st.plotly_chart(fig, use_container_width=True)

def display_member_details(df: pd.DataFrame, member_name: str):
    """
    Displays detailed stats for a single team member, including metrics,
    a daily deals chart, and a list of their recent deals.
    """
    st.subheader(f"👤 آمار عملکرد {member_name}")
    member_deals = df[df['deal_owner'] == member_name]

    if member_deals.empty:
        st.info(f"داده‌ای برای نمایش آمار {member_name} در این ماه وجود ندارد.")
        return

    # Calculate per-member metrics
    total_deals = len(member_deals)
    total_nights = member_deals['product_quantity'].sum()
    total_value = member_deals['deal_value'].sum()
    avg_nights = round(total_nights / total_deals, 1) if total_deals > 0 else 0
    avg_value = round(total_value / total_deals, 1) if total_deals > 0 else 0
    # Assuming deal_value is in Rials and we want to display millions of Tomans
    avg_value_million = (avg_value / 10 / 1_000_000)

    # Show metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric('تعداد کل معاملات', f"{total_deals:,}")
    col2.metric('میانگین تعداد شب', f"{avg_nights:,}")
    col3.metric('میانگین ارزش معامله (میلیون تومان)', f"{avg_value_million:,.1f}")

    # Display charts and data for the member
    display_daily_deals_chart(member_deals, f'📅 تعداد معاملات روزانه {member_name}')
    
    st.markdown(f"### 📋 لیست معاملات {member_name}")
    st.dataframe(member_deals, use_container_width=True, hide_index=True)


# --- Main App Function ---

def sales():
    """Main function to render the Sales team dashboard Streamlit page."""
    st.title("📊 داشبورد تیم Sales")

    # --- 1. Authentication and Initialization ---
    if not all(key in st.session_state for key in ['username', 'role', 'data', 'team', 'auth']):
        st.error("لطفا ابتدا وارد سیستم شوید")
        st.stop()

    role = st.session_state.role
    username = st.session_state.username
    name = st.session_state.name
    st.write(f"{name} عزیز خوش آمدی 😃")
    
    # Define team members
    team_members = [
        "پویا  ژیانی", "محمد آبساران/روز", "محمد آبساران/شب", "زینب فلاح نژاد", "پویا وزیری",
        "پوریا کیوانی", "بابک  مسعودی", "حسین  طاهری", "فرشته فرج نژاد", "حافظ قاسمی", "آرمین مربی"
    ]
    team_members_names = [
        "پویا  ژیانی", "محمد آبساران", "زینب فلاح نژاد", "پویا وزیری", "پوریا کیوانی",
        "بابک  مسعودی", "حسین  طاهری", "فرشته فرج نژاد", "حافظ قاسمی", "آرمین مربی"
    ]

    # --- 2. Data Loading and Pre-processing ---
    # Load data and filter for the sales team
    data = st.session_state['data']
    data = data[(data['deal_owner'].isin(team_members)) & (data['team'] == 'sales')].copy()
    data['deal_owner'] = data['deal_owner'].apply(normalize_owner)
    data['deal_created_date'] = pd.to_datetime(data['deal_created_date']).dt.date
    data['jalali_date'] = data['deal_created_date'].apply(safe_to_jalali)
    data['jalali_year_month'] = data['jalali_date'].apply(lambda d: f"{d.year}-{d.month:02d}")

    # --- 3. Month Selection Filter ---
    month_choose = st.selectbox(
        label='ماه مورد نظر را انتخاب کنید:',
        options=['این ماه', 'ماه پیش'],
        key='month_select_box'
    )
    today_jalali = jdatetime.date.today()
    if month_choose == 'ماه پیش':
        first_day_of_month = today_jalali.replace(day=1)
        last_month_jalali = first_day_of_month - jdatetime.timedelta(days=1)
        target_month_str = f"{last_month_jalali.year}-{last_month_jalali.month:02d}"
    else:
        target_month_str = f"{today_jalali.year}-{today_jalali.month:02d}"
        
    # Filter main DataFrame for the selected month and deal type
    df_filtered = data[
        (data['jalali_year_month'] == target_month_str) &
        (data['deal_type'] == 'New Sale')
    ]
    st.info(f'نمایش آمار برای ماه: {target_month_str}')

        # Calculate leaderboards and load parameters
    wolf_board, wolf_first_persons = cal_wolfs(data.copy(), target_month_str)
    sherlock_board = load_sherlock_data(target_month_str)
    parametrs_df = load_sheet_uncache('Sales team parameters')

    # Ensure parameters are loaded into a dictionary with default values
    default_params = {
        "Target": 0, "Reward percent": 0, "Wolf1": 0, "Wolf2": 0,
        "Sherlock": 0, "Performance": 0
    }
    parametrs = parametrs_df.iloc[0].to_dict() if isinstance(parametrs_df, pd.DataFrame) and not parametrs_df.empty else default_params

    # --- 4. Role-Based UI Rendering ---
    is_manager = role in ["admin", "manager"]
    
    if is_manager:
        # Manager/Admin View with Tabs
        tabs = st.tabs(['صفحه اصلی', 'گرگ وال استریت', 'شرلوک', 'تنظیمات'])
        
        with tabs[0]: # Main Tab
            reward_amount = display_team_metrics(df_filtered, parametrs)
            display_daily_deals_chart(df_filtered, 'تعداد کل معاملات روزانه تیم')
            st.divider()

            selected_member = st.selectbox("انتخاب کارشناس:", team_members_names)
            display_member_details(df_filtered, selected_member)

            if reward_amount > 0:
                reward_details_df = calculate_reward_details(reward_amount, wolf_board, sherlock_board, parametrs, team_members_names)
                member_reward_row = reward_details_df[reward_details_df['کارشناس'] == selected_member]
                st.markdown(f"### 💰 میزان پاداش {selected_member}")
                st.dataframe(member_reward_row, use_container_width=True, hide_index=True)

        with tabs[1]: # Wolf Tab
            st.markdown("### 🐺 جدول امتیاز گرگ‌های وال‌استریت (ماه جاری)")
            st.dataframe(wolf_board.rename(columns={"deal_owner": "شخص", "score": "امتیاز"}), use_container_width=True, hide_index=True)
            
            st.markdown("### 👑 تعداد دفعات اول شدن هر شخص (ماه جاری)")
            if wolf_first_persons:
                wolf_first_df = pd.DataFrame(list(wolf_first_persons.items()), columns=["مالک", "تعداد اول شدن"]).sort_values("تعداد اول شدن", ascending=False)
                st.dataframe(wolf_first_df, use_container_width=True, hide_index=True)
            else:
                st.info("هنوز کسی در این ماه اول نشده است.")

        with tabs[2]: # Sherlock Tab
            st.markdown("### 🕵️ جدول امتیاز شرلوک (ماه جاری)")
            if sherlock_board is not None and not sherlock_board.empty:
                st.dataframe(sherlock_board.rename(columns={"deal_owner": "شخص", "score": "امتیاز"}), use_container_width=True, hide_index=True)
            else:
                st.info("هنوز امتیازی در بخش شرلوک ثبت نشده است.")
        
        with tabs[3]: # Settings Tab
            # The settings form remains complex and is kept here directly.
            with st.form("sales_params_form", clear_on_submit=False):
                st.markdown("### 🎯 تنظیم پارامترهای اصلی تیم فروش")
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.container():
                            st.markdown('<div class="param-card">', unsafe_allow_html=True)
                            st.markdown('<div class="param-title">🎯 تارگت این ماه</div>', unsafe_allow_html=True)
                            target = st.number_input(
                                label=" ", key='target_number',
                                step=1.0, format="%f", value=float(parametrs.get('Target', 0))
                            )
                            st.markdown('<div class="param-title">💰 درصد پاداش این ماه</div>', unsafe_allow_html=True)
                            reward_percent = st.number_input(
                                label=" ", key='reward_percent', format="%f",
                                step=1.0, min_value=0.0, max_value=100.0, value=float(parametrs.get('Reward percent', 0))
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        with st.container():
                            st.markdown('<div class="param-card">', unsafe_allow_html=True)
                            st.markdown('<div class="param-title">🐺 درصد گرگ اول</div>', unsafe_allow_html=True)
                            wolf1_percent = st.number_input(
                                label=" ", key='wolf1_percent', format="%f",
                                step=1.0, min_value=0.0, max_value=100.0, value=float(parametrs.get('Wolf1', 0))
                            )
                            st.markdown('<div class="param-title">🐺 درصد گرگ دوم</div>', unsafe_allow_html=True)
                            wolf2_percent = st.number_input(
                                label=" ", key='wolf2_percent', format="%f",
                                step=1.0, min_value=0.0, max_value=100.0, value=float(parametrs.get('Wolf2', 0))
                            )
                            st.markdown('<div class="param-title">🕵️ درصد شرلوک</div>', unsafe_allow_html=True)
                            sherlock_percent = st.number_input(
                                label=" ", key='sherlock_percent', format="%f",
                                step=1.0, min_value=0.0, max_value=100.0, value=float(parametrs.get('Sherlock', 0))
                            )
                            st.markdown('<div class="param-title">📈 درصد عملکرد</div>', unsafe_allow_html=True)
                            performance_percent = st.number_input(
                                label=" ", key='performance_percent', format="%f",
                                step=1.0, min_value=0.0, max_value=100.0, value=float(parametrs.get('Performance', 0))
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                member_percents = {}
                with st.expander("👥 درصد عملکرد اعضای تیم", expanded=True):
                    st.markdown("""درصد عملکرد هر عضو تیم را وارد کنید:""", unsafe_allow_html=True)
                    n = len(team_members_names)
                    n_cols = 4 if n >= 8 else 2  # More columns for larger teams
                    rows = [team_members_names[i:i+n_cols] for i in range(0, n, n_cols)]
                    for row in rows:
                        cols = st.columns(len(row))
                        for idx, member in enumerate(row):
                            with cols[idx]:
                                member_percent = st.number_input(
                                    label=f"{member}",
                                    key=f'{member}_percent',
                                    format="%f",
                                    step=1.0,
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=float(parametrs.get(f'{member}_percent', 0))
                                )
                                member_percents[member] = member_percent
                    st.caption("🔢 جمع درصد عملکرد اعضا باید دقیقا 100 باشد.")

                submitted = st.form_submit_button('تنظیم مجدد')
                if submitted:
                    # Validate that main percents sum to 100
                    if sum([wolf1_percent, wolf2_percent, sherlock_percent, performance_percent]) != 100:
                        st.warning("جمع درصد های اصلی (گرگ‌ها، شرلوک، عملکرد) باید دقیقا 100 باشد!")
                    # Validate that member percents sum to 100 (allowing for float rounding)
                    elif abs(sum(member_percents.values()) - 100) > 0.01:
                        st.warning("جمع درصد عملکرد اعضا باید دقیقا 100 باشد!")
                    else:
                        # Combine main parameters and member percents into one dictionary
                        param_dict = {
                            "Target": target,
                            "Reward percent": reward_percent,
                            "Wolf1": wolf1_percent,
                            "Wolf2": wolf2_percent,
                            "Sherlock": sherlock_percent,
                            "Performance": performance_percent
                        }
                        for member, percent in member_percents.items():
                            param_dict[f"{member}_percent"] = percent

                        df = pd.DataFrame([param_dict])
                        success = write_df_to_sheet(df, sheet_name='Sales team parameters')
                        if success:
                            st.info("پارامترها با موفقیت آپدیت شد.")
                        else:
                            st.info("خطا در آپدیت پارامترها!!!")


    else:
        # Regular User View (No Tabs)
        reward_amount = display_team_metrics(df_filtered, parametrs)
        st.divider()
        
        display_member_details(df_filtered, username)
        
        if reward_amount > 0:
            reward_details_df = calculate_reward_details(reward_amount, wolf_board, sherlock_board, parametrs, team_members_names)
            user_reward_row = reward_details_df[reward_details_df['کارشناس'] == username]
            st.markdown("### 💰 میزان پاداش شما")
            st.dataframe(user_reward_row, use_container_width=True, hide_index=True)