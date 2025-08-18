import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
import numpy as np
from utils.write_sheet import write_df_to_sheet
from utils.load_sheet import load_sheet, load_sheet_uncache

def normalize_owner(owner):
    # Merge "محمد آبساران/روز" and "محمد آبساران/شب" as one person
    if owner in ["محمد آبساران/روز", "محمد آبساران/شب"]:
        return "محمد آبساران"
    return owner

# calculate wolfs
def cal_wolfs(df: pd.DataFrame) -> pd.DataFrame:
    """functions for calculate wolfs - filter for current Jalali month, with extra reward for the person that is first most often"""
    # Convert deal_created_date to datetime
    df['deal_created_date'] = pd.to_datetime(df['deal_created_date'])

    df['deal_owner'] = df['deal_owner'].apply(normalize_owner)

    # Filter for current Jalali month
    df['jalali_date'] = df['deal_created_date'].apply(lambda x: jdatetime.date.fromgregorian(date=x.date()))
    # Get current Jalali year and month
    today_jalali = jdatetime.date.today()
    current_jalali_year = today_jalali.year
    current_jalali_month = today_jalali.month
    df = df[ 
            (df['jalali_date'].apply(lambda d: d.year) == current_jalali_year) & 
            (df['jalali_date'].apply(lambda d: d.month) == current_jalali_month) &
            (df['deal_type'] == "New Sale")
            ]
    
    df['Date'] = df['deal_created_date'].dt.date

    # Sum deal_value for each person per day
    daily_sum = df.groupby(['Date', 'deal_owner'])['deal_value'].sum().reset_index()
    scores = {}
    first_place_counts = {}

    all_owners = df['deal_owner'].unique()

    # For each day, rank and assign scores
    for day, group in daily_sum.groupby('Date'):
        group_sorted = group.sort_values('deal_value', ascending=False).reset_index(drop=True)
        owners_today = group_sorted['deal_owner'].tolist()
        for idx, row in group_sorted.iterrows():
            owner = row['deal_owner']
            # Scoring: 1st=3, 2nd=2, 3rd=1, last=-1, others=0
            if idx == 0:
                score = 3
                first_place_counts[owner] = first_place_counts.get(owner, 0) + 1
            elif idx == 1:
                score = 2
            elif idx == 2:
                score = 1
            elif idx == len(group_sorted) - 1:
                score = -1
            else:
                score = 0
            scores[owner] = scores.get(owner, 0) + score

    # Build final list of owners and their scores
    wolf_board = [{'deal_owner': owner, 'score': scores.get(owner, 0)} for owner in all_owners]
    
    return pd.DataFrame(wolf_board).sort_values('score', ascending=False).reset_index(drop=True), first_place_counts

# load_sherlock data
def load_sherlock_data() -> pd.DataFrame:
    """"""
    sherlock_df = load_sheet('Sherlock')
    sherlock_df['Date'] = pd.to_datetime(sherlock_df['Date'])

    # Add Jalali date column
    sherlock_df['jalali_date'] = sherlock_df['Date'].apply(lambda x: jdatetime.date.fromgregorian(date=x.date()))
    today_jalali = jdatetime.date.today()
    current_jalali_year = today_jalali.year
    current_jalali_month = today_jalali.month

    # Filter for current Jalali month
    sherlock_df = sherlock_df[
        (sherlock_df['jalali_date'].apply(lambda d: d.year) == current_jalali_year) &
        (sherlock_df['jalali_date'].apply(lambda d: d.month) == current_jalali_month)
    ]

    # Prepare scores
    scores = {}

    # For each day, assign points: 1st=10, 2nd=5, 3rd=-3
    for day, group in sherlock_df.groupby('Date'):
        # Get persons for the day
        if group['First person'].values[0] == '':
            continue
        first = group['First person'].dropna().values
        second = group['Second person'].dropna().values
        third = group['Last person'].dropna().values

        # If multiple rows per day, count all
        for p in first:
            scores[p] = scores.get(p, 0) + 10
        for p in second:
            scores[p] = scores.get(p, 0) + 5
        for p in third:
            scores[p] = scores.get(p, 0) - 3
    
    # Map short/variant names to full names
    name_map = {
        'حافظ': 'حافظ قاسمی',
        'پویا(وزیری)': 'پویا وزیری',
        'محمد (آبساران)': 'محمد آبساران',
        'بابک': 'بابک مسعودی',
        'حسین (طاهری)': 'حسین طاهری',
        'پویا (شب)': 'پویا ژیانی',
        'پوریا': 'پوریا کیوانی',
        'فرشته': 'فرشته فرج نژاد',
        'زینب': 'زینب فلاح نژاد',
        'مربی': 'آرمین مربی',
        '': None,
    }

    # Only keep mapped names that are in the final list
    valid_names = [
        'حسین طاهری', 'پوریا کیوانی', 'پویا ژیانی', 'فرشته فرج نژاد',
        'محمد آبساران', 'حافظ قاسمی', 'بابک مسعودی', 'زینب فلاح نژاد', 'پویا وزیری', 'آرمین مربی'
    ]

    # Remap scores to full names, sum if multiple map to same
    mapped_scores = {}
    for person, score in scores.items():
        mapped = name_map.get(person, person)
        if mapped in valid_names:
            mapped_scores[mapped] = mapped_scores.get(mapped, 0) + score

    result = pd.DataFrame([
        {'deal_owner': person, 'score': score}
        for person, score in mapped_scores.items()
    ]).sort_values('score', ascending=False).reset_index(drop=True)

    return result

def sales():
    """sales team dashboard with optimized metrics and visualizations"""
    st.title("📊 داشبورد تیم Sales")
    
    if not all(key in st.session_state for key in ['username', 'role', 'data', 'team', 'auth']):
        st.error("لطفا ابتدا وارد سیستم شوید")
        return

    # Initialize data and variables
    role = st.session_state.role
    username = st.session_state.username
    name = st.session_state.name

    st.write(f"{name}  عزیز خوش آمدی😃")    

    team_members = [
            "پویا  ژیانی", "محمد آبساران/روز", "محمد آبساران/شب", "زینب فلاح نژاد", "پویا وزیری",
            "پوریا کیوانی", "بابک  مسعودی", "حسین  طاهری", "فرشته فرج نژاد", "حافظ قاسمی", "آرمین مربی"
        ]
    team_members_names = [
            "پویا  ژیانی","محمد آبساران", "زینب فلاح نژاد", "پویا وزیری", "پوریا کیوانی",
            "بابک  مسعودی", "حسین  طاهری", "فرشته فرج نژاد", "حافظ قاسمی", "آرمین مربی"
        ]

    if role in ["admin", "manager"]:
        data = st.session_state['data']  
        data = data[data['deal_owner'].isin(team_members)]
        wolf_board, wolf_first_persons = cal_wolfs(data.copy())
        sherlock_board = load_sherlock_data()
        parametrs_df = load_sheet_uncache('Sales team parameters')

        # Ensure parametrs is a dict/Series of scalars, not a DataFrame/Series of length > 1
        if isinstance(parametrs_df, pd.DataFrame):
            if not parametrs_df.empty:
                parametrs = parametrs_df.iloc[0].to_dict()
            else:
                parametrs = {
                    "Target": 0,
                    "Reward percent": 0,
                    "Wolf1": 0,
                    "Wolf2": 0,
                    "Sherlock": 0,
                    "Performance": 0
                }
        elif isinstance(parametrs_df, pd.Series):
            parametrs = parametrs_df.to_dict()
        else:
            parametrs = {
                "Target": 0,
                "Reward percent": 0,
                "Wolf1": 0,
                "Wolf2": 0,
                "Sherlock": 0,
                "Performance": 0
            }

        tabs = st.tabs(['صفحه اصلی', 'گرگ وال استریت', 'شرلوک', 'تنظیمات'])
        with tabs[0]:
            df = data.copy()

            df['deal_owner'] = df['deal_owner'].apply(normalize_owner)

            # Ensure 'deal_created_date' is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['deal_created_date']):
                df['deal_created_date'] = pd.to_datetime(df['deal_created_date'], errors='coerce')
            # Add Jalali date column
            df['jalali_date'] = df['deal_created_date'].apply(lambda x: jdatetime.date.fromgregorian(date=x.date()) if pd.notnull(x) else None)
            # Get current Jalali year and month
            today_jalali = jdatetime.date.today()
            current_jalali_year = today_jalali.year
            current_jalali_month = today_jalali.month
            # Filter for current Jalali month and year and new sales
            df = df[
                (df['jalali_date'].apply(lambda d: d.year if d else None) == current_jalali_year) &
                (df['jalali_date'].apply(lambda d: d.month if d else None) == current_jalali_month) &
                (df['deal_type'] == "New Sale")
            ]
            # Ensure DealValue column exists and is numeric
            if 'DealValue' not in df.columns:
                if 'deal_value' in df.columns:
                    df['DealValue'] = df['deal_value']
                else:
                    df['DealValue'] = 0
            df['DealValue'] = pd.to_numeric(df['DealValue'], errors='coerce').fillna(0)

            # Calculate stats
            target = float(parametrs.get('Target', 0))
            reward_percent = float(parametrs.get('Reward percent', 0))
            total_sales = df['DealValue'].sum() / 10
            deals_count = df.shape[0]
            diff = total_sales - target
            reward_amount = max(0, diff) * reward_percent / 100

            st.metric("🎯 تارگت این ماه:", f'{target:,.0f}')
            st.metric(f"💵 مجموع فروش تا این لحظه:", f"{total_sales:,.0f}")
            st.metric("🔢 تعداد معامله‌ها: ", deals_count)

            if reward_amount > 0 :
                st.write(f"🏆 میزان پاداش: {reward_amount:,.0f}")
            else:
                st.metric(f"📈 تا رسیدن به پاداش:", f"{diff:,.0f}")


            # deals per day (using plotly)
            if not df.empty:
                df['date_only'] = df['deal_created_date'].dt.date
                deals_per_day = df.groupby('date_only').size().reset_index(name='deals_count')
                st.subheader('تعداد معاملات روزانه (این ماه)')
                # Calculate y-axis range
                y_min = 0
                y_max = deals_per_day['deals_count'].max()
                if pd.isna(y_max):
                    y_max = 1
                else:
                    y_max = int(y_max * 1.15) + 1
                fig = px.line(
                    deals_per_day,
                    x='date_only',
                    y='deals_count',
                    markers=True,
                    labels={'date_only': 'تاریخ', 'deals_count': 'تعداد معاملات'},
                    title=''
                )
                fig.update_layout(
                    xaxis_title='تاریخ',
                    yaxis_title='تعداد معاملات',
                    template='plotly_white',
                    yaxis=dict(range=[y_min, y_max])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("داده‌ای برای نمایش نمودار معاملات این ماه وجود ندارد.")

            # team members
            st.subheader("آمار اعضای تیم (این ماه)")
            if not df.empty:
                member_stats = (
                    df.groupby('deal_owner')
                    .agg(
                        total_sales=('DealValue', 'sum'),
                        deals_count=('DealValue', 'count')
                    )
                    .reindex(team_members_names, fill_value=0)
                )
                if reward_amount > 0 :
                    # Calculate sum_percent for each member and add as a new column to member_stats
                    sum_percent_dict = {}
                    detail_percent_dict = {}
                    for member in team_members_names:
                        performance_percent = parametrs.get(f'{member}_percent', 0) * parametrs.get('Performance', 0)/100
                        wolf_percent = 0
                        sherlock_percent = 0

                        # Wolf percent
                        if not wolf_board.empty:
                            wolf_sorted = wolf_board.sort_values("score", ascending=False).reset_index(drop=True)
                            # Find all with max score (first place)
                            if len(wolf_sorted) > 0:
                                max_score = wolf_sorted.iloc[0]["score"]
                                wolf_first_group = wolf_sorted[wolf_sorted["score"] == max_score]
                                wolf_first_names = wolf_first_group["deal_owner"].tolist()
                                wolf1_percent = parametrs.get('Wolf1', 0)
                                if member in wolf_first_names and len(wolf_first_names) != 1:
                                    wolf_percent = (wolf1_percent + wolf2_percent) / len(wolf_first_names) if len(wolf_first_names) > 0 else 0
                                elif member in wolf_first_names and len(wolf_first_names) == 1: 
                                    wolf_percent = wolf1_percent / len(wolf_first_names) if len(wolf_first_names) > 0 else 0
                                else:
                                    if len(wolf_first_names) == 1:
                                        second_score_group = wolf_sorted[wolf_sorted["score"] < max_score]
                                        if not second_score_group.empty:
                                            second_max_score = second_score_group.iloc[0]["score"]
                                            wolf_second_group = wolf_sorted[wolf_sorted["score"] == second_max_score]
                                            wolf_second_names = wolf_second_group["deal_owner"].tolist()
                                            wolf2_percent = parametrs.get('Wolf2', 0)
                                            if member in wolf_second_names:
                                                wolf_percent = wolf2_percent / len(wolf_second_names) if len(wolf_second_names) > 0 else 0
                        # Sherlock percent
                        if sherlock_board is not None and not sherlock_board.empty:
                            sherlock_sorted = sherlock_board.sort_values("score", ascending=False).reset_index(drop=True)
                            if len(sherlock_sorted) > 0:
                                max_score = sherlock_sorted.iloc[0]["score"]
                                sherlock_first_group = sherlock_sorted[sherlock_sorted["score"] == max_score]
                                sherlock_first_names = sherlock_first_group["deal_owner"].tolist()
                                sherlock_percent_value = parametrs.get('Sherlock', 0)
                                if member in sherlock_first_names:
                                    sherlock_percent = sherlock_percent_value / len(sherlock_first_names) if len(sherlock_first_names) > 0 else 0
                        
                        # Save percents
                        sum_percent = sherlock_percent + wolf_percent + performance_percent
                        sum_percent_dict[member] = sum_percent

                        detail_percent_dict[member] = {
                            'performance_percent': performance_percent,
                            'wolf_percent': wolf_percent,
                            'sherlock_percent': sherlock_percent,
                        }

                    member_stats['درصد'] = member_stats.index.map(lambda x: sum_percent_dict.get(x, 0))
                    member_stats['پاداش'] = member_stats.index.map(lambda x: int(sum_percent_dict.get(x, 0)*reward_amount/100))
                    member_stats['درصد عملکرد'] = member_stats.index.map(lambda x: detail_percent_dict.get(x, {}).get('performance_percent', 0))
                    member_stats['درصد گرگ'] = member_stats.index.map(lambda x: detail_percent_dict.get(x, {}).get('wolf_percent', 0))
                    member_stats['درصد شرلوک'] = member_stats.index.map(lambda x: detail_percent_dict.get(x, {}).get('sherlock_percent', 0))

                    member_stats = member_stats.reset_index().rename(columns={
                        'deal_owner': 'کارشناس',
                        'total_sales': 'مجموع فروش',
                        'deals_count': 'تعداد معامله'
                    })

                # Use a selectbox to show each person
                selected_member = st.selectbox(
                    "انتخاب کارشناس:",
                    team_members_names,
                    index=1
                )
                member_row = member_stats[member_stats['کارشناس'] == selected_member]
                st.markdown(f"### آمار {selected_member}")
                st.dataframe(member_row, use_container_width=True)

                member_deals = df[df['deal_owner']==selected_member]
                st.markdown(f"### معامله های {selected_member}")
                st.dataframe(member_deals, use_container_width=True)
                st.metric('میانگین تعداد شب :', round(member_deals['product_quantity'].sum()/len(member_deals), 1))
            
            else:
                st.info("داده‌ای برای نمایش آمار اعضای تیم وجود ندارد.")


        with tabs[1]: 
            st.markdown("### 🐺 جدول امتیاز گرگ‌های وال‌استریت")
            st.dataframe(
                wolf_board.rename(columns={"deal_owner": "شخص", "score": "امتیاز"}),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 👑 تعداد دفعات اول شدن هر شخص")
            # Convert first_place_counts dict to DataFrame for better display
            if wolf_first_persons:
                wolf_first_df = (
                    pd.DataFrame(list(wolf_first_persons.items()), columns=["مالک معامله", "تعداد اول شدن"])
                    .sort_values("تعداد اول شدن", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(wolf_first_df, use_container_width=True, hide_index=True)
            else:
                st.info("هنوز کسی اول نشده است.")

        with tabs[2]:
            st.markdown("### 🕵️ جدول امتیاز شرلوک")
            if sherlock_board is not None and not sherlock_board.empty:
                st.dataframe(
                    sherlock_board.rename(columns={"deal_owner": "شخص", "score": "امتیاز"}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("هنوز کسی امتیاز شرلوک نگرفته است.")

        # setting tabs for set new values for parameters
        with tabs[3]:
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
                    if sum([wolf1_percent, wolf2_percent, sherlock_percent, performance_percent]) != 100:
                        st.warning("جمع درصد های اصلی (گرگ‌ها، شرلوک، عملکرد) باید دقیقا 100 باشد!")
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
                        # Add member percents to the dictionary
                        for member, percent in member_percents.items():
                            param_dict[f"{member}_percent"] = percent

                        df = pd.DataFrame([param_dict])
                        success = write_df_to_sheet(df, sheet_name='Sales team parameters')
                        if success:
                            st.info("پارامترها با موفقیت آپدیت شد.")
                        else:
                            st.info("خطا در آپدیت پارامترها!!!")

    else:
        # users
        pass
