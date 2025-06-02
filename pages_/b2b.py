import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import jdatetime

def b2b():
    """B2B team dashboard."""
    st.title("📊 داشبورد تیم B2B ")
    
    if 'username' in st.session_state and 'role' in st.session_state \
        and 'data' in st.session_state and 'team' in st.session_state and 'auth' in st.session_state:
        role = st.session_state.role
        username = st.session_state.username
        st.write(f"{username} ({role}) عزیز خوش آمدی😃")
        
        filter_data = st.session_state.data.copy()
        filter_data = filter_data[filter_data['team'] ==  'b2b']
        filter_data['deal_done_date'] = pd.to_datetime(filter_data['deal_done_date'])
        # convert to rial
        filter_data['deal_value'] = pd.to_numeric(filter_data['deal_value'], errors='coerce') / 10

        # Calculate current week's start date (Saturday)
        today = datetime.today().date()
        current_week_start = today - timedelta(days=(today.weekday() + 2)%7)  # start from Saturday


        # Create weekly ranges for last 4 weeks
        week_ranges = [(current_week_start - timedelta(weeks=i), 
            current_week_start - timedelta(weeks=i-1) - timedelta(days=1)) 
            for i in range(4, 0, -1)]

        # Calculate weekly metrics for team
        weekly_counts = []
        weekly_values = []
        
        for start, end in week_ranges:
            # Team metrics
            mask = (filter_data['deal_done_date'].dt.date >= start) & \
            (filter_data['deal_done_date'].dt.date <= end)
            weekly_counts.append(filter_data[mask].shape[0])
            weekly_values.append(filter_data[mask]['deal_value'].sum())
            

        # Find max weeks for team
        max_count_week_index = weekly_counts.index(max(weekly_counts))
        max_count_week = 4 - max_count_week_index
        max_value_week_index = weekly_values.index(max(weekly_values))
        max_value_week = 4 - max_value_week_index

        # Calculate this week's metrics for team
        this_week_mask = (filter_data['deal_done_date'].dt.date >= current_week_start) & \
                        (filter_data['deal_done_date'].dt.date <= today)
        this_week_count = filter_data[this_week_mask].shape[0]
        this_week_value = filter_data[this_week_mask]['deal_value'].sum()

        # Display metrics
        st.subheader("آمار تیم")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("تعداد فروش روزانه تیم", f"{filter_data[filter_data['deal_done_date'].dt.date == today].shape[0]:,.0f}")
            st.metric("بیشترین تعداد فروش هفتگی تیم", 
            f"{max_count_week}هفته قبل: {max(weekly_counts):,.0f}", delta=f"{jdatetime.date.fromgregorian(date=week_ranges[max_count_week_index][0]).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=week_ranges[max_count_week_index][1]).strftime('%Y/%m/%d')}",
            )
            st.metric("تعداد فروش این هفته", f"{this_week_count:,.0f}")

        with col2:
            st.metric("مقدار فروش روزانه تیم", f"{filter_data[filter_data['deal_done_date'].dt.date == today]['deal_value'].sum():,.0f} تومان")
            st.metric("بیشترین مقدار فروش هفتگی تیم", 
            f"{max_value_week} هفته قبل : {max(weekly_values):,.0f} تومان",  delta=f"{jdatetime.date.fromgregorian(date=week_ranges[max_value_week_index][0]).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=week_ranges[max_value_week_index][1]).strftime('%Y/%m/%d')}",
            )
            st.metric("مقدار فروش این هفته", f"{this_week_value:,.0f} تومان")


        # Team charts
        col1, col2 = st.columns(2)
        
        with col1:
            df_counts = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges],
            'تعداد': weekly_counts,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} تا {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges]
            })
            fig_counts = px.bar(df_counts, x='هفته', y='تعداد',
              hover_data=['بازه زمانی'], title='تعداد فروش هفتگی تیم')
            fig_counts.update_layout(title_x=0.5, title_font=dict(size=20))
            fig_counts.update_layout(xaxis_title="بازه زمانی ", yaxis_title="تعداد")
            fig_counts.update_traces(marker_color=['#90EE90' if i == max_count_week_index else 'gray' for i in range(len(weekly_counts))])
            st.plotly_chart(fig_counts)

        with col2:
            df_values = pd.DataFrame({
            'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges],
            'مقدار': weekly_values,
            'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} تا {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges]
            })
            fig_values = px.bar(df_values, x='هفته', y='مقدار',
              hover_data=['بازه زمانی'], title='مقدار فروش هفتگی تیم')
            fig_values.update_layout(xaxis_title="بازه زمانی ", yaxis_title="مقدار (تومان)")
            fig_values.update_layout(title_x=0.5, title_font=dict(size=20))
            fig_values.update_traces(marker_color=['#90EE90' if i == max_value_week_index else 'gray' for i in range(len(weekly_values))])
            st.plotly_chart(fig_values)

        # Member charts
        if  role == 'member' or role == 'manager':
            member_data = filter_data[filter_data['deal_owner'] == username]

            # Create weekly ranges for last 4 weeks
            week_ranges = [(current_week_start - timedelta(weeks=i), 
            current_week_start - timedelta(weeks=i-1) - timedelta(days=1)) 
            for i in range(4, 0, -1)]

            # Calculate weekly metrics for member
            member_weekly_counts = []
            member_weekly_values = []
            
            for start, end in week_ranges:
            # Member metrics
                member_mask = (member_data['deal_done_date'].dt.date >= start) & \
                (member_data['deal_done_date'].dt.date <= end)
                member_weekly_counts.append(member_data[member_mask].shape[0])
                member_weekly_values.append(member_data[member_mask]['deal_value'].sum())

            
            # Find max weeks for member
            max_member_count_week_index = member_weekly_counts.index(max(member_weekly_counts))
            max_member_count_week = 4 - max_member_count_week_index
            max_member_value_week_index = member_weekly_values.index(max(member_weekly_values))
            max_member_value_week = 4 - max_member_value_week_index

            # Calculate this week's metrics for member
            member_this_week_mask = (member_data['deal_done_date'].dt.date >= current_week_start) & \
                    (member_data['deal_done_date'].dt.date <= today)
            member_this_week_count = member_data[member_this_week_mask].shape[0]
            member_this_week_value = member_data[member_this_week_mask]['deal_value'].sum()


            st.subheader("آمار شما")
            col3, col4 = st.columns(2)
            
            with col3:
                st.metric("تعداد فروش روزانه شما", f"{member_data[member_data['deal_done_date'].dt.date == today].shape[0]:,.0f}")
                st.metric("بیشترین تعداد فروش هفتگی شما", 
                f"{max_member_count_week}هفته قبل: {max(member_weekly_counts):,.0f}", delta=f"{jdatetime.date.fromgregorian(date=week_ranges[max_member_count_week_index][0]).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=week_ranges[max_member_count_week_index][1]).strftime('%Y/%m/%d')}",
            )
                st.metric("تعداد فروش این هفته", f"{member_this_week_count:,.0f}")

            with col4:
                st.metric("مقدار فروش روزانه شما", f"{member_data[member_data['deal_done_date'].dt.date == today]['deal_value'].sum():,.0f} تومان")
                st.metric("بیشترین مقدار فروش هفتگی شما", 
                f"{max_member_value_week} هفته قبل : {max(member_weekly_values):,.0f} تومان", delta=f"{jdatetime.date.fromgregorian(date=week_ranges[max_member_value_week_index][0]).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=week_ranges[max_member_value_week_index][1]).strftime('%Y/%m/%d')}")
                st.metric("مقدار فروش این هفته", f"{member_this_week_value:,.0f} تومان")
                
            
            col3, col4 = st.columns(2)
            with col3:
                df_member_counts = pd.DataFrame({
                'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges],
                'تعداد': member_weekly_counts,
                'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} تا {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges]
                })
                fig_member_counts = px.bar(df_member_counts, x='هفته', y='تعداد',
                    hover_data=['بازه زمانی'], title='تعداد فروش هفتگی شما')
                fig_member_counts.update_layout(xaxis_title="بازه زمانی ", yaxis_title="تعداد", title_font=dict(size=20))
                fig_member_counts.update_layout(title_x=0.5)
                fig_member_counts.update_traces(marker_color=['#90EE90' if i == max_member_count_week_index else 'gray' for i in range(len(member_weekly_counts))])
                st.plotly_chart(fig_member_counts, key=f"counts_{username}")

            with col4:
                df_member_values = pd.DataFrame({
                'هفته':  [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges],
                'مقدار': member_weekly_values,
                'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} تا {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges]
                })
                fig_member_values = px.bar(df_member_values, x='هفته', y='مقدار',
                    hover_data=['بازه زمانی'], title='مقدار فروش هفتگی شما')
                fig_member_values.update_layout(title_x=0.5, title_font=dict(size=20))
                fig_member_values.update_layout(xaxis_title="بازه زمانی ", yaxis_title="مقدار (تومان)")
                fig_member_values.update_traces(marker_color=['#90EE90' if i == max_member_value_week_index else 'gray' for i in range(len(member_weekly_values))])
                st.plotly_chart(fig_member_values, key=f"values_{username}")
                
            # target (if they got to the max of the four week they will got 5 percent reward (0.05 * (this weelk - terget)))
            target = max(member_weekly_values) 
            reward_percentage = 0.05
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎁تارگت پاداش: ",  f"{target:,.0f} تومان")
                if this_week_value > target:
                    reward = reward_percentage * (this_week_value - target)
                    st.markdown(f"<span style='font-size: larger;'>🎉 میزان پاداش:   {reward:,.0f} </span>", unsafe_allow_html=True)


        if role == 'manager' or role == 'admin':
            # team members 
            user_list = st.secrets['user_lists']['b2b'].copy() 
            roles = st.secrets['roles']
            if username in user_list:
                user_list.remove(username)
            for user in user_list:
                if roles[user] == 'admin':
                    user_list.remove(user)

            for member in user_list:
                member_data = filter_data[filter_data['deal_owner'] == member]
                # Calculate member weekly metrics
                member_weekly_counts = []
                member_weekly_values = []
                for start, end in week_ranges:
                    member_mask = (member_data['deal_done_date'].dt.date >= start) & (member_data['deal_done_date'].dt.date <= end)
                    member_weekly_counts.append(member_data[member_mask].shape[0])
                    member_weekly_values.append(member_data[member_mask]['deal_value'].sum())
                
                # Find max weeks for member
                max_member_count_week_index = member_weekly_counts.index(max(member_weekly_counts)) if member_weekly_counts else None
                max_member_count_week = 4 - max_member_count_week_index if max_member_count_week_index is not None else None
                max_member_value_week_index = member_weekly_values.index(max(member_weekly_values)) if member_weekly_values else None
                max_member_value_week = 4 - max_member_value_week_index if max_member_value_week_index is not None else None

                # Calculate this week's metrics for member
                member_this_week_mask = (member_data['deal_done_date'].dt.date >= current_week_start) & \
                            (member_data['deal_done_date'].dt.date <= today)
                member_this_week_count = member_data[member_this_week_mask].shape[0]
                member_this_week_value = member_data[member_this_week_mask]['deal_value'].sum()
                
                st.subheader(f"آمار فرد: {member}")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.metric("تعداد فروش روزانه", f"{member_data[member_data['deal_done_date'].dt.date == today].shape[0]:,.0f}")
                    st.metric("بیشترین تعداد فروش هفتگی", 
                    f"{max_member_count_week} هفته قبل: {max(member_weekly_counts) if member_weekly_counts else 0:,.0f}", delta=f"{jdatetime.date.fromgregorian(date=week_ranges[max_member_count_week_index][0]).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=week_ranges[max_member_count_week_index][1]).strftime('%Y/%m/%d')}" if max_member_count_week_index is not None else None)
                    st.metric("تعداد فروش این هفته", f"{member_this_week_count:,.0f}")

                with col4:
                    st.metric("مقدار فروش روزانه شما", f"{member_data[member_data['deal_done_date'].dt.date == today]['deal_value'].sum():,.0f} تومان")
                    st.metric("بیشترین مقدار فروش هفتگی شما", 
                    f"{max_member_value_week} هفته قبل : {max(member_weekly_values):,.0f} تومان", delta=f"{jdatetime.date.fromgregorian(date=week_ranges[max_member_value_week_index][0]).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=week_ranges[max_member_value_week_index][1]).strftime('%Y/%m/%d')}" if max_member_value_week_index is not None else None,)
                    st.metric("مقدار فروش این هفته", f"{member_this_week_value:,.0f} تومان")
                
                # charts
                col3, col4 = st.columns(2)
                
                with col3:
                    df_member_counts = pd.DataFrame({
                    'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges],
                    'تعداد': member_weekly_counts,
                    'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} تا {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges]
                    })
                    fig_member_counts = px.bar(df_member_counts, x='هفته', y='تعداد',
                    hover_data=['بازه زمانی'],  title=f'تعداد فروش هفتگی {member}')
                    fig_member_counts.update_layout(xaxis_title="بازه زمانی ", yaxis_title="تعداد", title_font=dict(size=20))
                    fig_member_counts.update_traces(marker_color=['#90EE90' if max_member_count_week_index is not None and i == max_member_count_week_index else 'gray' for i in range(len(member_weekly_counts))])
                    fig_member_counts.update_layout(title_x=0.5)
                    st.plotly_chart(fig_member_counts, key=f"counts_{member}")

                with col4:
                    df_member_values = pd.DataFrame({
                    'هفته': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} - {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges],
                    'مقدار': member_weekly_values,
                    'بازه زمانی': [f'{jdatetime.date.fromgregorian(date=start).strftime('%Y/%m/%d')} تا {jdatetime.date.fromgregorian(date=end).strftime('%Y/%m/%d')}' for start, end in week_ranges]
                    })
                    fig_member_values = px.bar(df_member_values, x='هفته', y='مقدار',
                    hover_data=['بازه زمانی'], title=f'مقدار فروش هفتگی {member}')
                    fig_member_values.update_layout(xaxis_title="بازه زمانی ", yaxis_title="مقدار (تومان)", title_font=dict(size=20))
                    fig_member_values.update_traces(marker_color=['#90EE90' if max_member_value_week_index is not None and i == max_member_value_week_index else 'gray' for i in range(len(member_weekly_values))])
                    fig_member_values.update_layout(title_x=0.5)
                    st.plotly_chart(fig_member_values, key=f"values_{member}")
                # target (if they got to the max of the four week they will got 5 percent reward (0.05 * (this weelk - terget)))
                target = max(member_weekly_values) 
                reward_percentage = 0.05
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎁تارگت پاداش: ",  f"{target:,.0f} تومان")
                    if this_week_value > target:
                        reward = reward_percentage * (this_week_value - target)
                        st.write(f"🎉 میزان پاداش:   {reward:,.0f} ")