from data_loader import load_data
from datetime import datetime, timedelta
import streamlit as st
from custom_css import apply_custom_css
from auth import login
from pages_.b2b import b2b
from pages_.platform import platform
from pages_.social import social
from pages_.sales import sales
from load_sheet import load_sheet


# Constants for date range
DEFAULT_DAYS = 40
COMMISSION_DASHBOARD = "داشبورد کمیسیون"
SELECT_YOUR_TEAM = "🎯 تیم خود را انتخاب کنید:"
GO_BACK_TO_MAIN_PAGE = "بازگشت"
LOGOUT = "خروج"
PLATFORM = "Platform"
B2B = "B2B"
SALES = "Sales"
SOCIAL = "Social"
ACCESS_DENIED = "شما به این بخش دسترسی ندارید."

# Date range initialization
from_date = (datetime.today() - timedelta(days=DEFAULT_DAYS)).strftime('%Y-%m-%d')
to_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

@st.cache_data
def load_data_cached(sheet, from_date, to_date, won=False):
    """Load data with caching."""
    if sheet:
        return load_sheet()
    else:
        return load_data(from_date, to_date, WON=won)


user_lists = st.secrets["user_lists"]

@st.cache_data(ttl=60000)
def map_team(name):
    if name in ['پلت‌فرم']:
        return 'platform'
    
    elif name in ['مهمان واسطه']:
        return 'b2b'
    
    elif name in ['دایرکت اینستاگرام', 'تلگرام(سوشال)', 'واتساپ(سوشال)']:
        return 'social'
    
    elif name in ['تماس ورودی (مشتری)', 'چت واتس‌اپ', 'معرف', 'سایر', 'چت سایت',
                 'چت تلگرام', 'پیامک فرم', 'تماس فرم سایت',
                ]:
        return 'sales'
    
    else:
        return 'others'



def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title=COMMISSION_DASHBOARD, 
        layout="wide",
    )
    apply_custom_css()
    # logo
    st.image("static/logo.svg", width=300)
    with st.sidebar:
        st.title(COMMISSION_DASHBOARD)

    
    # Load initial data from sheet
    if 'data' not in st.session_state:
        data = load_data_cached(True, from_date, to_date, won=True)
        data = data[data['deal_status']=='Won'].reset_index(drop=True)
        data['team'] = data['deal_source'].map(map_team)
        st.session_state.data = data

        st.dataframe(data)

    # Add refresh button in sidebar
    with st.sidebar:
        if st.button("🔄 بروزرسانی داده‌ها", use_container_width=True):
            # Clear cache and reload data
            load_data_cached.clear()
            data = load_data_cached(False, from_date, to_date, won=True)
            data = data[data['deal_status']=='Won'].reset_index(drop=True)
            data['team'] = data['deal_source'].map(map_team)
            st.session_state.data = data
            st.success("داده‌ها با موفقیت بروزرسانی شدند!")

    # Authentication and Team Selection Logic

    if 'auth' in st.session_state and st.session_state.auth:
        with st.sidebar:
            if st.button(LOGOUT, use_container_width=True):
                st.session_state.team = None
                st.session_state.auth = False
                st.rerun()
    
    if "team" not in st.session_state or st.session_state.team is None:
        st.title(SELECT_YOUR_TEAM)
        team_selection()

    else:
        col1, col2, col3 = st.columns([1,3,1])
        with st.sidebar:
            if st.button(GO_BACK_TO_MAIN_PAGE, use_container_width=True):
                st.session_state.team = None
                st.rerun()
        auth_check()

def team_selection():
    """Team selection buttons."""
    cols = st.columns(2)
    teams = {
        PLATFORM: cols[0],
        B2B: cols[1],
        SALES: cols[0],
        SOCIAL: cols[1],
    }

    for team, col in teams.items():
        with col:
            if st.button(team, use_container_width=True):
                st.session_state.team = team.lower()
                st.rerun()

def auth_check():
    """Authentication check and page routing."""
    if 'auth' not in st.session_state or not st.session_state.auth:
        login()
    else:
        st.dataframe(st.session_state.data.drop_duplicates(subset=['deal_owner']))

        team_page_mapping = {
            "platform": (platform, 'platform'),
            "b2b": (b2b, 'b2b'),
            "sales": (sales, 'sales'),
            "social": (social, 'social'),
        }
        
        selected_team = st.session_state.team
        if selected_team in team_page_mapping:
            page, team_key = team_page_mapping[selected_team]
            if st.session_state.username in user_lists[team_key]:
                page()
            else:
                st.error(ACCESS_DENIED)
                st.stop()
        else:
            st.error(ACCESS_DENIED)
            st.stop()

if __name__ == "__main__":
    main()