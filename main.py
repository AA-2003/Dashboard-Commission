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
ACCESS_DENIED = "You do not have access to this section."

# Date range initialization
from_date = (datetime.today() - timedelta(days=DEFAULT_DAYS)).strftime('%Y-%m-%d')
to_date = datetime.today().strftime('%Y-%m-%d')

@st.cache_data
def load_data_cached(sheet, from_date, to_date, won=False):
    """Load data with caching."""
    if sheet:
        return load_sheet()
    else:
        return load_data(from_date, to_date, WON=won)


user_lists = st.secrets["user_lists"]

@st.cache_data
def map_team(name):
    for team, members in user_lists.items():
        if name in members:
            return team
    return 'others'

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title=COMMISSION_DASHBOARD, 
        layout="wide",
    )
    apply_custom_css()
    st.title(COMMISSION_DASHBOARD)

    data = load_data_cached(True, from_date, to_date, won=True)
    data = data[data['deal_status']=='Won'].reset_index(drop=True)
    data['team'] = data['deal_owner'].map(map_team)
    st.session_state.data = data

    # Authentication and Team Selection Logic
    if 'auth' in st.session_state and st.session_state.auth:
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            if st.button(LOGOUT, use_container_width=True):
                st.session_state.team = None
                st.session_state.auth = False
                st.rerun()
    
    if "team" not in st.session_state or st.session_state.team is None:
        st.title(SELECT_YOUR_TEAM)
        team_selection()

    else:
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
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