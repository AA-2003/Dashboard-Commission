from datetime import datetime, timedelta
import streamlit as st

from utils.custom_css import apply_custom_css
from utils.data_loader import load_data
from utils.load_data import load_sheet
from utils.auth import login
from teams.platform import platform
from teams.social import social
from teams.sales import sales
from teams.b2b import b2b

from utils.logger import logger


# Constants
DEFAULT_DAYS = 80
COMMISSION_DASHBOARD = "داشبورد عملکرد تیم ها"
SELECT_YOUR_TEAM = "🎯 تیم خود را انتخاب کنید:"
GO_BACK_TO_MAIN_PAGE = "بازگشت"
LOGOUT = "خروج"
ACCESS_DENIED = "شما به این بخش دسترسی ندارید."

# Date range initialization
from_date = (datetime.today() - timedelta(days=DEFAULT_DAYS)).strftime('%Y-%m-%d')
to_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

@st.cache_data(ttl=600, show_spinner=False)
def load_data_cached(sheet, from_date, to_date, won=False):
    """Load data with caching."""
    if sheet:
        try:
            return load_sheet()
        except Exception as e:
            logger.error(f"Error in load_sheet(): {e!r}")
            return None
    else:
        try:
            return load_data(from_date, to_date, WON=won)
        except Exception as e:
            logger.error(f"Error in load_data(): {e!r}")
            return None

user_lists = st.secrets.get("user_lists", {})

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title=COMMISSION_DASHBOARD, 
        layout="wide",
    )

    apply_custom_css()

    # logo
    st.image("static/logo.svg", width=300)

    # Load initial data from sheet
    if 'data' not in st.session_state:
        data = load_data_cached(True, from_date, to_date, won=True)
        if data is None:
            logger.error("Loaded data is None. Cannot continue.")
            st.error("داده‌ها با خطا دریافت شد. لطفاً بعدا مجدد تلاش کنید.")
            st.stop()
        else:
            # Defensive check for columns
            if not hasattr(data, "__getitem__"):
                logger.error("Loaded data is not subscriptable.")
                st.error("فرمت داده‌ها صحیح نیست.")
                st.stop()
            elif not all(col in data for col in ['deal_status', 'deal_value']):
                logger.error("Loaded data missing required columns: 'deal_status', 'deal_value'.")
                st.error("ستون‌های مورد نیاز یافت نشدند.")
                st.stop()
            else:
                try:
                    filtered = data[
                        (data['deal_status'] == 'Won') &
                        (data['deal_value'] != 0)
                    ].reset_index(drop=True)
                    logger.info(f"Loaded {len(filtered)} deals after filtering.")
                except Exception as e:
                    logger.error(f"Error filtering data: {e!r}")
                    st.error("خطا در فیلتر کردن داده‌ها.")
                    st.stop()
                st.session_state.data = filtered
    # st.dataframe(st.session_state.data)
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
        col1, col2, col3 = st.columns([1, 3, 1])
        with st.sidebar:
            if st.button(GO_BACK_TO_MAIN_PAGE, use_container_width=True):
                st.session_state.team = None
                st.rerun()
        auth_check()

def team_selection():
    """Team selection buttons."""
    cols = st.columns(2)
    teams = {
        'platform': cols[0],
        'b2b': cols[1],
        'sales': cols[0],
        'social': cols[1],
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
            username = st.session_state.get("username")
            user_list_for_team = user_lists.get(team_key, [])
            if username in user_list_for_team:
                page()
            else:
                logger.warning(f"Unauthorized access attempt by {username} to team {team_key}.")
                st.error(ACCESS_DENIED)
                st.stop()
        else:
            logger.warning(f"Unknown team selected: {selected_team}")
            st.error(ACCESS_DENIED)
            st.stop()

if __name__ == "__main__":
    main()