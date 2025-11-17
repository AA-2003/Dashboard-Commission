import streamlit as st

from utils.sheetConnect import load_sheet
from utils.func import load_data_cached
from utils.logger import log_event

def refresh_data():
    # users = st.session_state.get('users', None)
    st.cache_data.clear()
    # st.session_state.users  = users
    st.session_state.refresh_trigger = True

def render_sidebar():
    # extract users
    if 'users' not in st.session_state:
        try:
            users = load_sheet(key='MAIN_SPREADSHEET_ID', sheet_name='Users')
            st.session_state.users = users
        except Exception as e:
            log_event('', "Error loading users data", str(e))
            st.error("خطا در بارگذاری داده‌ها")

    if 'deals_data' not in st.session_state or st.session_state.deals_data is None:
        try:
            deals_data = load_data_cached(spreadsheet_key='DEALS_SPREADSHEET_ID', sheet_name='Didar Deals')
            st.session_state.deals_data = deals_data
        except Exception as e:
            log_event('', "Error loading deals data", str(e))
            st.error("خطا در بارگذاری داده‌ها")


    with st.sidebar:
        st.button("رفرش داده‌ها", on_click=refresh_data)
        
        if st.session_state.get('refresh_trigger', False):
            st.session_state.refresh_trigger = False
            st.rerun()

        try:
            if st.session_state.get('logged_in', None) is None or not st.session_state.logged_in:
                st.header("ورود به داشبورد")
                username = st.selectbox(options=st.session_state.users['name'].tolist(), label="نام کاربری")
                password = st.text_input("رمز عبور", type="password")
                if st.button("ورود"):
                    user_row = st.session_state.users[st.session_state.users['name'] == username]
                    if not user_row.empty:
                        if str(password) == str(user_row.iloc[0]['password']):
                            st.session_state.logged_in = True
                            st.session_state.userdata = user_row.iloc[0]
                            st.success("ورود موفقیت‌آمیز بود!")
                            log_event(username, "login", f"User {username} logged in.")
                            st.rerun()
                        else:
                            log_event(username, "failed_login", f"User {username} provided incorrect password.")
                            st.error("نام کاربری یا رمز عبور اشتباه است.")
                    else:
                        log_event(username, "failed_login", f"User {username} not found.")
                        st.error("کاربر یافت نشد.")
            else:
                st.success(f"خوش آمدید، {st.session_state.userdata['name']}!")
                if st.button("خروج"):
                    log_event(st.session_state.userdata['name'], "logout", f"User {st.session_state.userdata['name']} logged out.")
                    st.session_state.logged_in = False
                    st.session_state.userdata = None
                    st.rerun()
        except Exception as e:
            st.error(f"خطا در ورود: {str(e)}")