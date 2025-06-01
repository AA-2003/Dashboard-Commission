import streamlit as st
import time
def login():
    st.title("صفحه ورود")
    st.write("لطفاً اطلاعات کاربری خود را وارد کنید.")
    user_lists = st.secrets["user_lists"]
    passwords = st.secrets["passwords"]
    roles = st.secrets["roles"]


    usernames = user_lists.get(st.session_state.team, [])
    username = st.selectbox("نام کاربری", usernames)
    password = st.text_input("رمز عبور", type="password")
    if st.button('ورود'):
        if username and password:
            if username in passwords and passwords[username] == password:
                st.session_state.username = username
                st.session_state.role = roles.get(username, 'member')
                st.session_state.auth = True
                st.success(f"ورود موفقیت آمیز! خوش آمدید {username}")
                time.sleep(1) 
                st.rerun()
            else:
                st.error("رمز عبور اشتباه است.")
        else:
            st.warning("لطفاً رمز عبور را وارد کنید.")
        
