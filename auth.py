import streamlit as st
import time
def login():
    st.title("صفحه ورود")
    st.write("لطفاً اطلاعات کاربری خود را وارد کنید.")
    user_lists = st.secrets["user_lists"]
    passwords = st.secrets["passwords"]
    roles = st.secrets["roles"]


    usernames = user_lists.get(st.session_state.team, [])
    usernames_map = [st.secrets['names'][name] for name in usernames]
    name_ = st.selectbox("نام کاربری", usernames_map)
    username = next((name for name, mapped_name in st.secrets['names'].items() if mapped_name == name_), name_)
    password = st.text_input("رمز عبور", type="password")
    print(password)
    print(passwords[username])
    if st.button('ورود'):
        if username and password:
            if username in passwords and passwords[username] == password:
                st.session_state.username = username
                st.session_state.name = name_
                st.session_state.role = roles.get(username, 'member')
                st.session_state.auth = True
                st.success(f"ورود موفقیت آمیز! خوش آمدید")
                time.sleep(1) 
                st.rerun()
            else:
                st.error("رمز عبور اشتباه است.")
        else:
            st.warning("لطفاً رمز عبور را وارد کنید.")
        
