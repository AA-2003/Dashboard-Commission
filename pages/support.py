import streamlit as st
from datetime import datetime

from utils.sidebar import render_sidebar
from utils.custom_css import apply_custom_css
from utils.write_data import append_to_sheet

def support():
    """Support page content."""
    st.set_page_config(
        page_title="پشتیبانی", 
        layout="wide",
    )       
    apply_custom_css()
    render_sidebar()

    st.title("صفحه پشتیبانی")

    with st.form(key='support_form'):
        name = st.text_input('نام و نام خانوادگی:')
        email = st.text_input('ایمیل (اختیاری):')
        des = st.text_area('شرح مشکل یا درخواست خود را بنویسید:', height=200) 
        submit_button = st.form_submit_button('ارسال')

        if submit_button:
            if name  and email and des:
                row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name, email, des, 'Commission Dashboard']
                append_to_sheet(row, 'Dashboard reports')
                st.success("درخواست شما با موفقیت ثبت شد!") 
            else:
                st.warning("لطفا همه فیلدها را پر کنید")


if __name__ == "__main__":
    support()