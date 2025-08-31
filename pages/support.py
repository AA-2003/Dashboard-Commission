import streamlit as st
from utils.custom_css import apply_custom_css
from utils.write_data import append_to_sheet
from datetime import datetime

def support():
    """Support page content."""
    st.set_page_config(
        page_title="پشتیبانی", 
        layout="wide",
    )
    apply_custom_css()


    st.title("صفحه پشتیبانی")

    with st.form(key='support_form'):
        name = st.text_input('نام خود را وارد کنید:')
        email = st.text_input('ایمیل خود را وارد کنید:')
        des = st.text_area('توضیحات:', height=200)  # Use text_area for longer descriptions

        submit_button = st.form_submit_button('ارسال')

        if submit_button:
            if name  and email and des:
                row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name, email, des]
                append_to_sheet(row, 'Requests')
                st.success("درخواست شما با موفقیت ثبت شد!")  # Show a success message
            else:
                st.warning("لطفا همه فیلدها را پر کنید")  # Show a warning if fields are missing


if __name__ == "__main__":
    support()