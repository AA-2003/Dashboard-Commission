import streamlit as st

def apply_custom_css():
    st.markdown(
        """
        <style>
        /* استفاده از فونت Tahoma برای همه‌ی متن‌ها */
        html, body, div, span, input, textarea, label, select, button, p, h1, h2, h3, h4, h5, h6 {
            font-family: Tahoma, sans-serif !important;
        }

        /* تنظیم جهت راست‌به‌چپ برای فارسی */
        .main, .block-container {
            direction: rtl !important;
            text-align: right !important;
        }

        /* اجبار راست‌چین برای فرم‌ها و ویجت‌ها */
        input, textarea, select,
        .stSelectbox > div, .stTextInput > div,
        .stTextArea > div, .stDateInput > div {
            direction: rtl !important;
            text-align: right !important;
        }

        /* راست‌چین کردن متن Markdown */
        .stMarkdown {
            direction: rtl !important;
            text-align: right !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
