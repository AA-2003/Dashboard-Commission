import streamlit as st

def apply_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Tahoma');

        /* Set font globally */
        html, body {
            font-family: Tahoma, sans-serif !important;
        }

        /* Only set RTL and text-align right for content areas */
        .main, .block-container {
            direction: rtl !important;
            text-align: right !important;
        }

        /* Fix direction in input widgets */
        .stSelectbox > div > div,
        .stDateInput > div > input,
        .stTextInput > div > input,
        .stTextArea > div > textarea {
            direction: rtl !important;
            text-align: right !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
