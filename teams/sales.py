import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import jdatetime
import numpy as np


def sales():
    """sales team dashboard with optimized metrics and visualizations"""
    st.title("📊 داشبورد تیم Sales")
    
    if not all(key in st.session_state for key in ['username', 'role', 'data', 'team', 'auth']):
        st.error("لطفا ابتدا وارد سیستم شوید")
        return

    # Initialize data and variables
    role = st.session_state.role
    username = st.session_state.username
    name = st.session_state.name

    st.write(f"{name}  عزیز خوش آمدی😃")    