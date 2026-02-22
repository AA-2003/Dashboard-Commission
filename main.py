import streamlit as st

from utils.custom_css import apply_custom_css
from utils.sidebar import render_sidebar


COMMISSION_DASHBOARD = "داشبورد عملکرد تیم ها"

    

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title=COMMISSION_DASHBOARD, 
        layout="wide",
    )
    apply_custom_css()

    # we cant load didar data from streamlit servers, so we use google sheet 
    # logo
    st.image("static/logo.svg", width=300)
    
    render_sidebar()

    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("لطفاً ابتدا وارد شوید.")
        return

    # Button for team pages
    st.title(COMMISSION_DASHBOARD)
    st.subheader("تیم های شما: ")
    user_teams = st.session_state.userdata['team']
    user_teams = sorted([team.strip() for team in user_teams.split('|')])
    for team in user_teams:
        if st.button(team, width='stretch'):
            st.session_state.selected_team = team
            st.switch_page(f"pages/{team}.py")

if __name__ == "__main__":
    main()