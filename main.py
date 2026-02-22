import streamlit as st

from utils.custom_css import apply_custom_css
from utils.sidebar import render_sidebar


COMMISSION_DASHBOARD = "داشبورد عملکرد تیم ها"

def load_didar_date():
    """test loading data from didar in streamlit servers"""
    import requests
    api_key = st.secrets.get("GENERAL").get("DIDAR_API_KEY", "")
    url = f"https://app.didar.me/api/deal/search?apikey={api_key}"
    try:
        payload = {
            "Criteria": {
                "ContactIds":["4d6157b5-a8aa-4d0e-b203-1da3f19e3dc2"]},
            "From":0,
            "Limit":1000
        }
        res = requests.post(url, json=payload)
        res.raise_for_status()

        data = res.json()
        st.write(data)
    except Exception as e:
        st.error(f"خطا در بارگذاری داده‌ها از دیدار: {e}")
        return None
    

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title=COMMISSION_DASHBOARD, 
        layout="wide",
    )
    apply_custom_css()

    # test loading data from didar in streamlit servers
    load_didar_date()

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