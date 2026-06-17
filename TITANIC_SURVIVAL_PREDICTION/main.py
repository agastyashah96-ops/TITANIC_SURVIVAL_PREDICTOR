import streamlit as st

# 1. Define your page objects pointing to your sub-files
home_page = st.Page("views/app.py", title="Home", icon=":material/home:", default=True)
visualization = st.Page("views/charts.py", title="Visualization", icon=":material/analytics:")
passenger = st.Page("views/pages.py", title="Create Your Passenger",  icon=":material/analytics:")
pg = st.navigation([home_page,visualization,passenger])
#..\.venv\Scripts\streamlit run main.py

# 3. Always invoke .run() at the end to display the active page
pg.run()

