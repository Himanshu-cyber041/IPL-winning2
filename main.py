import base64
import streamlit as st
import pickle
import pandas as pd

# Function to load image as base64 for background styling
@st.cache_data
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Load background image as base64
img = get_img_as_base64("background.jpg")

if img:
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    width: 100%;
    height:100%;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-size: cover;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"]] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"]] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Add title and description
st.markdown("""
    # **IPL VICTORY PREDICTOR**            
""")

# Dropdown selections for teams and cities
teams = ['--- select ---', 'Sunrisers Hyderabad', 'Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']
cities = ['Bangalore', 'Hyderabad', 'Kolkata', 'Mumbai', 'Visakhapatnam', 'Indore', 'Durban', 'Chandigarh', 'Delhi',
          'Dharamsala', 'Ahmedabad', 'Chennai', 'Ranchi', 'Nagpur', 'Mohali', 'Pune', 'Bengaluru', 'Jaipur', 'Port Elizabeth',
          'Centurion', 'Raipur', 'Sharjah', 'Cuttack', 'Johannesburg', 'Cape Town', 'East London', 'Abu Dhabi', 'Kimberley',
          'Bloemfontein']

# Load the pre-trained model with error handling
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    pipe = None

# Selectboxes for team and city
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', teams)

with col2:
    if batting_team == '--- select ---':
        bowling_team = st.selectbox('Select Bowling Team', teams)
    else:
        filtered_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox('Select Bowling Team', filtered_teams)

# City selection
seleted_city = st.selectbox('Select Venue', cities)

# Numerical input for game data
target = st.number_input('Target', min_value=0)

col1, col2, col3 = st.columns(3)
with col1:
    score = st.number_input('Score', min_value=0)
with col2:
    overs = st.number_input("Overs Completed", min_value=0.0)
with col3:
    wickets = st.number_input("Wickets Down", min_value=0)

# Predict button
if st.button('Predict Winning Probability'):
    try:
        # Ensure that score, overs, wickets, and target are sensible
        if overs == 0 or score > target:
            st.error("Invalid input values: Ensure that overs and score are valid.")
        else:
            # Calculate remaining values
            runs_left = target - score
            balls_left = 120 - (overs * 6)
            wickets_remaining = 10 - wickets
            crr = score / overs
            rrr = runs_left / (balls_left / 6)

            # Create input DataFrame
            input_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [seleted_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_remaining': [wickets_remaining],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            # Predict using the model
            if pipe:
                result = pipe.predict_proba(input_data)
                loss = result[0][0]
                win = result[0][1]

                st.header(f"{batting_team} = {round(win * 100)}%")
                st.header(f"{bowling_team} = {round(loss * 100)}%")
            else:
                st.error("Model is not loaded properly.")
    except Exception as e:
        st.error(f"Some error occurred: {e}")
