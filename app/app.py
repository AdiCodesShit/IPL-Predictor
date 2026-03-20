import streamlit as st
import pandas as pd
import joblib

# Config
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="centered"
)

# loading models
MODEL_PATH = "../models/model.pkl"
COLUMNS_PATH = "../models/columns.pkl"
ENCODER_PATH = "../models/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)
encoder = joblib.load(ENCODER_PATH)

# custom css
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #f9c80e;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #1c1f26;
        box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏏 IPL Match Predictor</div>', unsafe_allow_html=True)
st.write("")

teams = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Delhi Daredevils",
    "Kings XI Punjab",
    "Rajasthan Royals"
]

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)
    toss_winner = st.selectbox("Toss Winner", teams)

with col2:
    team2 = st.selectbox("Team 2", teams)
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

venue = st.text_input("Venue")

st.markdown('</div>', unsafe_allow_html=True)

#prediction
if st.button("Predict Match Outcome"):

    if team1 == team2:
        st.error("⚠️ Please select two different teams")
    else:
        input_dict = {
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "venue": venue
        }

        df = pd.DataFrame([input_dict])
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        # Predict probabilities
        probs = model.predict_proba(df)[0]
        teams_list = encoder.classes_

        prob_dict = dict(zip(teams_list, probs))

        team1_prob = prob_dict.get(team1, 0)
        team2_prob = prob_dict.get(team2, 0)

        total = team1_prob + team2_prob

        if total == 0:
            st.error("Not enough data to predict this matchup")
        else:
            team1_prob = (team1_prob / total) * 100
            team2_prob = (team2_prob / total) * 100

            # Winner
            if team1_prob > team2_prob:
                winner = team1
                win_prob = team1_prob
            else:
                winner = team2
                win_prob = team2_prob

            # Result
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.success(f"🏆 Predicted Winner: {winner}")
            st.write(f"### Confidence: {win_prob:.2f}%")

            # Confidence Meter
            st.progress(int(win_prob))

            st.write("")

            # Probabliity meter
            st.subheader("📊 Win Probability")

            col3, col4 = st.columns(2)

            with col3:
                st.metric(team1, f"{team1_prob:.2f}%")

            with col4:
                st.metric(team2, f"{team2_prob:.2f}%")

            # Bar chart
            chart_df = pd.DataFrame({
                "Team": [team1, team2],
                "Probability": [team1_prob, team2_prob]
            })

            st.bar_chart(chart_df.set_index("Team"))

            st.markdown('</div>', unsafe_allow_html=True)