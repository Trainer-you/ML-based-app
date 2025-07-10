import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Teen Patti ML Predictor", layout="centered")
st.title("🧠 ML-Powered Teen Patti Predictor")
st.markdown("Train the model live as you play. Add rounds below to improve predictions.")

if "rounds" not in st.session_state:
    st.session_state.rounds = []

# Input form
chairs = ["A", "B", "C"]
with st.form("round_input"):
    st.subheader("➕ Add Round Data")
    pot_a = st.number_input("Pot A (in thousands)", min_value=0, step=1)
    pot_b = st.number_input("Pot B (in thousands)", min_value=0, step=1)
    pot_c = st.number_input("Pot C (in thousands)", min_value=0, step=1)
    winner = st.selectbox("Winning Chair", chairs)
    submit = st.form_submit_button("Add Round")

if submit:
    st.session_state.rounds.append({"Round": len(st.session_state.rounds)+1, "A": pot_a, "B": pot_b, "C": pot_c, "Winner": winner})
    st.success(f"✅ Round {len(st.session_state.rounds)} added")

# Show history
if st.session_state.rounds:
    df = pd.DataFrame(st.session_state.rounds)
    st.subheader("📜 Match History")
    st.dataframe(df.tail(15), use_container_width=True)

    if len(df) >= 30:
        # Prepare data
        X = df[["A", "B", "C"]]
        y = df["Winner"]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)

        st.success("✅ Model trained on latest data")

        # Predict next round
        st.subheader("🔮 Predict Next Round")
        with st.form("predict_input"):
            new_a = st.number_input("Next Round Pot A", min_value=0, step=1, key="predict_a")
            new_b = st.number_input("Next Round Pot B", min_value=0, step=1, key="predict_b")
            new_c = st.number_input("Next Round Pot C", min_value=0, step=1, key="predict_c")
            predict_button = st.form_submit_button("Predict Winner")

        if predict_button:
            new_input = np.array([[new_a, new_b, new_c]])
            pred_encoded = model.predict(new_input)[0]
            pred_label = le.inverse_transform([pred_encoded])[0]
            proba = model.predict_proba(new_input)[0]

            st.subheader(f"🏆 Predicted Winner: **Chair {pred_label}**")
            st.markdown("### 📊 Prediction Probabilities")
            for i, prob in enumerate(proba):
                st.markdown(f"- **Chair {le.inverse_transform([i])[0]}**: {prob*100:.2f}%")
    else:
        st.warning("⚠️ Add at least 30 rounds to train the ML model.")
else:
    st.info("Enter some round data above to get started.")
