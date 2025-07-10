import streamlit as st import pandas as pd import numpy as np from sklearn.ensemble import RandomForestClassifier from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Teen Patti Pot Type Predictor", layout="centered") st.title("🧠 ML Predictor: Winning Pot Type (High / Medium / Low)") st.markdown("Train the model on your round history and predict whether the next winner will come from a High, Medium, or Low pot.")

if "rounds" not in st.session_state: st.session_state.rounds = []

Input form

with st.form("round_input"): st.subheader("➕ Add Round History") pot_a = st.number_input("Pot A (in thousands)", min_value=0, step=1) pot_b = st.number_input("Pot B (in thousands)", min_value=0, step=1) pot_c = st.number_input("Pot C (in thousands)", min_value=0, step=1) winner = st.selectbox("Winning Chair", ["A", "B", "C"]) submit = st.form_submit_button("Add Round")

if submit: st.session_state.rounds.append({"A": pot_a, "B": pot_b, "C": pot_c, "Winner": winner}) st.success(f"✅ Round {len(st.session_state.rounds)} saved")

Display data

if st.session_state.rounds: df = pd.DataFrame(st.session_state.rounds) st.subheader("📜 Round History") st.dataframe(df.tail(15), use_container_width=True)

if len(df) >= 30:
    # Label target: High / Medium / Low pot winner
    def determine_winner_pot_level(row):
        pots = {"A": row["A"], "B": row["B"], "C": row["C"]}
        sorted_chairs = sorted(pots.items(), key=lambda x: x[1])
        chair_rank = {sorted_chairs[0][0]: "Low", sorted_chairs[1][0]: "Medium", sorted_chairs[2][0]: "High"}
        return chair_rank[row["Winner"]]

    df["PotLevel"] = df.apply(determine_winner_pot_level, axis=1)

    # Train ML model
    X = df[["A", "B", "C"]]
    y = df["PotLevel"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    st.success("✅ Model trained to classify winning pot type")

    # Prediction section
    st.subheader("🔮 Predict Winning Pot Type for Next Round")
    with st.form("predict_input"):
        next_a = st.number_input("Next Pot A", min_value=0, step=1, key="na")
        next_b = st.number_input("Next Pot B", min_value=0, step=1, key="nb")
        next_c = st.number_input("Next Pot C", min_value=0, step=1, key="nc")
        predict_btn = st.form_submit_button("Predict Pot Type")

    if predict_btn:
        pred_input = np.array([[next_a, next_b, next_c]])
        pred_encoded = model.predict(pred_input)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(pred_input)[0]

        st.subheader(f"🏆 Predicted Winner Pot Type: **{pred_label.upper()}**")
        st.markdown("### 📊 Prediction Confidence:")
        for i, prob in enumerate(proba):
            st.markdown(f"- **{le.inverse_transform([i])[0]}** pot: {prob*100:.2f}%")

else:
    st.warning("⚠️ Add at least 30 rounds to train the model.")

else: st.info("Start entering your round history to unlock ML prediction.")

