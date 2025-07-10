import streamlit as st import pandas as pd import numpy as np from sklearn.ensemble import RandomForestClassifier from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Teen Patti Pot-Based Predictor", layout="centered") st.title("🧠 ML Predictor: Will High, Mid, or Low Pot Win Next?")

st.markdown("Train this ML model on 30+ rounds of history and then predict whether the high, mid, or low pot will win in the next round.")

if "rounds" not in st.session_state: st.session_state.rounds = []

=== 1. Add Round History ===

st.subheader("➕ Add Round History") with st.form("round_input"): pot_a = st.number_input("Pot A (in thousands)", min_value=0, step=1) pot_b = st.number_input("Pot B (in thousands)", min_value=0, step=1) pot_c = st.number_input("Pot C (in thousands)", min_value=0, step=1) winner = st.selectbox("Winning Chair", ["A", "B", "C"]) submitted = st.form_submit_button("Add Round")

if submitted: st.session_state.rounds.append({"A": pot_a, "B": pot_b, "C": pot_c, "Winner": winner}) st.success(f"✅ Round {len(st.session_state.rounds)} added")

=== 2. Show History ===

if st.session_state.rounds: df = pd.DataFrame(st.session_state.rounds) st.subheader("📜 Training History (Last 15 Rounds)") st.dataframe(df.tail(15), use_container_width=True)

if len(df) >= 30:
    # Label pot type (High/Mid/Low) of winning chair
    def get_winner_pot_rank(row):
        pots = {"A": row["A"], "B": row["B"], "C": row["C"]}
        sorted_pots = sorted(pots.items(), key=lambda x: x[1])
        rank_map = {
            sorted_pots[0][0]: "Low",
            sorted_pots[1][0]: "Mid",
            sorted_pots[2][0]: "High"
        }
        return rank_map[row["Winner"]]

    df["WinnerPotType"] = df.apply(get_winner_pot_rank, axis=1)

    # === 3. Train Model ===
    X = df[["A", "B", "C"]]
    y = df["WinnerPotType"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    st.success("✅ Model trained with your 30+ rounds of history")

    # === 4. Predict for Next Round ===
    st.subheader("🔮 Predict Next Round Winner Pot Type")
    with st.form("predict_form"):
        na = st.number_input("Pot A for next round", min_value=0, step=1, key="na")
        nb = st.number_input("Pot B for next round", min_value=0, step=1, key="nb")
        nc = st.number_input("Pot C for next round", min_value=0, step=1, key="nc")
        predict_btn = st.form_submit_button("Predict Winner Pot Type")

    if predict_btn:
        input_data = np.array([[na, nb, nc]])
        pred_encoded = model.predict(input_data)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(input_data)[0]

        st.subheader(f"🏆 Predicted Pot Type to Win: **{pred_label.upper()}**")
        st.markdown("### 📊 Confidence:")
        for i, prob in enumerate(proba):
            st.markdown(f"- **{le.inverse_transform([i])[0]}** pot: {prob*100:.2f}%")

        # Tip where to bet:
        sorted_pots = sorted([(na, "A"), (nb, "B"), (nc, "C")])
        pot_map = {"Low": sorted_pots[0][1], "Mid": sorted_pots[1][1], "High": sorted_pots[2][1]}
        best_bet = pot_map[pred_label]
        st.success(f"🎯 Suggested Bet: Chair **{best_bet}** (it has the predicted {pred_label.upper()} pot)")

else:
    st.warning("⚠️ Add at least 30 rounds of history to start predicting.")

else: st.info("Start by entering past game data (Pot A/B/C + Winner)")

