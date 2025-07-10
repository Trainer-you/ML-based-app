import streamlit as st 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import 
LabelEncoder

st.set_page_config(page_title="Simple Pot Winner Predictor", layout="centered") st.title("🔮 Predict Next Winning Pot (Based on Last 10 Rounds)")

st.markdown("Enter the last 10 rounds of data. The app will learn the pattern and predict the next likely winning pot type (High, Mid, or Low) without knowing current pot values.")

if "rounds" not in st.session_state: st.session_state.rounds = []

=== 1. Add Round History ===

st.subheader("➕ Add Round History") with st.form("round_input"): pot_a = st.number_input("Pot A", min_value=0, step=1) pot_b = st.number_input("Pot B", min_value=0, step=1) pot_c = st.number_input("Pot C", min_value=0, step=1) winner = st.selectbox("Winning Chair", ["A", "B", "C"]) submitted = st.form_submit_button("Add Round")

if submitted: st.session_state.rounds.append({"A": pot_a, "B": pot_b, "C": pot_c, "Winner": winner}) if len(st.session_state.rounds) > 10: st.session_state.rounds.pop(0)  # Keep only last 10 rounds st.success(f"✅ Round added. Total rounds stored: {len(st.session_state.rounds)}")

=== 2. Show History ===

if st.session_state.rounds: df = pd.DataFrame(st.session_state.rounds) st.subheader("📜 Last 10 Rounds") st.dataframe(df, use_container_width=True)

if len(df) == 10:
    # Label winner pot type
    def get_winner_pot_type(row):
        pots = {"A": row["A"], "B": row["B"], "C": row["C"]}
        sorted_pots = sorted(pots.items(), key=lambda x: x[1])
        rank_map = {
            sorted_pots[0][0]: "Low",
            sorted_pots[1][0]: "Mid",
            sorted_pots[2][0]: "High"
        }
        return rank_map[row["Winner"]]

    df["WinnerPotType"] = df.apply(get_winner_pot_type, axis=1)

    # Use only WinnerPotType to train a sequence model
    X = []
    y = []
    for i in range(len(df) - 3):
        seq = df["WinnerPotType"].iloc[i:i+3].tolist()
        label = df["WinnerPotType"].iloc[i+3]
        X.append(seq)
        y.append(label)

    if len(X) > 0:
        le = LabelEncoder()
        X_enc = [le.fit_transform(x) for x in X]
        y_enc = le.transform(y)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_enc, y_enc)

        # Predict using last 3
        recent_seq = df["WinnerPotType"].iloc[-3:].tolist()
        recent_seq_enc = le.transform(recent_seq).reshape(1, -1)
        pred = model.predict(recent_seq_enc)[0]
        pred_label = le.inverse_transform([pred])[0]

        st.subheader("🔮 Prediction:")
        st.success(f"The next likely winning pot type is: **{pred_label.upper()}**")
    else:
        st.warning("⚠️ Not enough sequential data to make prediction.")
else:
    st.warning("⚠️ Add exactly 10 rounds to enable prediction.")

else: st.info("👈 Add some rounds to begin.")

