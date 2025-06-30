import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Insightify", layout="wide")

st.title("📊 Insightify – Dashboard Interaktif + Prediksi Otomatis")

uploaded_file = st.file_uploader("Unggah file CSV kamu", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Pratinjau Data")
    st.dataframe(df)

    st.subheader("📈 Statistik Ringkas")
    st.write(df.describe())

    st.subheader("📊 Visualisasi Otomatis")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_col = st.selectbox("Pilih kolom untuk histogram", numeric_cols)
    if selected_col:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("🧠 Prediksi Otomatis")

    target = st.selectbox("Pilih kolom target prediksi", numeric_cols)
    features = st.multiselect("Pilih fitur (X)", [col for col in numeric_cols if col != target])

    if target and features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Model berhasil dilatih!")

        st.write("**📊 Hasil Evaluasi:**")
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))

        st.subheader("🔮 Coba Prediksi Sendiri")
        user_input = {}
        for feat in features:
            val = st.number_input(f"{feat}", value=float(df[feat].mean()))
            user_input[feat] = val

        if st.button("Prediksi"):
            user_df = pd.DataFrame([user_input])
            hasil = model.predict(user_df)[0]
            st.success(f"Hasil Prediksi: **{hasil:.2f}**")

