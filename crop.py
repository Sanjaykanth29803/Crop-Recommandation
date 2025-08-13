import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Crop_recommendation.csv")

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X = df.drop('label', axis=1)
y = df['label']
model = RandomForestClassifier()
model.fit(X, y)

st.set_page_config(layout="wide")
st.title("🌿 Crop Recommendation System")

tab1, tab2, tab3, tab4 = st.tabs(["🌱 Predict", "📊 Visualise", "📁 Dataset", "ℹ️ About"])

with tab1:
    st.header("🌾 Predict Suitable Crop")
    st.markdown("Fill in the values below to get the recommended crop:")

    col1, col2, col3 = st.columns(3)

    with col1:
        nitrogen = st.number_input("Nitrogen (N)", 0, 140)
        phosphorous = st.number_input("Phosphorous (P)", 5, 145)
    with col2:
        potassium = st.number_input("Potassium (K)", 5, 205)
        temperature = st.number_input("Temperature (°C)", 8, 45)
    with col3:
        humidity = st.number_input("Humidity (%)", 10, 100)
        ph = st.number_input("pH Level", 3.5, 9.5)

    rainfall = st.slider("Rainfall (mm)", 10.0, 300.0, step=1.0)

    if st.button("🌱 Recommend Crop"):
        input_data = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        predicted_crop = le.inverse_transform(prediction)[0]
        st.success(f"✅ Recommended Crop: **{predicted_crop.upper()}**")

with tab2:
    st.header("📊 Data Visualizations")
    st.markdown("Explore the relationships and distributions in the crop dataset:")

    fig1 = px.histogram(df, x='temperature', nbins=30, title='Temperature Distribution')
    fig2 = px.histogram(df, x='humidity', nbins=30, title='Humidity Distribution')
    fig3 = px.histogram(df, x='ph', nbins=30, title='pH Distribution')
    fig4 = px.histogram(df, x='rainfall', nbins=30, title='Rainfall Distribution')
    fig5 = px.box(df, y='rainfall', title='Rainfall Outliers')
    fig6 = px.scatter(df, x='temperature', y='humidity', color=le.inverse_transform(y), title='Temperature vs Humidity')
    fig7 = px.scatter(df, x='ph', y='rainfall', color=le.inverse_transform(y), title='pH vs Rainfall')
    fig8 = px.violin(df, y='temperature', box=True, title='Temperature Violin Plot')
    fig9 = px.density_contour(df, x='ph', y='humidity', title='pH vs Humidity Density')

    label_counts = pd.Series(le.inverse_transform(y)).value_counts().reset_index()
    label_counts.columns = ['Crop', 'Count']
    fig10 = px.bar(label_counts, x='Crop', y='Count', title='Crop Count Distribution')

    st.subheader("🔗 Correlation Heatmap")
    fig_corr, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig_corr)

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig5, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)
    st.plotly_chart(fig7, use_container_width=True)
    st.plotly_chart(fig8, use_container_width=True)
    st.plotly_chart(fig9, use_container_width=True)
    st.plotly_chart(fig10, use_container_width=True)

with tab3:
    st.header("📁 Crop Dataset Preview")
    st.dataframe(df.head(20))

    if st.checkbox("Show Full Dataset"):
        st.dataframe(df)

    st.markdown("**Shape:**")
    st.write(df.shape)

    st.markdown("**Columns:**")
    st.write(df.columns.tolist())

with tab4:
    st.header("ℹ️ About the Project")
    st.markdown("""
    This Crop Recommendation System is built using **Machine Learning** and **Streamlit** to help farmers 
    or agricultural experts recommend the most suitable crop to grow based on the following environmental parameters:
    
    - Nitrogen, Phosphorous, Potassium (NPK) levels
    - Temperature
    - Humidity
    - pH of the soil
    - Rainfall
    
    ### 🔧 Tools & Technologies Used:
    - Python
    - Pandas, NumPy
    - Random Forest Classifier
    - Plotly, Seaborn, Matplotlib
    - Streamlit

    ### 👨‍💻 Developed by:
    - Sanjaykanth Chandran  
    - IT, GCE Erode (IRTT)  
    - Data Analyst

    ---
    For more details, contact me on [LinkedIn](https://www.linkedin.com/in/sanjaykanth-chandran-765529298).
    """)
