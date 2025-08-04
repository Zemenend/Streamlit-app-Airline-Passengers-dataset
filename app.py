import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# App title
st.set_page_config(page_title="Airline Passengers Satisfaction App", layout="wide", page_icon = "image/logo.jpg")
st.title("🛫 Airline Passenger Satisfaction Prediction App")
st.divider()
# Navigation
page = st.radio("Navigate", ["🏠 Home", "📊 EDA", "🛠️ Model Training and Evaluation","📈 Model Performance", "🤖 Predict", "💡 Recommendations"], horizontal=True)
st.divider()
# Home Page
if page == "🏠 Home":
    st.subheader("Welcome!")
    st.image("image/airport_terminal.jpg")
    st.markdown("""
    This interactive app allows you to:
    - Explore airline passenger satisfaction data
    - Predict satisfaction based on passenger features
    - View model performance
    - Get actionable insights
    """)

# EDA Page
elif page == "📊 EDA":
    st.subheader("Exploratory Data Analysis")
    df = pd.read_csv("data/airline_passenger_satisfaction.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Age Distribution")
        fig = plt.figure()
        sns.histplot(df['Age'], kde=True, bins=30)
        st.pyplot(fig)

    with col2:
        st.write("Satisfaction by Class")
        fig = plt.figure()
        sns.countplot(data=df, x='Class', hue='satisfaction')
        st.pyplot(fig)

    st.write("Flight Distance vs. Satisfaction")
    fig = plt.figure(figsize=(10,4))
    sns.boxplot(x='satisfaction', y='Flight Distance', data=df)
    st.pyplot(fig)


#Model Training and Evaluation page
elif page == "🛠️ Model Training and Evaluation":
    df = pd.read_csv("data/cleaned_airline_passenger_satisfaction.csv")

    # Select features (based on prior correlation result)
    features = ['Online boarding', 'Inflight entertainment', 'Seat comfort', 'On-board service', 'Leg room service', 'Cleanliness', 'Flight Distance', 'Inflight wifi service', 'Baggage handling', 'Inflight service', 'Checkin service', 'Food and drink', 'Ease of Online booking', 'Age']
    X = df[features]
    y = df['satisfaction']
    
    #Train_test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state = 42 , train_size =0.8, stratify = y)
    
    # Normalize the data
    scaler = StandardScaler()
    # Fit and transform the training data
    X_train_sc = scaler.fit_transform(X_train)
    # Transform the test set
    X_test_sc = scaler.transform (X_test)
    
    # Train model
    model = RandomForestClassifier( bootstrap = False, max_depth = 20, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 200)
    model.fit(X_train_sc, y_train)

    # Display training and test accuracy
    if st.checkbox("Show Training and Test Accuracy"):
        st.write(f"Training Accuracy: **{model.score(X_train_sc, y_train):.2f}**")
        st.write(f"Test Accuracy: **{model.score(X_test_sc, y_test):.2f}**")



# Model Performance Page
elif page == "📈 Model Performance":
    st.subheader("📋 Model Evaluation for Random Forest Classifier")
    df = pd.read_csv("data/cleaned_airline_passenger_satisfaction.csv")

    # Select features (based on prior correlation result)
    features = ['Online boarding', 'Inflight entertainment', 'Seat comfort', 'On-board service', 'Leg room service', 'Cleanliness', 'Flight Distance', 'Inflight wifi service', 'Baggage handling', 'Inflight service', 'Checkin service', 'Food and drink', 'Ease of Online booking', 'Age']
    X = df[features]
    y = df['satisfaction']
    
    #Train_test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state = 42 , train_size =0.8, stratify = y)
    
    # Normalize the data
    scaler = StandardScaler()
    # Fit and transform the training data
    X_train_sc = scaler.fit_transform(X_train)
    # Transform the test set
    X_test_sc = scaler.transform (X_test)
    
    # Train model
    model = RandomForestClassifier( bootstrap = False, max_depth = 20, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 200)
    model.fit(X_train_sc, y_train)
    
    y_pred = model.predict(X_test_sc) 
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label= 1) 
    recall = recall_score(y_test, y_pred, pos_label= 1)
    f1 = f1_score(y_test, y_pred, pos_label= 1)

    # Print results
    if st.checkbox("Show performance metrics and Confusion matrix"):
        st.write(f"Model Accuracy Value is:  **{accuracy:.4f}**")
        st.write(f"Model Precision Value is: **{precision:.4f}**")
        st.write(f"Model Recall Value is: **{recall:.4f}**")
        st.write(f" Model F1-score Value is: **{f1:.4f}**")
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test_sc, y_test, ax=ax, cmap='Blues')
        st.pyplot(fig)


# Predict Page
elif page == "🤖 Predict":
    st.subheader("🔍 Predict Passenger Satisfaction")

    age = st.slider("Age", 10, 80, 30)
    distance = st.slider("Flight Distance", 100, 5000, 1000)
    boarding = st.selectbox("Online bording", [0,1,2,3,4,5])
    entertainment = st.selectbox("Inflight entertainment", [0,1,2,3,4,5])
    wifi = st.selectbox("Inflight WiFi Service", [0,1,2,3,4,5])
    seat = st.selectbox("Seat Comfort", [0,1,2,3,4,5])
    clean = st.selectbox("Cleanliness", [0,1,2,3,4,5])
    on_board_service= st.selectbox("On-board service", [0,1,2,3,4,5])
    leg_room = st.selectbox("Leg room service", [0,1,2,3,4,5])
    baggage = st.selectbox("Baggage handling", [0,1,2,3,4,5])
    service = st.selectbox("Inflight service", [0,1,2,3,4,5])
    checkin = st.selectbox("Checkin service", [0,1,2,3,4,5]) 
    meal = st.selectbox("Food and drink", [0,1,2,3,4,5])
    booking =st.selectbox("Ease of Online booking", [0,1,2,3,4,5])

    df = pd.read_csv("data/cleaned_airline_passenger_satisfaction.csv")

    # Select features (based on prior correlation result)
    features = ['Online boarding', 'Inflight entertainment', 'Seat comfort', 'On-board service', 'Leg room service', 'Cleanliness', 'Flight Distance', 'Inflight wifi service', 'Baggage handling', 'Inflight service', 'Checkin service', 'Food and drink', 'Ease of Online booking', 'Age']
    X = df[features]
    y = df['satisfaction']
    
    #Train_test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state = 42 , train_size =0.8, stratify = y)
    
    # Normalize the data
    scaler = StandardScaler()
    # Fit and transform the training data
    X_train_sc = scaler.fit_transform(X_train)
    # Transform the test set
    X_test_sc = scaler.transform (X_test)
    feature = pd.DataFrame([[age, boarding, boarding,distance, wifi, seat, clean,on_board_service, leg_room, baggage,service,checkin, meal,booking]],
                            columns=['Age', 'Online boarding','Inflight entertainment','Flight Distance', 'Inflight wifi service', 'Seat comfort', 'Cleanliness','On-board service', 'Leg room service', 'Baggage handling', 'Inflight service', 'Checkin service', 'Food and drink', 'Ease of Online booking'])
    model = RandomForestClassifier( bootstrap = False, max_depth = 20, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 200)
    model.fit(X_train_sc, y_train)
   
    if st.button("Predict Satisfaction"):
        prediction = model.predict(feature)[0]
        label = "😊 Satisfied" if prediction == 1 else "😐 Neutral or Dissatisfied"
        st.success(f"The model predicts the passenger is **{label}**.")

# Recommendations Page
elif page == "💡 Recommendations":
    st.divider()
    st.subheader("📌 Insights & Recommendations")
    st.divider()
    st.markdown("""
    Based on our analysis, here are key recommendations:
    - 🛜 **Improve Inflight Wi-Fi**: Strong correlation with satisfaction
    - 🎧 **More entertainment options**: Strong correlation with satisfaction
    - 💺 **Upgrade Seat Comfort**: Critical factor in satisfaction levels
    - 🧹 **Enhance Cleanliness**: Clean cabins increase positive reviews
    - 💬 **Focus on Customer Service**: Service ratings matter significantly
    """)
