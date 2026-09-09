# ============================================================
# Hotel Booking Cancellation Prediction
# Faculty of Computers & Artificial Intelligence
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.title("🏨 Hotel Booking Cancellation Prediction")

st.markdown("""
Predict whether a hotel reservation is likely to be **Canceled**
before the customer's arrival using a Machine Learning model.
""")

st.divider()

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📊 Project Information")

st.sidebar.success("Best Model : CatBoost")

st.sidebar.metric(
    "Recall",
    "84.78%"
)

st.sidebar.metric(
    "ROC AUC",
    "92.21%"
)
st.sidebar.metric(
    "Accuracy",
    "83.24%"
)

st.sidebar.metric(
    "Selected Features",
    "20"
)

st.sidebar.divider()

st.sidebar.markdown("""
### Dataset

Hotel Booking Demand Dataset

Records : 74,361

Machine Learning Task :

Classification

Target :

is_canceled
""")

# ============================================================
# Load Pickle Files
# ============================================================

@st.cache_resource
def load_files():

    with open("final_model.pkl","rb") as f:
        model = pickle.load(f)

    with open("numeric_imputer.pkl","rb") as f:
        numeric_imputer = pickle.load(f)

    with open("categorical_imputer.pkl","rb") as f:
        categorical_imputer = pickle.load(f)

    with open("label_encoders.pkl","rb") as f:
        label_encoders = pickle.load(f)
    with open("hotel_encoder.pkl","rb") as f:
        hotel_encoder = pickle.load(f)

    with open("month_encoder.pkl","rb") as f:
        month_encoder = pickle.load(f)

    with open("country_frequency.pkl","rb") as f:
        country_frequency = pickle.load(f)

    with open("scaler.pkl","rb") as f:
        scaler = pickle.load(f)

    with open("best_features.pkl","rb") as f:
        best_features = pickle.load(f)

    return (
        model,
        numeric_imputer,
        categorical_imputer,
        label_encoders,

        hotel_encoder,
        month_encoder,
        country_frequency,
        scaler,
        best_features
    )


(
    final_model,
    numeric_imputer,
    categorical_imputer,
    label_encoders,
    hotel_encoder,
    month_encoder,
    country_frequency,
    scaler,
    best_features
) = load_files()

# ============================================================
# User Inputs
# ============================================================

st.header("🏨 Hotel Information")

col1, col2 = st.columns(2)

with col1:

    hotel = st.selectbox(
        "Hotel Type",
        [
            "City Hotel",
            "Resort Hotel"
        ]
    )

    meal = st.selectbox(
        "Meal Plan",
        [
            "BB",          # Bed & Breakfast (~77% of bookings)
            "HB",          # Half Board
            "FB",          # Full Board
            "SC",          # Self Catering
            "Undefined"
        ],
        help="BB (Bed & Breakfast) is the most common plan in the training data."
    )

    market_segment = st.selectbox(
        "Market Segment",
        [
            "Online TA",
            "Offline TA/TO",
            "Direct",
            "Corporate",
            "Groups",
            "Complementary",
            "Aviation",
            "Undefined"
        ]
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        [
            "TA/TO",
            "Direct",
            "Corporate",
            "GDS",
            "Undefined"
        ]
    )

    deposit_type = st.selectbox(
        "Deposit Type",
        [
            "No Deposit",
            "Refundable",
            "Non Refund"
        ]
    )

with col2:

    customer_type = st.selectbox(
        "Customer Type",
        [
            "Transient",
            "Transient-Party",
            "Contract",
            "Group"
        ]
    )

    reserved_room_type = st.selectbox(
        "Reserved Room Type",
        [
            "A","B","C","D","E",
            "F","G","H","L","P"
        ]
    )

    assigned_room_type = st.selectbox(
        "Assigned Room Type",
        [
            "A","B","C","D","E","F",
            "G","H","I","K","L","P"
        ]
    )

    country = st.text_input(
        "Country",
        value="PRT",
        help="ISO-3 country code as seen in training, e.g. PRT, GBR, USA, ESP."
    )

st.header("📅 Booking Information")

col1, col2 = st.columns(2)

with col1:

    arrival_date_month = st.selectbox(
        "Arrival Month",
        [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ],
        index=6,  # default to July, the seasonal high-cancellation month found in EDA
        help="Cancellation rates peak in July/August per the project's EDA — this now "
             "genuinely affects the prediction instead of being fixed to January."
    )

    lead_time = st.number_input(
        "Lead Time",
        0,
        800,
        30
    )

    arrival_date_year = st.selectbox(
        "Arrival Year",
        [
            2015,
            2016,
            2017
        ]
    )

    arrival_date_week_number = st.slider(
        "Arrival Week Number",
        1,
        53,
        25
    )

    arrival_date_day_of_month = st.slider(
        "Arrival Day",
        1,
        31,
        15
    )

with col2:

    stays_in_weekend_nights = st.slider(
        "Weekend Nights",
        0,
        20,
        1
    )

    stays_in_week_nights = st.slider(
        "Week Nights",
        0,
        60,
        2
    )

    previous_cancellations = st.slider(
        "Previous Cancellations",
        0,
        30,
        0
    )

    booking_changes = st.slider(
        "Booking Changes",
        0,
        20,
        0
    )

    adr = st.number_input(
        "Average Daily Rate (ADR)",
        0.0,
        6000.0,
        100.0
    )

st.header("📌 Additional Information")

col1, col2, col3 = st.columns(3)

with col1:

    required_car_parking_spaces = st.slider(
        "Parking Spaces",
        0,
        10,
        0
    )

with col2:

    total_of_special_requests = st.slider(
        "Special Requests",
        0,
        5,
        0
    )

with col3:

    agent = st.number_input(
        "Agent ID",
        0,
        600,
        0
    )
# ============================================================
# Create Full DataFrame (28 Features)
# ============================================================

data = pd.DataFrame({

    # -------- Categorical --------

    "hotel": [hotel],

    "meal": [meal],                              # Now a real user input

    "country": [country],

    "market_segment": [market_segment],

    "distribution_channel": [distribution_channel],

    "reserved_room_type": [reserved_room_type],

    "assigned_room_type": [assigned_room_type],

    "deposit_type": [deposit_type],

    "customer_type": [customer_type],

    "arrival_date_month": [arrival_date_month],  # Now a real user input


    # -------- Numerical --------

    "lead_time": [lead_time],

    "arrival_date_year": [arrival_date_year],

    "arrival_date_week_number": [arrival_date_week_number],

    "arrival_date_day_of_month": [arrival_date_day_of_month],

    "stays_in_weekend_nights": [stays_in_weekend_nights],

    "stays_in_week_nights": [stays_in_week_nights],

    "adults": [2],                        # Default

    "children": [0],                      # Default

    "babies": [0],                        # Default

    "is_repeated_guest": [0],             # Default

    "previous_cancellations": [previous_cancellations],

    "previous_bookings_not_canceled": [0],# Default

    "booking_changes": [booking_changes],

    "agent": [agent],

    "days_in_waiting_list": [0],          # Default

    "adr": [adr],

    "required_car_parking_spaces": [required_car_parking_spaces],

    "total_of_special_requests": [total_of_special_requests]

})

    # Missing Values
numeric_cols = [

    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "agent",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests"

]
categorical_cols = [

    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type"

]
data[numeric_cols] = numeric_imputer.transform(data[numeric_cols])

data[categorical_cols] = categorical_imputer.transform(data[categorical_cols])
    # Encoding

data["hotel"] = hotel_encoder.transform(data["hotel"])

data["arrival_date_month"] = month_encoder.transform(
    data[["arrival_date_month"]]
)

data["country"] = data["country"].map(country_frequency).fillna(0)

label_columns = [

    "meal",

    "market_segment",

    "distribution_channel",

    "reserved_room_type",

    "assigned_room_type",

    "deposit_type",

    "customer_type"

]

for col in label_columns:

    data[col] = label_encoders[col].transform(
        data[col].astype(str)
    )
    # Scaling

data[numeric_cols] = scaler.transform(
    data[numeric_cols]
)
    # Feature Selection

data = data[best_features]
    # ============================================================
    # Prediction
    # ============================================================

prediction = final_model.predict(data)[0]

probability = final_model.predict_proba(data)[0]
    # ============================================================
    # Result
    # ============================================================

confidence = probability.max() * 100

st.divider()

st.subheader("Prediction Result")

if prediction == 1:
    st.error("❌ Booking Will Be Cancelled")


else:
    st.success("✅ Booking Will Not Be Cancelled")

st.metric(
    "Confidence",
    f"{confidence:.2f}%"
)

st.progress(float(confidence) / 100)

st.write("Prediction Probabilities")

st.write({

    "Not Cancelled": f"{probability[0]*100:.2f}%",

    "Cancelled": f"{probability[1]*100:.2f}%"

})
