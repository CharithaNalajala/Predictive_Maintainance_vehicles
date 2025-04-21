import streamlit as st
import pandas as pd
import pickle
from streamlit_extras.metric_cards import style_metric_cards


st.set_page_config(
    page_title="Vehicle Failure Predictor", 
    layout="wide",
    page_icon="🚗"
)


st.markdown("""
    <style>
    /* Main content area - light background */
    .main {
        background-color: #ffffff;
    }
    
    /* Sidebar - darker but not too dark */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #3a4a6b, #4a5d8c) !important;
        color: white !important;
    }
    
    /* Better contrast for sidebar text */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stSlider p {
        color: #f8f9fa !important;
    }
    
    /* Header styling */
    h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(to right, #1a2a6c, #b21f1f, #fdbb2d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subheader styling */
    h2 {
        font-size: 1.6rem !important;
        color: #2c3e50 !important;
        border-bottom: 2px solid #4a6491;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(to right, #1a2a6c, #4a6491);
        color: white;
        font-size: 1.1rem;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1.5rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(to right, #4a6491, #1a2a6c);
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Vehicle Failure Prediction")
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; color: #4a5568;">
            Enter your vehicle's parameters to get a failure prediction
        </p>
    </div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 style="color: white !important; font-size: 1.8rem !important;">🔧 Vehicle Parameters</h1>
            <p style="color: #e2e8f0; font-size: 0.9rem;">
                Adjust the sliders to match your vehicle's current readings
            </p>
        </div>
    """, unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
    
    with col1:
        engine_rpm = st.slider("Engine RPM", 500, 5000, 1500, step=50)
        lub_oil_pressure = st.slider("Lub Oil Pressure (bar)", 0.0, 10.0, 3.5, step=0.1)
        fuel_pressure = st.slider("Fuel Pressure (bar)", 0.0, 200.0, 100.0, step=1.0)
        coolant_pressure = st.slider("Coolant Pressure (bar)", 0.0, 5.0, 1.5, step=0.1)
        lub_oil_temp = st.slider("Lub Oil Temp (°C)", 20.0, 150.0, 90.0, step=0.5)
    
    with col2:
        coolant_temp = st.slider("Coolant Temp (°C)", 20.0, 150.0, 85.0, step=0.5)
        tire_pressure = st.slider("Tire Pressure (psi)", 20.0, 40.0, 32.0, step=0.5)
        battery_voltage = st.slider("Battery Voltage (V)", 10.0, 15.0, 12.6, step=0.1)
        brake_pad_thickness = st.slider("Brake Pad Thickness (mm)", 1.0, 12.0, 6.0, step=0.1)
        transmission_temp = st.slider("Transmission Temp (°C)", 30.0, 130.0, 85.0, step=0.5)

# ---------------------------
# CREATE INPUT DATAFRAME
# ---------------------------
input_data = pd.DataFrame([[
    engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure,
    lub_oil_temp, coolant_temp, tire_pressure, battery_voltage,
    brake_pad_thickness, transmission_temp
]], columns=[
    "Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure",
    "lub oil temp", "Coolant temp", "tire_pressure", "battery_voltage",
    "brake_pad_thickness", "transmission_temp"
])

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import joblib
        return joblib.load("rf_model.joblib")
    except:
        try:
            with open("rf_model.pkl", "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

model = load_model()


st.subheader("📋 Input Summary")
st.dataframe(input_data.style.format("{:.1f}"), use_container_width=True)

# Prediction button
if st.button("🔍 Predict Vehicle Condition"):
    if model is None:
        st.error("Model failed to load. Please check the model file.")
    else:
        with st.spinner("Analyzing vehicle data..."):
            # Get prediction
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][prediction]
            
            # Store in session state
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.prediction_made = True

# Show results if prediction was made
if 'prediction_made' in st.session_state and st.session_state.prediction_made:
    if st.session_state.prediction == 1:
        st.markdown(
            f"""
            <div class="prediction-box">
                <div style="font-size: 2rem; margin-bottom: 1rem; color: #d32f2f;">⚠️</div>
                <h3 style="color: #d32f2f; margin-bottom: 0.5rem;">Potential Failure Detected</h3>
                <p>Our analysis indicates your vehicle may require immediate attention.</p>
                <div style="margin-top: 1rem; background-color: #ffebee; border-radius: 8px; padding: 10px;">
                    <span style="font-weight: 700;">Confidence: </span>
                    <span style="color: #d32f2f; font-weight: 700;">{st.session_state.confidence:.1%}</span>
                </div>
                <p style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                    Recommendation: Schedule a diagnostic check with your mechanic.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="prediction-box">
                <div style="font-size: 2rem; margin-bottom: 1rem; color: #388e3c;">✅</div>
                <h3 style="color: #388e3c; margin-bottom: 0.5rem;">Vehicle in Good Condition</h3>
                <p>All systems appear to be operating within normal parameters.</p>
                <div style="margin-top: 1rem; background-color: #e8f5e9; border-radius: 8px; padding: 10px;">
                    <span style="font-weight: 700;">Confidence: </span>
                    <span style="color: #388e3c; font-weight: 700;">{st.session_state.confidence:.1%}</span>
                </div>
                <p style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                    Recommendation: Continue regular maintenance schedule.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


with st.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    **Vehicle Failure Predictor**  
    This application uses a Random Forest Classifier trained on historical vehicle sensor data to predict whether a vehicle is likely to fail.
    
    **How to Use:**  
    1. Adjust the vehicle parameters in the sidebar  
    2. Click "Predict Vehicle Condition"  
    3. View the prediction result with confidence percentage  
    
    **Input Features:**  
    - Engine RPM  
    - Lub Oil Pressure  
    - Fuel Pressure  
    - Coolant Pressure  
    - Lub Oil Temp  
    - Coolant Temp  
    - Tire Pressure  
    - Battery Voltage  
    - Brake Pad Thickness  
    - Transmission Temp  
    
    **Disclaimer:**  
    This tool provides predictive analysis only and should not replace professional mechanical inspection.  
    Always consult with a certified mechanic for vehicle repairs.
    """)

st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: #666; font-size: 0.8rem;">
        <hr style="border-top: 1px solid #ddd; margin-bottom: 1rem;">
        <p>Vehicle Failure Predictor v1.0 • Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)