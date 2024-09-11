import pickle
import streamlit as st
import numpy as np

# Custom CSS for background image and styling
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://imgur.com/gallery/laptop-keyboard-9StoKJQ#/t/laptop/Laptop Keyboard.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 15px;
    }
    h1 {
        color: #FF5733;
        text-shadow: 2px 2px 5px #000000;
    }
    h2 {
        color: #3498db;
        text-shadow: 1px 1px 3px #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Importing the model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))

# Title and introduction with background effects
st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>🎯 Enter your laptop configuration to estimate the price</h3>", unsafe_allow_html=True)

# Layout - Use columns to organize inputs in a more attractive way
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('💼 Select Brand', data['Company'].unique())
    laptop_type = st.selectbox('🖥️ Laptop Type', data['TypeName'].unique())
    ram = st.selectbox('🔋 RAM in GB', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    touchscreen = st.selectbox('📱 Touchscreen', ['No', 'Yes'])
    weight = st.number_input('⚖️ Weight of the Laptop (kg)')

with col2:
    ips_panel = st.selectbox('🎨 IPS panel', ['Yes', 'No'])
    screen_size = st.number_input('📏 Screen Size (inches)')
    resolution = st.selectbox('🖼️ Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    cpu = st.selectbox('🧠 CPU', data['CPU Brand'].unique())

# Separate columns for hard drive, SSD, GPU, and OS
col3, col4 = st.columns(2)

with col3:
    hard_drive = st.selectbox('💽 Hard Drive (GB)', [0, 128, 256, 512, 1024, 2048])
    gpu = st.selectbox('🎮 GPU Brand', data['GPU Brand'].unique())

with col4:
    ssd = st.selectbox('💾 SSD (GB)', [0, 8, 128, 256, 512, 1024])
    os = st.selectbox('💻 Operating System', data['Operating System'].unique())

# Prediction button with a progress bar
if st.button('🔮 Predict Laptop Price'):
    # Progress bar
    progress = st.progress(0)
    for percent_complete in range(100):
        progress.progress(percent_complete + 1)

    # Convert categorical inputs into the correct format
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips_panel = 1 if ips_panel == 'Yes' else 0

    # Calculate PPI
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])
    ppi = ((X_resolution ** 2) + (Y_resolution ** 2)) ** 0.5 / screen_size

    # Prepare the query array
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips_panel, ppi, cpu, hard_drive, ssd, gpu, os])

    # Reshape the query to match the expected input format
    query = query.reshape(1, -1)

    # Predict the price
    predicted_price = np.exp(pipe.predict(query)[0])

    # Display the predicted price in a large font
    st.markdown(f"<h2 style='text-align: center;'>The predicted price is ₹{int(predicted_price):,}</h2>", unsafe_allow_html=True)
