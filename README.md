📈 Advanced Stock Price Forecast using a Multivariate LSTM Model
<div align="center">
An end-to-end deep learning application that forecasts stock prices using a multivariate Long Short-Term Memory (LSTM) model, enhanced with a rich set of technical indicators and an interactive web dashboard.
</div>
🌟 Overview
This project moves beyond simple price prediction by leveraging a multivariate time-series approach. The model analyzes 60 days of historical data, including 8 distinct features, to generate accurate multi-day forecasts. The entire pipeline, from data collection to visualization, is presented in a polished and user-friendly Streamlit application that includes detailed company information and performance metrics.
✨ Key Highlights
🎯 High Accuracy: A deep, 3-layer LSTM architecture for precise, context-aware predictions.
🧠 Multivariate Analysis: Incorporates 8 features (Close Price + 7 technical indicators) for a richer market view.
📊 Real-time Data: Live stock data from the Yahoo Finance API.
🖥️ Interactive Dashboard: An elegant Streamlit web interface with a searchable stock list, company profiles, and dynamic forecasting.
📈 Visual Analytics: Comprehensive charts showing historical backtesting alongside multi-day future forecasts.
🚀 Features
Feature
Description
🏢 Company Profiles
Displays company summary and key financial metrics (Market Cap, 52-Week High/Low).
🔍 Searchable Stock List
A searchable dropdown for all stocks on the NSE for easy selection.
⏳ Multivariate Analysis
Analyzes 60-day patterns across 8 features for next-day predictions.
🔮 Multi-Day Forecasting
Iteratively forecasts prices for a user-defined period (1-15 days).
📊 Performance Metrics
In-depth backtesting analysis with Accuracy, MAPE, and R² Score.
💾 Model Persistence
A dedicated training script saves the final enhanced_model.keras for use by the app.

📂 Project Structure
LSTM-stock-price-prediction/
├── 🌐 app.py              # The main Streamlit web application
├── 🧠 train_model.py       # Script to train the enhanced LSTM model
├── 📋 requirements.txt    # Python dependencies for the project
├── 📜 get_tickers.py       # Utility script to fetch NSE stock symbols
├── 🗂️ nse_tickers.csv     # Data file for the searchable stock list
├── 💾 enhanced_model.keras # The final trained model (auto-created)
└── 📚 README.md           # Project documentation


🛠️ Installation & Setup
Prerequisites
Python 3.11 or a compatible version
Pip package manager
Git
Quick Start
# 1️⃣ Clone the repository
git clone <your-github-repository-url>
cd LSTM-stock-price-prediction

# 2️⃣ Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# 3️⃣ Install all dependencies
pip install -r requirements.txt

# 4️⃣ Generate the stock ticker file
python get_tickers.py


🎯 Usage
1. Training the Model (One-Time Step)
Run the training script from your terminal. This will create the enhanced_model.keras file.
python train_model.py


What happens during training:
📥 Downloads historical stock data for the target stock.
🔧 Calculates 7 additional technical indicators.
🧠 Trains the deep, 3-layer LSTM model.
💾 Saves the final model as enhanced_model.keras.
2. Running the Web Application
Launch the Streamlit dashboard.
streamlit run app.py


Features of the web app:
🔍 Search for and select any NSE stock.
🎚️ Adjust the number of days to forecast.
📊 View the company profile and key metrics.
📈 Analyze the interactive chart with historical backtesting and future predictions.
🧠 Technical Details
Model Architecture
Network Type: Long Short-Term Memory (LSTM)
Input Window: 60 trading days
Prediction Horizon: Multi-day (iterative)
Optimizer: Adam
Data Processing
Source: Yahoo Finance API via yfinance
Features (8 total): Close, SMA, EMA, RSI, Bollinger Bands (High/Low), MACD, Stochastic Oscillator.
Scaling: MinMax normalization for all features.
Split: 80% training, 20% testing.
Performance Metrics
Model Accuracy (100 - MAPE)
Mean Absolute Percentage Error (MAPE)
R² Score (Coefficient of Determination)
📦 Dependencies
TensorFlow/Keras: For building and training the deep learning model.
Streamlit: For the interactive web application framework.
yfinance: For real-time stock data retrieval.
TA: For calculating technical analysis indicators.
nsepy: For fetching the list of NSE stock tickers.
Pandas & NumPy: For data manipulation.
Plotly: For interactive data visualization.
scikit-learn: For data preprocessing and metrics.
🔮 Future Enhancements
[ ] 🤖 Train a generalized base model on multiple stocks and fine-tune for specifics.
[ ] 📰 Integrate sentiment analysis from financial news.
[ ] 🧠 Explore advanced architectures like Transformers.
[ ] ☁️ Deploy the application to a cloud service for public access.
