# Stock Price Prediction using LSTM and Artificial Rabbit Optimization (ARO)

## 1. Project Overview
This project predicts stock prices using the deep learning technique Long Short-Term Memory (LSTM) neural network. To optimize the hyperparameters, the Artificial Rabbit Optimization (ARO) algorithm is utilized. A user-friendly graphical interface (GUI) is built using Streamlit, where users can upload their CSV file and view the stock data predictions.

## 2. Project Features
- **Data Preprocessing**: Cleans and scales stock data to ensure proper model training.
- **LSTM Model**: A specialized neural network designed for time series data.
- **Artificial Rabbit Optimization**: A novel nature-inspired algorithm for hyperparameter tuning.
- **Streamlit GUI**: An interactive interface where users can upload a file and get real-time results.
- **Evaluation Metrics**: Uses MAE, MSE, and RMSE to measure model performance.
- **Visualization**: Plots graphs comparing actual and predicted stock prices.

## 3. Project Structure
| Folder/File | Description |
|---|---|
| `app.py` | Main script for the Streamlit GUI application |
| `data/` | Sample stock CSV files |
| `models/lstm_model.py` | LSTM neural network code |
| `optimization/aro_optimizer.py` | Artificial Rabbit Optimization algorithm |
| `utils/preprocessing.py` | Data cleaning and scaling functions |
| `utils/metrics.py` | Performance metrics calculation |
| `utils/plot.py` | Functions for generating plots |
| `requirements.txt` | List of required Python packages |
| `README.md` | Project documentation file |

## 4. Installation Guide
Install Python 3.7 or a newer version.

Create a virtual environment (optional, but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 5. How to Run
Open your terminal/command prompt and navigate to the project folder.

Run the Streamlit app:

```bash
streamlit run app.py
```
The Streamlit app will open in your web browser.

Upload your stock data CSV file.

The model will perform hyperparameter tuning, train, and display prediction results on the screen. Evaluation metrics and graphs will also be displayed.

## 6. CSV File Format Requirements
The CSV file should contain the following columns:

- `Date` (optional)
- `Close/Last` (or whatever the closing price column is named)
- `Volume`, `Open`, `High`, `Low` (optional)

Example format:

| Date | Close/Last | Volume | Open | High | Low |
|---|---|---|---|---|---|
| 2020-01-02 | $296.24 | 33870100 | 296.24 | 300.60 | 295.19 |

## 7. Notes and Tips
- The Date column is not used for model training and is ignored during preprocessing.
- The code automatically handles the presence of '$' signs or commas within the Close price column.
- Hyperparameter tuning is set to 5 iterations by default — more iterations will improve accuracy but will also increase processing time.
- You can customize the LSTM model and ARO parameters in `models/lstm_model.py` and `optimization/aro_optimizer.py`.

## 8. Contribution
You can fork this project to improve it. Feel free to report issues or submit pull requests.

## 10. Contact Information
Email: nailakhani5457@gmail.com
