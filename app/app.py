from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Folder to store processed CSV files
PROCESSED_FOLDER = 'processed_files'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    df = pd.read_csv(file, nrows=1000)

    # === Dummy Fraud Prediction (replace with your model) ===
    df['isFraud_Predicted'] = df['isFraud'] if 'isFraud' in df.columns else 0

    # Save processed CSV
    processed_filename = 'processed_transactions.csv'
    df.to_csv(f'{PROCESSED_FOLDER}/{processed_filename}', index=False)

    # Metrics (replace with your model metrics)
    accuracy = 1.0
    report = df.describe().to_html(classes='table table-striped table-bordered')

    # --- Generate Plots ---
    numeric_df = df.select_dtypes(include='number')
    corr_html = pio.to_html(px.imshow(numeric_df.corr(), text_auto=True, title="Correlation Heatmap"), full_html=False) \
                if not numeric_df.empty else "<p>No numeric columns available for correlation heatmap.</p>"

    fraud_html = pio.to_html(px.histogram(df, x='isFraud', text_auto=True, title="Fraud Distribution"), full_html=False) \
                 if 'isFraud' in df.columns else "<p>No 'isFraud' column available.</p>"

    scatter_html = pio.to_html(px.scatter(df, x='oldbalanceOrg', y='amount', color='isFraud', title="Amount vs Old Balance"), full_html=False) \
                   if 'amount' in df.columns and 'oldbalanceOrg' in df.columns else "<p>Required columns not found for scatter plot.</p>"

    return render_template(
        'results.html',
        accuracy=accuracy,
        report=report,
        processed_file_name=processed_filename,
        data=df.head().to_html(classes='table table-striped table-bordered'),
        corr_html=corr_html,
        fraud_html=fraud_html,
        scatter_html=scatter_html
    )

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(directory=PROCESSED_FOLDER, path=filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
