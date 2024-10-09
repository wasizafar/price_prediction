import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import multiprocessing
import qrcode
from PIL import Image
import io, os
import base64
import openpyxl

# Load the background image
background_image_path = "img.jpg"
with open(background_image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Define the HTML template with inline CSS for background and button
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asset Development</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-image: url(data:image/jpg;base64,{encoded_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            height: 100vh;
            width: 100vw;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .container {{
            width: 60%;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
        }}
        .container h1 {{
            font-size: 24px;
            margin-bottom: 20px;
        }}
        .container button {{
            background-color: #ff4747;
            color: #fff;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sync Synapse Bill Generator</h1>
    </div>
</body>
</html>
"""

# Use the html_template to render in the Streamlit app
st.markdown(html_template, unsafe_allow_html=True)

# Load the saved model and preprocessing objects
model = load_model('discounted_price_model.h5')
scaler = joblib.load('scaler.pkl')
tfidf_vectorizer_name = joblib.load('tfidf_vectorizer_name.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load the dataset for brand and product_category_tree lookup
data_path = 'flipkart_com-ecommerce_sample.csv'
df = pd.read_csv(data_path)

# Fill missing values in relevant columns for lookup
df['product_name'].fillna('Unknown', inplace=True)
df['brand'].fillna('Unknown', inplace=True)
df['product_category_tree'] = df['product_category_tree'].str.split('>>').str[0].str.strip().fillna('Unknown')

# Function for processing discount predictions in parallel
def process_predictions(product_details, return_list, index):
    predictions = []
    for product in product_details:
        # Preprocessing the inputs
        product_name_tfidf = tfidf_vectorizer_name.transform([product['product_name']]).toarray()

        # Encode the brand and category tree using label encoders
        product_category_tree_encoded = label_encoders['product_category_tree'].transform([product['selected_category']])
        brand_encoded = label_encoders['brand'].transform([product['selected_brand']])

        # Combine the features (TF-IDF and numeric features)
        X_combined = np.concatenate([ 
            product_name_tfidf,
            np.array([[product['retail_price'], product_category_tree_encoded[0], product['overall_rating'], brand_encoded[0]]])
        ], axis=1)

        # Scale the combined features
        X_scaled = scaler.transform(X_combined)

        # Predict the discounted price
        predicted_discounted_price = model.predict(X_scaled)
        discounted_price = predicted_discounted_price[0][0]

        # Calculate the price after applying 18% GST
        gst = 0.18 * discounted_price
        final_price_with_gst = discounted_price + gst

        predictions.append({
            'product_name': product['product_name'],
            'retail_price': product['retail_price'],
            'discounted_price': discounted_price,
            'gst': gst,
            'final_price_with_gst': final_price_with_gst,
            'selected_brand': product['selected_brand'],
            'overall_rating': product['overall_rating']
        })
    return_list[index] = predictions

def prediction_page():
    st.title("Multiple Product Discount Prediction")
    st.write("Predict discounts for multiple products.")

    # Input field for customer email
    customer_email = st.text_input('Customer Email')

    # Select multiple products from the dataset
    selected_products_names = st.multiselect("Select Products", df['product_name'].unique())

    if selected_products_names:
        # Filter the dataset based on selected product names
        selected_products_df = df[df['product_name'].isin(selected_products_names)]
        
        # Create a list to hold product data
        product_details = []

        for _, product in selected_products_df.iterrows():
            product_name = product['product_name']
            retail_price = product['retail_price']
            overall_rating = np.random.uniform(3, 5)  # Mock rating between 3 and 5 if not available

            product_details.append({
                'product_name': product_name,
                'retail_price': retail_price,
                'overall_rating': overall_rating,
                'selected_brand': product['brand'],
                'selected_category': product['product_category_tree']
            })

        # Button to predict discounts
        if st.button('Predict Discounts'):
            # Process predictions in multiprocessing
            manager = multiprocessing.Manager()
            return_list = manager.dict()

            processes = []
            for i in range(len(product_details)):
                p = multiprocessing.Process(target=process_predictions, args=([product_details[i]], return_list, i))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()

            predictions = sum(return_list.values(), [])

            # Save predictions and customer email in session state
            st.session_state['predictions'] = predictions
            st.session_state['customer_email'] = customer_email

            st.success("Discounts predicted successfully! Proceed to the next page to generate the bill.")

# Function to append purchase data to an Excel file
def append_purchase_data_to_excel(selected_products, email_address, excel_file='sales_data.xlsx'):
    purchase_data = []
    for product in selected_products:
        purchase_data.append({
            'Product Name': product['product_name'],
            'Brand': product['selected_brand'],
            'Retail Price': product['retail_price'],
            'Discounted Price': product['discounted_price'],
            'GST': product['gst'],
            'Final Price': product['final_price_with_gst'],
            'Email': email_address,
            'Date': pd.Timestamp.now()
        })

    # Convert the purchase data to a DataFrame
    df_purchase = pd.DataFrame(purchase_data)

    # Check if the Excel file already exists
    if os.path.exists(excel_file):
        # Load existing data
        existing_data = pd.read_excel(excel_file, sheet_name='Sales Data')

        # Append the new data to the existing data
        updated_data = pd.concat([existing_data, df_purchase], ignore_index=True)

        # Write the updated data back to the Excel file
        with pd.ExcelWriter(excel_file, mode='w', engine='openpyxl') as writer:
            updated_data.to_excel(writer, sheet_name='Sales Data', index=False)
    else:
        # If the file doesn't exist, create a new one with the new data
        with pd.ExcelWriter(excel_file, mode='w', engine='openpyxl') as writer:
            df_purchase.to_excel(writer, sheet_name='Sales Data', index=False)


# Function to display bill and handle payment
def display_bill_and_payment(selected_products):
    if st.button("View Bill"):
        if not selected_products:
            st.error("Please select at least one product to generate the bill.")
            return

        st.subheader("Bill Summary")
        total_final_price = 0

        for product in selected_products:
            st.write(f"**Product Name**: {product['product_name']}")
            st.write(f"**Brand**: {product['selected_brand']}")
            st.write(f"**Rating**: {product['overall_rating']:.1f}")
            st.write(f"**Retail Price**: ₹ {product['retail_price']:.2f}")
            discount = product['retail_price'] - product['discounted_price']
            st.write(f"**Discount**: ₹ {discount:.2f}")
            st.write(f"**Price After Discount**: ₹ {product['discounted_price']:.2f}")
            st.write(f"**GST (18%)**: ₹ {product['gst']:.2f}")
            st.write(f"**Final Price (with GST)**: ₹ {product['final_price_with_gst']:.2f}")
            st.write("---")

            total_final_price += product['final_price_with_gst']

        st.subheader(f"Total Amount Payable: ₹ {total_final_price:.2f}")

        # Generate QR code for payment
        qr = qrcode.make(f"Pay ₹ {total_final_price:.2f} to XYZ Merchant")
        buf = io.BytesIO()
        qr.save(buf)
        buf.seek(0)

        st.image(buf, caption='Scan to Pay', use_column_width=True)

        if st.button("Confirm Payment"):
            # Append purchase data to Excel
            append_purchase_data_to_excel(selected_products, st.session_state.get('customer_email'))
            st.success("Payment confirmed and data saved!")

# Function for sending emails with the generated bill
def send_email(email, subject, body):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    sender_email = 'mw9403@srmist.edu.in'  # Change to your email
    sender_password = 'Mdwasi9631@'  # Change to your password

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
# Pages
def billing_page():
    selected_products = st.session_state.get('predictions', [])
    if not selected_products:
        st.warning("No product predictions available. Please go to the Prediction Page first.")
    else:
        display_bill_and_payment(selected_products)

# Define the Email Page
def email_page():
    st.title("Send Email with Bill")
    if 'predictions' in st.session_state:
        selected_products = st.session_state['predictions']
        email_address = st.text_input("Enter recipient email address:")
        
        if st.button("Send Bill"):
            email_body = "Here is your bill:\n\n"
            total_final_price = 0
            
            for product in selected_products:
                email_body += f"Product Name: {product['product_name']}\n"
                email_body += f"Brand: {product['selected_brand']}\n"
                email_body += f"Final Price (with GST): ₹ {product['final_price_with_gst']:.2f}\n\n"
                total_final_price += product['final_price_with_gst']
            
            email_body += f"Total Amount Payable: ₹ {total_final_price:.2f}\n"
            
            send_email(email_address, "Your Bill", email_body)
            st.success("Bill sent successfully!")
    else:
        st.error("No predictions to send. Please generate a bill first.")

def sales_analysis_page():
    st.title("Sales Analysis Page")
    excel_file = 'sales_data.xlsx'

    # Check if the Excel file exists
    if os.path.exists(excel_file):
        # Load the sales data from the Excel file
        df_sales = pd.read_excel(excel_file, sheet_name='Sales Data')
        
        # Display the DataFrame to check if the data is loaded correctly
        st.write("Sales Data:")
        st.dataframe(df_sales)

        # Check the column names
        st.write("Column names in the sales data:", df_sales.columns.tolist())

        # Group the data by 'Product Name' and calculate total final price
        try:
            sales_summary = df_sales.groupby('Product Name').agg({'Final Price': 'sum'}).reset_index()
            st.subheader("Sales Summary")
            st.dataframe(sales_summary)
        except KeyError:
            st.error("The 'Product Name' column is missing in the sales data. Please ensure the data is appended correctly.")
    else:
        st.warning("Sales data file not found. Please generate some sales first.")

        
# Page navigation
pages = {
    "Prediction Page": prediction_page,
    "Billing Page": billing_page,
    "Email Page": email_page,
    "Sales Analysis Page": sales_analysis_page
}

# Sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Render the selected page
pages[selected_page]()
