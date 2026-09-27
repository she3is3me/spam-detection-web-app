import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)


# -----------------------------
# Load dataset
# -----------------------------

data = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

# Remove duplicate messages
data.drop_duplicates(inplace=True)

# Keep only required columns
data = data[["v1", "v2"]]

# Rename columns
data.columns = ["category", "message"]


# -----------------------------
# Prepare data
# -----------------------------

message = data["message"]
category = data["category"]


# Split data into training and testing
message_train, message_test, category_train, category_test = train_test_split(
    message,
    category,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Convert text to numbers
# -----------------------------

cv = CountVectorizer(stop_words="english")

features = cv.fit_transform(message_train)


# -----------------------------
# Train model
# -----------------------------

model = MultinomialNB()

model.fit(features, category_train)


# -----------------------------
# Test model
# -----------------------------

features_test = cv.transform(message_test)

accuracy = model.score(
    features_test,
    category_test
)


# -----------------------------
# Streamlit interface
# -----------------------------

st.title("📱 SMS Spam Detector by SHAISTA")

st.write(
    "Enter an SMS message below to check whether "
    "it is Spam or Not Spam."
)

st.info(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)


# Text input
user_message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a lottery!"
)


# Prediction button
if st.button("🔍 Check Message"):

    if user_message.strip() == "":
        
        st.warning("⚠️ Please enter a message.")

    else:

        # Convert message to numerical features
        message_features = cv.transform([user_message])

        # Make prediction
        prediction = model.predict(message_features)[0]

        # Display result
        if prediction == "spam":

            st.error("🚨 This message is SPAM!")

        else:

            st.success("✅ This message is NOT SPAM!")