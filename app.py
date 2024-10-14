import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore

model = load_model('model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))  
    image_array = np.array(image) / 255.0  
    return np.expand_dims(image_array, axis=0)  

def main():
    st.set_page_config(layout='wide')
    html_temp = """
        <div style="background-color:#025246 ;padding:2px">
        <h1 style="color:white;text-align:center;">Diabetic Retinopathy Detection</h1>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    html_temp2 = """
        <div style="background-color:#025246 ;padding:0px">
       <h2 style = "color:white;text-align:center;">Navigation</h2></div>"""
    
    st.sidebar.markdown(html_temp2,unsafe_allow_html=True)
    options = st.sidebar.radio("Select a page:", ["Home", "Upload Image", "About"])

    if options == "Home":
        st.write("") 
        st.markdown("<h2 style='text-align: center;'>Welcome to the Diabetic Retinopathy Image Classification!</h2>", unsafe_allow_html=True)
        
        st.write("This application helps predict whether an uploaded retinal image indicates diabetic retinopathy. "
                "Diabetic retinopathy is a serious eye condition that can lead to vision loss if not detected early.")
        
        st.write("### How to Use the App:")
        st.write("- Navigate to the **'Upload Image'** page in the sidebar to submit a retinal image.")
        st.write("- After uploading, the app will classify the image as either **retinopathy positive** or **retinopathy negative**.")
        st.write("- The classification results will include a probability score, indicating the model's confidence in its prediction.")
        
        st.write("### Why Diabetic Retinopathy Screening Matters:")
        st.write("Early detection of diabetic retinopathy is crucial for preserving vision in individuals with diabetes. "
                "Regular screening can help identify changes in the retina before they lead to more serious complications.")
        
        st.write("### Sample Images:")
        st.write("Below are examples of retinal images for reference:")
        
        # Display two sample images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image("sample_positive.jpg", caption="Positive for Diabetic Retinopathy", use_column_width=True)
        
        with col2:
            st.image("sample_negative.jpg", caption="Negative for Diabetic Retinopathy", use_column_width=True)

        st.write("These images illustrate the differences in retinal appearance for diabetic retinopathy. "
                "Use them as a reference while uploading your images.")
        
        st.write("Feel free to explore the app and take the first step towards understanding diabetic retinopathy!")

    elif options == "Upload Image":
        st.subheader("Upload your image for classification")
        st.write("Please upload an image of the retina. The model will predict whether signs of diabetic retinopathy are present.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button("Classify"):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                class_label = "Positive" if prediction[0][0] > 0.5 else "Negative"
                st.success(f'The image is classified as: **{class_label}**')
                
                st.write("**Probability:**", prediction[0][0])
                st.write("A positive classification indicates the presence of diabetic retinopathy, while a negative classification indicates no signs.")

    elif options == "About":
        
        st.subheader("About this App")
        st.write("This app classifies retinal images to determine if they show signs of diabetic retinopathy.")
        
        st.write("### Overview")
        st.write("Diabetic retinopathy is a serious complication of diabetes that can lead to blindness if not detected and treated early. "
                "This application aims to aid in the early detection of diabetic retinopathy by leveraging machine learning techniques.")
        
        st.write("### Technology Stack")
        st.write("The app is developed using:")
        st.write("- **Streamlit**: A powerful framework for building web applications for machine learning and data science.")
        st.write("- **TensorFlow**: An open-source deep learning framework used for training the underlying model.")
        st.write("- **Keras**: A high-level neural networks API, built on top of TensorFlow, simplifying the model-building process.")

        st.write("### How It Works")
        st.write("1. **Image Upload**: Users can upload retinal images in JPG, JPEG, or PNG formats."
                " The app accepts images that represent the retina from diabetic patients.")
        st.write("2. **Image Processing**: Uploaded images are preprocessed to ensure they meet the model's input requirements.")
        st.write("3. **Prediction**: The model analyzes the image and predicts whether signs of diabetic retinopathy are present.")
        st.write("4. **Results**: Users receive feedback on the classification, along with a probability score indicating the model's confidence.")

        st.write("### Importance of Early Detection")
        st.write("Early detection of diabetic retinopathy can prevent vision loss and improve the quality of life for patients. "
                "Regular screening is essential for individuals with diabetes. This app serves as a supplementary tool to help identify those at risk.")


        st.write("### Disclaimer")
        st.write("While this application provides valuable insights, it is not a substitute for professional medical advice, diagnosis, or treatment. "
                "Always consult a qualified healthcare provider for any questions regarding a medical condition.")


if __name__ == "__main__":
    main()
