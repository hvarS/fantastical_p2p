import streamlit as st
from PIL import Image
import configparser
import os
import replicate
import requests
from io import BytesIO

config = configparser.ConfigParser()
config.read('key.ini')


os.environ["REPLICATE_API_TOKEN"] = config['SECRET KEY']['REPLICATE_API_TOKEN']

# Function to load and display image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to generate styled image
def generate_style_image(image_filePath, style):
    img_to_use = open(image_filePath, "rb")
    prompt = f"turn the person/object in this image to a character from {style} Universe"
    input = {
        "image": img_to_use,
        "prompt": prompt
    } 
    output = replicate.run(
        "timothybrooks/instruct-pix2pix:30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f",
        input=input
    )

    response = requests.get(output[0])
    img = Image.open(BytesIO(response.content))
    return img

# Main function to run the app
def main():
    _, _, col3, _ = st.columns([1, 1, 4, 1])
    col3.title("FlickFantasy")
    
    # Upload button
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    # Style dropdown
    style = st.selectbox("Choose Style", ["Avatar", "Harry Potter", "Pirates of the Carribean", "Lord of the Rings", "Star Wars", "Marvel"])

    
    # Custom prompt button
    if st.button("Custom Prompt"):
        custom_prompt = st.text_area("Enter your custom prompt")
        if custom_prompt:
            # Here you can implement the logic to handle the custom prompt
            st.success("Custom prompt sent successfully!")

    # Central alignment for buttons
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    # State variable to track button click
    magic_button_clicked = col3.button("Magic 🪄")

    # Display image section
    if uploaded_file is not None:
        with open(os.path.join("temp_dir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Get the file path
        file_path = os.path.join("temp_dir", uploaded_file.name)

        # Load and display uploaded image
        original_image = load_image(uploaded_file)
        col4, col5 = st.columns(2)
        col4.header("Original Image")
        col4.image(original_image, use_column_width=True)

        # Generate and display styled image
        if magic_button_clicked:
            styled_image = generate_style_image(file_path, style)
            col5.header("Styled Image")
            col5.image(styled_image, use_column_width=True)
    elif uploaded_file is None:
        st.warning("Please upload an image.")
    elif not magic_button_clicked:
        st.info("Click 'Magic 🪄' to generate styled image.")
    
if __name__ == "__main__":
    main()