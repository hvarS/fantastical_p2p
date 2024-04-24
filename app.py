import streamlit as st
from PIL import Image

# Function to load and display image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to generate styled image
def generate_style_image(image_file, style):
    # Here, you would implement the logic to generate the styled image
    # For demonstration purposes, I'm just returning the original image
    return image_file

# Main function to run the app
def main():
    _, _, col3, _ = st.columns([1, 1, 4, 1])
    col3.title("FlickFantasy")

    # Upload button
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Style dropdown
    style = st.selectbox("Choose Style", ["Style 1", "Style 2", "Style 3"])

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
    if uploaded_file is not None and magic_button_clicked:
        # Load and display uploaded image
        original_image = load_image(uploaded_file)
        col4, col5 = st.columns(2)
        col4.header("Original Image")
        col4.image(original_image, use_column_width=True)

        # Generate and display styled image
        styled_image = generate_style_image(original_image, style)
        col5.header("Styled Image")
        col5.image(styled_image, use_column_width=True)
    elif uploaded_file is None:
        st.warning("Please upload an image.")
    elif not magic_button_clicked:
        st.info("Click 'Magic 🪄' to generate styled image.")
    
if __name__ == "__main__":
    main()