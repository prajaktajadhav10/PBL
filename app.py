import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Wiener deconvolution function (applied to each channel of the image)
def wiener_deconvolution(img, psf, balance=0.1):
    img_ft = np.fft.fft2(img)
    psf_ft = np.fft.fft2(psf, s=img.shape)
    psf_ft_conj = np.conj(psf_ft)
    ratio = psf_ft_conj / (np.abs(psf_ft) ** 2 + balance)
    deblurred_ft = img_ft * ratio
    deblurred_img = np.fft.ifft2(deblurred_ft)
    deblurred_img = np.abs(deblurred_img)
    return np.uint8(np.clip(deblurred_img * 255, 0, 255))

# Streamlit UI Enhancements
st.set_page_config(page_title='Image Deblurring App', layout='wide', page_icon='📸')
st.title('📷 Image Deblurring')
st.write('Upload an image to apply Wiener deconvolution and restore clarity.')

# Sidebar for configuration
st.sidebar.header('🔧 Adjust Parameters')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
kernel_size = st.sidebar.slider('Kernel size', 3, 15, 5, step=2)
sigma = st.sidebar.slider('Sigma (Standard Deviation)', 0.1, 5.0, 1.0)
balance = st.sidebar.slider('Balance (Noise Reduction)', 0.01, 1.0, 0.1)

if uploaded_file is not None:
    # Read and display original image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    original_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Split into channels
    blue_channel, green_channel, red_channel = cv2.split(original_img)

    # Normalize channels
    blue_channel_float = blue_channel.astype(np.float32) / 255.0
    green_channel_float = green_channel.astype(np.float32) / 255.0
    red_channel_float = red_channel.astype(np.float32) / 255.0

    # PSF (Gaussian kernel)
    psf = np.outer(cv2.getGaussianKernel(kernel_size, sigma), cv2.getGaussianKernel(kernel_size, sigma))

    # Apply Wiener filter to each channel
    deblurred_blue = wiener_deconvolution(blue_channel_float, psf, balance)
    deblurred_green = wiener_deconvolution(green_channel_float, psf, balance)
    deblurred_red = wiener_deconvolution(red_channel_float, psf, balance)

    # Merge deblurred channels
    deblurred_color_img = cv2.merge([deblurred_blue, deblurred_green, deblurred_red])

    # Convert to PIL image for display
    deblurred_pil = Image.fromarray(cv2.cvtColor(deblurred_color_img, cv2.COLOR_BGR2RGB))

    # Display side-by-side comparison using dynamic columns
    st.write("### 🖼️ Comparison: Original vs Deblurred")
    col1, col2 = st.columns([1, 1])
    col1.image(image, caption='🟢 Original Image', use_column_width=True)
    col2.image(deblurred_pil, caption='🔵 Deblurred Image', use_column_width=True)

    # Provide download option with proper image conversion
    import io
    img_byte_arr = io.BytesIO()
    deblurred_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    st.sidebar.download_button('⬇️ Download Deblurred Image', data=img_byte_arr, file_name='deblurred_image.png', mime='image/png')

    st.success('🎉 Image deblurred successfully! Adjust parameters if needed.')
else:
    st.info('Please upload an image to proceed.')
