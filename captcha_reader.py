import cv2
from PIL import Image
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# If using Windows, set Tesseract path (update the path accordingly)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
image = cv2.imread("captchaimage.jpeg")  # Change to your image file
print(image)
# Convert image to grayscale (improves OCR accuracy)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding (binarization)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Perform OCR
text = pytesseract.image_to_string(thresh)

# Print detected text
print("Detected Text:", text)
