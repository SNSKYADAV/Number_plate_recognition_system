import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow  

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def detect_plate(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found!")
        return

    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Top 10 largest contours

    plate_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:  # Check if it's a quadrilateral
            plate_contour = approx
            break

    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Crop plate
        plate = gray[y:y+h, x:x+w]
        
        # Apply OCR
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(plate, config=custom_config)
        text = ''.join(e for e in text if e.isalnum())

        # Display results
        plt.figure(figsize=(10, 6))
        plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Detected Plate')
        plt.subplot(122), plt.imshow(plate, cmap='gray'), plt.title('Extracted Plate')
        plt.show()

        print(f"Detected License Plate: {text}")
    else:
        print("No plate detected!")

# Example usage (upload your image to Colab)
!wget -O test_car.jpg "https://raw.githubusercontent.com/ankurankan/ANPR/master/test_images/test1.jpg"
detect_plate("test_car.jpg")
