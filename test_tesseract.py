try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
#print(pytesseract.image_to_string(Image.open('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg')))

# French text image to string
#print(pytesseract.image_to_string(Image.open('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg'), lang='vie'))

# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
#print(pytesseract.image_to_string('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg'))

# Batch processing with a single file containing the list of multiple image file paths
#print(pytesseract.image_to_string('images.txt'))

# Timeout/terminate the tesseract job after a period of time
# try:
#     print(pytesseract.image_to_string('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg', timeout=2)) # Timeout after 2 seconds
#     print(pytesseract.image_to_string('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg', timeout=0.5)) # Timeout after half a second
# except RuntimeError as timeout_error:
#     # Tesseract processing is terminated
#     pass

# Get bounding box estimates
#print(pytesseract.image_to_boxes(Image.open('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg')))

# Get verbose data including boxes, confidences, line and page numbers
#print(pytesseract.image_to_data(Image.open('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg')))

# Get information about orientation and script detection
#print(pytesseract.image_to_osd(Image.open('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg')))

# Get a searchable PDF
#pdf = pytesseract.image_to_pdf_or_hocr('test.png', extension='pdf')
#with open('test.pdf', 'w+b') as f:
 #   f.write(pdf) # pdf type is bytes by default

# Get HOCR output
#print(hocr = pytesseract.image_to_pdf_or_hocr('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg', extension='hocr'))

# Get ALTO XML output
#xml = pytesseract.image_to_alto_xml('/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image003.jpg')

import pytesseract
from pytesseract import Output
import cv2
filename = "/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image009.jpg"

# read the image and get the dimensions
img = cv2.imread(filename)

img = cv2.medianBlur(img, 3)

h, w, _ = img.shape # assumes color image

# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img) # also include any config options you use

# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

# show annotated image and wait for keypress
cv2.imshow(filename, img)
cv2.waitKey(0)

