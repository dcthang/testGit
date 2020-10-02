import cv2
from tesseract2dict import TessToDict

td=TessToDict()

inputImage=cv2.imread("/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image009.jpg")

#inputImage = cv2.medianBlur(inputImage, 3)
gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

### function 1
word_dict=td.tess2dict(gray,'out','outfolder')
print(word_dict)
### function 2
text_plain=td.word2text(word_dict,(0,0,inputImage.shape[1],inputImage.shape[0]))
