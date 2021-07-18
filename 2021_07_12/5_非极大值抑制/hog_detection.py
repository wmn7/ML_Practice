'''
@Author: WANG Maonan
@Date: 2021-07-18 16:03:46
@Description: 
@LastEditTime: 2021-07-18 16:20:17
'''
from skimage import feature
import cv2
import joblib 

class HOG:
    def __init__(self, orientations = 9, pixelsPerCell = (8, 8),
        cellsPerBlock = (3, 3), transform = False):
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.transform = transform
        
    def describe(self, image):
        hist = feature.hog(image, orientations = self.orienations,
                           pixels_per_cell = self.pixelsPerCell,
                           cells_per_block = self.cellsPerBlock,
                           transform_sqrt = self.transform)
        return hist

def sliding_window(image, window = (64, 128), step = 4):
    for y in range(0, image.shape[0] - window[1], step):
        for x in range(0, image.shape[1] - window[0], step):
            yield (x, y, image[y:y + window[1], x:x + window[0]]) 

def pyramid(image, top = (224, 224), ratio = 1.5):
    yield image
    while True:
        (w, h) = (int(image.shape[1] / ratio), int(image.shape[0] / ratio))
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        if w < top[1] or h < top[0]:
            break
        
        yield image

def resize(image, height = None, width = None):
    h, w = image.shape[:2]
    dim = None
    
    if width is None and height is None:
        return image
    
    if width is None:
        dim = (int(w * (height / h)), height)
    else:
        dim = (width, int(h * (width / w)))
        
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def coordinate_transformation(height, width, h, w, x, y, roi):
    if h is 0 or w is 0:
        print("divisor can not be zero !!")
    
    img_h = int(height/h * roi[1])
    img_w = int(width/w * roi[0])

    img_y = int(height/h * y)
    img_x = int(width/w * x)

    return (img_x, img_y, img_w, img_h) 


def run(img_path = "man.jpg", model_path='model'):
    ratio = 1.5
    i_roi = (64, 128)
    step = 20

    roi_loc = []

    model = joblib.load(model_path)
    hog = HOG(transform = True)

    image = cv2.imread(img_path)
    resized = resize(image, height = 500)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    for image in pyramid(gray, top = (128, 128), ratio = ratio):
        h, w = image.shape[:2]
    
        for (x, y, roi) in sliding_window(image, window = i_roi, step = step):
            roi = cv2.resize(roi, (64, 128), interpolation = cv2.INTER_AREA)   
            hist = hog.describe(roi)
        
            if model.predict([hist])[0]:
                img_x, img_y, img_w, img_h = coordinate_transformation(height, width, h, w, x, y, i_roi) 
                roi_loc.append([img_x, img_y, img_w, img_h])
    
    return (roi_loc, resized)
