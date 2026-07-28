"""
Legacy face-detection helper for locating and cropping the encoded region
of a face (LBP cascade or PRNet keypoints) and stitching the encoded face
back into the full image after encoding.
"""

from Face_Detection_Utilies.api import PRN
import cv2
import numpy as np




def Load_PRN():
    prn = PRN(is_dlib = True )
    return prn
    
    
    

    
class FaceDetection:
    """
    info: 
    - sould put Data file and api.py file and predector file.
    - you need CasscadeModel file, too.
    -----------------------------------------------------------
    input: image, encoded_image
    -----------------------------------------------------------
    output: faceCroped, encoded_image
    """
    def __init__(self, image):
        self.image = image
        self.croped_pints = []
        self.face = 0
        self.prn = 0
        
    def __help__():
        pass
    def LBpCascade(self):
        face_cascade = cv2.CascadeClassifier('./CasscadeModel/lbpcascade_frontalface_improved.xml')
        face_values = face_cascade.detectMultiScale(self.image, 1.1, 4)
        for (x, y, w, h)  in face_values:
            x, y, w, h = x, y, w, h
        
        face = image[y:y+h,x:x+w]
        face = face.copy()
        
        self.face = face
        self.croped_pints = [y,y+h,x,x+w]
        
        self.face_height, self.face_width = face.shape[0],face.shape[1]
        
    def kpt(self, prn):
        max_size = max(self.image.shape[0], self.image.shape[1])
        if max_size> 1000:
            image = rescale(self.image, 1000./max_size)
            image = (image*255).astype(np.uint8)
            print(tcolors.RED, "decresies image in kpt FaceDetection", tcolors.ENDC)
        pos = prn.process(image) # use dlib to detect face   
        kpt = prn.get_landmarks(pos)
    
        #
        first,  second, third = [],[],[]
        for ith,_ in enumerate(kpt):
            for jth, value in enumerate(kpt[ith]):
                if jth == 0:
                    first.append(value)
                if jth == 1:
                    second.append(value)
                if jth == 2:
                    third.append(value)    
    
        #
        img5 = image.copy()
        y_max, y_min = int(max(second)), int(min(second))
        x_max, x_min = int(max(first)), int(min(first))
        crop_img = img5[y_min:y_max, x_min:x_max]
        crop_img = crop_img.copy()
        croped_pints = np.array([y_min, y_max, x_min,x_max])
        #plt.imshow(crop_img);plt.show()
        
        self.face = crop_img
        self.croped_pints = croped_pints
    def Kpst_extra(self):
        pass
    def Vertix(self):
        pass
    def MTCNN(self):
        pass
    
    def Match_Images(self, encoded_face):
        encoded_face = cv2.resize(encoded_face, (self.face_width, self.face_height))
        y_min,y_max, x_min,x_max = self.croped_pints
        image = self.image.copy()
        image[y_min:y_max, x_min:x_max] = encoded_face
        encoded_image = image.copy()
        
        return encoded_image
        
def Covert_Array_to_image(input_array):
    open_cv_image = np.array(input_array) 
    # Convert RGB to BGR 
    input_array = open_cv_image[:, :, ::-1].copy() 
    output_img = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)
    return output_img
