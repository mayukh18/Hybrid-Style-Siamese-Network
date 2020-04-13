import json
import numpy as np

class Data_augmentation:
    def __init__(self):
        '''
        Import image
        :param path: Path to the image
        :param image_name: image name
        '''
        return

    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image 
    
    
    def image_augment(self, image): 
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        '''
        v_flip = random.choice([True, False])
        h_flip = random.choice([True, False])
        #angle = random.choice([0,45,90,135,180,225,270,315,0])
        angle = random.choice([0,-25,25,-45,45])
        img_flip = self.flip(image, vflip=False, hflip=h_flip)
        img_rot = self.rotate(img_flip, angle)
        return img_rot

		
def MAPScorer(preds_1, preds_2):
    #print(preds_1.shape, preds_2.shape)
    map_ = 0.
    for i in range(300):
        dists = []
        for j in range(300):
            dist = np.linalg.norm(preds_1[i,:] - preds_2[j,:])
            #print("dist", dist)
            dists.append(dist)
        args = list(np.argsort(dists))
        idx = args.index(i)
        map_ += 1/(idx + 1)
    
    map1 = map_/300
    #print("MAP is", map_/300)
    
    map_ = 0.
    for i in range(300):
        dists = []
        for j in range(300):
            dist = np.linalg.norm(preds_2[i,:] - preds_1[j,:])
            #print("dist", dist)
            dists.append(dist)
        args = list(np.argsort(dists))
        idx = args.index(i)
        map_ += 1/(idx + 1)
    
    map2 = map_/300
    #print("MAP is", map_/300)
    
    map_ = 0.5*(map1+map2)
    return map_
	
class jsonf:
    def __init__(self,name):
        self.name = name
        data = {}
        self.write(data)

    def read(self):
        with open(self.name, 'r') as json_file:
            return json.load(json_file)

    def write(self,data):
        with open(self.name, 'w') as json_file:
            json.dump(data, json_file)

def printx(*argv, file):
    s = ''
    for x in argv:
        s = s + str(x)
    print(s)
    print(s, file=file)