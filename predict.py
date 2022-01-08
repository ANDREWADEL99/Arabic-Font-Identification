import numpy as np

from cv2 import cv2

import sys
import os
import pickle
import matplotlib.pyplot as plt
import time
import skimage.io as io
from sklearn.neural_network import MLPClassifier
import skimage as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skimage import data
from scipy.signal import convolve2d

def show_images(images,titles=None):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def PreProcessing(images):
    threshed = []
    for img in images:
        # (1) RGB to Gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # (2) threshold
        threshed.append(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])
    return threshed

def PreProcessingImage(img):
    # (1) RGB to Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (2) threshold
    threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return threshed

def lpq(img,winSize=3,freqestim=1,mode='nh'):
    rho=0.90

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA=8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
        w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode=='im':
        LPQdesc=np.uint8(LPQdesc)
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title("Original image")

        plt.subplot(122)
        plt.imshow(LPQdesc, cmap='gray')
        plt.title("lpq")
        plt.show()


    ## Histogram if needed
    if mode=='nh' or mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode=='nh':
        LPQdesc=LPQdesc/LPQdesc.sum()

    return LPQdesc
def train():
    # Note : all images are in the base folder ACdata_base
    # ex   : /ACdata_base
    #         |-> 0001.jpg
    #         |-> 0002.jpg
    # etc ....
    images = load_images_from_folder('ACdata_base')
    
    labels = np.ones(len(images))
    ###################### LAbels ############################
    labels[190:380] = 2
    labels[380:560] = 3
    labels[560:745] = 4
    labels[745:940] = 5
    labels[940:1120] = 6
    labels[1120:1305] = 7
    labels[1305:1495] = 8
    labels[1495:1684] = 9
    ##########################################################

    #################### Preprocessing #######################
    lines = PreProcessing(images)
    # this line for spliting data set into training set and test set
    #X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=0.2, random_state=42)
    X_train = lines
    y_train = labels
    ##########################################################


    ################ Feature Extraction ######################
    X_train_FE = []
    for i in X_train:
        X_train_FE.append(lpq(i))
    # for train and test for same data set
    #X_test_FE = []
    # for i in X_test:
    #     X_test_FE.append(lpq(i))
    ##########################################################
    

    ################ Training Model ##########################
    # define lists to collect scores
    train_scores = list()
    #test_scores = list()

    clf = RandomForestClassifier(max_depth=9, random_state=42).fit(X_train_FE, y_train)
    # evaluate on the train dataset
    clf.predict(X_train_FE)
    train_acc = clf.score(X_train_FE, y_train)
    train_scores.append(train_acc)
    ##########################################################


    ################ Saving the Model ########################
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    ##########################################################

if __name__ == "__main__":
    
    # Read the args from terminal  
    data_path = sys.argv[1]
    out_path = sys.argv[2]

    if data_path == 'ACdata_base':
        train()
    else :
        ################ Test pipline ########################
        # load saved model
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        
        # read imgs from test file
        test_images = load_images_from_folder(data_path)
        show_images([test_images[0]])
        # start time and open file outputs
        results_file = open(out_path +'/results.txt',"w")
        time_file = open(out_path +'/times.txt',"w")

        # Main processing
        for test_img in test_images:
            start = time.time()
            # preprocessing for test file 
            test_img_processed = PreProcessingImage(test_img)
            # feature extraction 
            test_img_feature = lpq(test_img_processed)
            # classification for the test file
            label = loaded_model.predict(test_img_feature.reshape(1,-1))
            # end time
            end = time.time()

            # print outputs
            Timer = end - start

            #check if predict function returns None
            if label[0] is None:
                results_file.write(str(-1))
            else:
                results_file.write(str(int(label[0])))
            results_file.write('\n')
            results_file.flush()
            time_file.write(str("{:.2f}".format(Timer)))
            time_file.write('\n')
            time_file.flush()

        results_file.close()
        time_file.close()


