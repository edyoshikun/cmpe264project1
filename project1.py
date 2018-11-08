import cv2
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import linregress
from array import *
import matplotlib.gridspec as gp

#File to output the values of a and g based on the estimation
with open("RGB_estimation_results.txt", "w") as file:
    file.write("#Estimation of parameter g from B'(T)\n#CMPE264 \n#Eduardo Hirata\n#Joshua Pena\n")

images=[]
titles=['blue','green','red']

#Read the image files 'p' for phone and 'w' for Nikon pictures
for n in range(1,6):
   images.append(cv2.imread('./phoneCalibration/p'+str(n)+'.JPG'))

#For Nikon pictures 'w1.jpg'
# time=np.array([1/2500,1/1000,1/500,1/50,1/40,1/25],dtype='float64')
time=np.array([1.0/320,1.0/200,1.0/80,1.0/50,1.0/25],dtype='float64')

blue=[]
green=[]
red=[]

for img in images:
    b,g,r = cv2.split(img)
    blue.append(b)
    green.append(g)
    red.append(r)

colors=[blue,green,red]

for i,col in enumerate(tuple(colors)):

    plt.figure()
    for n , img in enumerate(tuple(col)):
        plt.subplot(3,3,n+1), plt.imshow(img,'gray')
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.suptitle("Original Images")
        plt.title("Col="+ str(titles[i])+" img=%d"%n)
        plt.savefig('init_images-'+ str(titles[i])+'.png', bbox_inches='tight')
    plt.close()

    ###############################Plot the Histograms ###############################
    plt.figure()
    for n, img in enumerate(tuple(col)):
        plt.subplot(3,3,n+1),
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.hist(img.ravel(),256,[0,256])
        plt.suptitle("Histogram of Images")
        plt.title("Col="+ str(titles[i]) + " img=%d" %n)
        plt.savefig('histogram_col-'+ str(titles[i])+'.png', bbox_inches='tight')
    plt.close()
    #To get the central pixel, lets create a rectangular mask that takes the pixels in the middle
    mask = np.zeros(col[0].shape[:2],np.uint8)
    center_w = int(round(col[0].shape[0]/2))
    center_h = int(round(col[0].shape[1]/2))
    # print(col[0].shape[0], col[0].shape[1])
    # print(center_w , center_h)
    off= 400;

    ##################################Cropped images##############################
    cr_im_array = []
    for img in col:
        cr_im_array.append(np.mean(img[center_w-off:center_w+off, center_h-off:center_h+off]))

    ###############################Apply a Mask #####################################
    masked_img=[]
    hist_mask=[]
    crop_img=[]
    mu_img=[]
    mask[(center_w-off):(center_w+off), (center_h-off):(center_h+off)]=255  #Make the rest dark
    for img in col:
        masked_img.append(cv2.bitwise_and(img,img, mask = mask))
        hist_mask.append(cv2.calcHist([img],[0],mask,[256],[0,256]))
        crop_img.append(img[(center_w-off):(center_w+off), (center_h-off):(center_h+off)])
        mu_img.append(np.mean(crop_img))

    ###################Plot the masked images ############################
    plt.figure()
    for n, img in enumerate(tuple(masked_img)):
        plt.subplot(3,3,n+1), plt.imshow(img,'gray')
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.suptitle("Masked Img")
        plt.title("Col= "+ str(titles[i])+" img=%d" %n)
        plt.savefig('masked_img-'+ str(titles[i])+'.png', bbox_inches='tight')
    plt.close()
    #####################plot the masked images histograms #######################
    plt.figure()
    for n, img in enumerate(tuple(hist_mask)):
        plt.subplot(3,3,n+1), plt.plot(img)
        plt.suptitle("Center Pixels of Images")
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.xlim([0,256])
        plt.title('Col=' + str(titles[i])+ " img=%d" %n)
        plt.savefig('mask_hist-'+ str(titles[i])+'.png', bbox_inches='tight')

    plt.close()
############################ CROPPED IMAGES #############################
    plt.figure()
    for n, img in enumerate(tuple(crop_img)):
        plt.subplot(3,3,n+1), plt.imshow(img,'gray')
        plt.suptitle("Cropped Images")
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.title('Col=' + str(titles[i])+" img=%d"%n)
        plt.savefig('crop_img-'+ str(titles[i])+'.png', bbox_inches='tight')
    plt.close()
    ####################### ESTIMATION of G by B'(T)########################################
    slope, intercept, r,p,sigma = linregress(time,mu_img)
    # print(slope,intercept,r,p,sigma)
    y= intercept + slope * time
    plt.figure()
    plt.plot(time,mu_img,'b', time,y,'r', time,mu_img,'*' , label ='y=%.2f ax+%.2f'%(slope,intercept)),
    plt.title("Estimation for parameter g from B'(T)' Col="+ str(titles[i]))
    plt.ylabel("Pixel Intesity B(T)")
    plt.xlabel("T(s)")
    plt.legend(loc='upper left')
    plt.savefig('g_estimate-'+ str(titles[i])+'.png', bbox_inches='tight')
    plt.close()
#################################### LOG(B'(T)) ##############################
    plt.figure()
    logB=np.log10(mu_img)
    logT=np.log10(time)
    slope2, intercept2, r2,p2,sigma2=linregress(logT,logB)
    a=slope2
    g=1/a
    b=intercept2
    ylog=b + a*logT
    # print(logB,logT,ylog)
    plt.plot(logT,logB,'b',logT,ylog,'r', logT,logB,'*', label ='y=%.2fx + %.2f'%(a,b))
    plt.title("Estimation for parameter g from Log(B'(T)') Col="+ str(titles[i]))
    plt.ylabel("Pixel Intesity Log(B(s)')")
    plt.xlabel("Log(T(s))")
    plt.legend(loc='upper left')
    plt.savefig('g_estimate_log-'+ str(titles[i]) +'.png', bbox_inches='tight')
   
    print("g=", g)
    plt.close()
#################### B=1/a Estimation
    plt.figure()
    Bg=np.power(mu_img,g)
    plt.plot(time,Bg)
    plt.title("Estimation for parameter g from B'=B^1/a Col=" + str(titles[i]))
    plt.ylabel("Pixel Intesity B=B^1/a")
    plt.xlabel("T(s)")
    plt.savefig('g_estimate_a-'+ str(titles[i])+'.png', bbox_inches='tight')
    with open("RGB_estimation_results.txt", "a") as file:
        file.write(str(titles[i])+ "\n" + "g=%f a=%f b=%f\n"%(g,slope2,intercept2))
    plt.close()

# plt.show()