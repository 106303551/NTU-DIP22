import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
#P1.a
def dith2thr(dither_m,b=0.5):
    n = dither_m.shape[0]
    thr_m = 255*(dither_m+b)/(n**2)
    return thr_m
def ditherimg(img,thr_m):
    n = thr_m.shape[0]
    for i in range(0,len(img)):
        for j in range(0,len(img[0])):
            real_i = i%n
            real_j = j%n
            if img[i,j]<thr_m[real_i,real_j]:
                img[i,j] = 0
            else:
                img[i,j]=255
    return img
dither_m = [[1,2],[3,0]]
dither_m = np.asarray(dither_m)
thr_m = dith2thr(dither_m,0.5)
img = cv2.imread('SampleImage/sample1.png',cv2.IMREAD_GRAYSCALE)
dith_img =ditherimg(img,thr_m)
cv2.imwrite('result1.png',dith_img)
cv2.imshow('reuslt1',dith_img)
# #P1.b
def expanddither_m(dither_m,target_dim):
    while(dither_m.shape[0]<target_dim):
        dither_m = 4*dither_m
        first = dither_m+1
        second = dither_m+2
        m_1 = np.concatenate((first,second),axis=1)
        m_2 = np.concatenate((dither_m+3,dither_m),axis=1)
        dither_m = np.concatenate((m_1,m_2))
    return dither_m
dither_m = expanddither_m(dither_m,256)
thr_m = dith2thr(dither_m,0.5)
img = cv2.imread('SampleImage/sample1.png',cv2.IMREAD_GRAYSCALE)
dith_img =ditherimg(img,thr_m)
cv2.imwrite('result2.png',dith_img)
cv2.imshow('result2',dith_img)
#P1.c
def floyd_steinberg(img,thr):
    n_img = (img/255).copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            n_val = img[i,j]/255
            if i>0:
                n_val = n_val+(5/16)*n_img[i-1,j]
                if j-1>0:
                    n_val = n_val+(1/16)*n_img[i-1,j-1]
                if j+1<img.shape[1]:
                    n_val =  n_val+(3/16)*n_img[i-1,j+1]
            if j>0:
                n_val = n_val+(7/16)*n_img[i,j-1]
            if n_val>=thr:
                n_img[i,j] = n_val-1
                img[i,j]=255
            else:
                n_img[i,j] = n_val
                img[i,j]=0
    return img
def jarvis_pattern(img,thr):
    n_img = (img/255).copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            n_val = img[i,j]/255
            if i>0:
                n_val = n_val+(7/48)*n_img[i-1,j]
                if j>0:
                    n_val = n_val+(5/48)*n_img[i-1,j-1]
                if j>1:
                    n_val = n_val+(3/48)*n_img[i-1,j-2]
                if j+1<img.shape[1]:
                    n_val =  n_val+(5/48)*n_img[i-1,j+1]
                if j+2<img.shape[1]:
                    n_val = n_val+(3/48)*n_img[i-1,j+2]
            if i>1:
                n_val = n_val+(5/48)*n_img[i-2,j]
                if j>0:
                    n_val = n_val+(3/48)*n_img[i-2,j-1]
                if j>1:
                    n_val = n_val+(1/48)*n_img[i-2,j-2]
                if j+1<img.shape[1]:
                    n_val =  n_val+(3/48)*n_img[i-2,j+1]
                if j+2<img.shape[1]:
                    n_val = n_val+(1/48)*n_img[i-2,j+2]
            if j>0:
                n_val = n_val+(7/48)*n_img[i,j-1]
            if j>1:
                n_val = n_val+(5/48)*n_img[i,j-2]
            if n_val>=thr:
                n_img[i,j] = n_val-1
                img[i,j]=255
            else:
                n_img[i,j] = n_val
                img[i,j]=0
    return img
img = cv2.imread('SampleImage/sample1.png',cv2.IMREAD_GRAYSCALE)
fs_img = floyd_steinberg(img,0.5)
cv2.imwrite('result3.png',fs_img)
cv2.imshow('result3',fs_img)
img = cv2.imread('SampleImage/sample1.png',cv2.IMREAD_GRAYSCALE)
jp_img = jarvis_pattern(img,0.5)
cv2.imwrite('result4.png',jp_img)
cv2.imshow('result4',jp_img)
#p2.a

def img_sampling(img,sample_rate):
    new_img = []
    delta_x = 2*math.pi/sample_rate
    delta_y = 2*math.pi/sample_rate
    max_idx_y = int((img.shape[0]-1)//delta_x)
    max_idx_x = int((img.shape[1]-1)//delta_y)
    for i in range(max_idx_y):
        img_list=[]
        for j in range(max_idx_x):
            val = img[int(i*delta_y),int(j*delta_x)]
            img_list.append(val)
        new_img.append(img_list)
    return new_img

img = cv2.imread('SampleImage/sample2.png',cv2.IMREAD_GRAYSCALE)
sam_img = img_sampling(img,2)
sam_img = np.asarray(sam_img)
cv2.imwrite('result5.png',sam_img)
cv2.imshow('result5',sam_img)
#p2.b
def g_highpass_filter(img,sigma,increase):
    #sigma for cutoff
    highpass_f = np.zeros((img.shape[0],img.shape[1]))
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    xx, yy = np.meshgrid(x, y)
    gaussian = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    highpass_f = increase*(1-gaussian)
    img = img*highpass_f
    return img
def g_lowpass_filter(img,sigma):
    lowpass_f = np.zeros((img.shape[0],img.shape[1]))
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    xx, yy = np.meshgrid(x, y)
    lowpass_f = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    img = img*lowpass_f
    return img

img = cv2.imread('SampleImage/sample2.png',cv2.IMREAD_GRAYSCALE)
f_img = np.fft.fft2(img)
f_img = np.fft.fftshift(f_img)
f_img = g_lowpass_filter(f_img,80)
f_img = g_highpass_filter(f_img,80,3)
f_filtered = np.fft.ifftshift(f_img)
abs_img = np.abs(np.fft.ifft2(f_filtered))
abs_img = abs_img.astype('uint8')
cv2.imwrite('result6.png',abs_img)
cv2.imshow('result6',abs_img)

#p2.c
def g_lowpass_filter(img,sigma):
    lowpass_f = np.zeros((img.shape[0],img.shape[1]))
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    xx, yy = np.meshgrid(x, y)
    lowpass_f = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    img = img*lowpass_f
    return img

img = cv2.imread('SampleImage/sample3.png',cv2.IMREAD_GRAYSCALE)
f_img = np.fft.fft2(img)
f_filtered = np.fft.fftshift(f_img)
f_filtered = g_lowpass_filter(f_filtered,40)
f_filtered = np.fft.ifftshift(f_filtered)
abs_img = np.abs(np.fft.ifft2(f_filtered))
abs_img = abs_img.astype('uint8')
cv2.imshow('result7',abs_img)
cv2.imwrite('result7.png',abs_img)
cv2.waitKey(0)

            

