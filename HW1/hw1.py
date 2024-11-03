import cv2
import matplotlib.pyplot as plt
import numpy as np
#P0
# img = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
# cv2.imwrite("result1.png",img)
# ver_img = img.copy()
# for i in range(int(len(ver_img)/2)):
#     a = ver_img[i].copy()
#     ver_img[i] = ver_img[-1-i]
#     ver_img[-1-i] = a

# cv2.imwrite("result2.png",ver_img)
# #P1
# tot_pixel = 600*800
# def modify_contrast(intensity,img):
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             new_val = int(img[i][j]*intensity)
#             if new_val>255:
#                 img[i][j]=255
#             else:
#                 img[i][j]=new_val
#     return img
# img = cv2.imread("./SampleImage/sample2.png",cv2.IMREAD_GRAYSCALE)
# #(a)
# img_1 = img.copy()
# img_1 = modify_contrast(1/3,img_1)
# cv2.imwrite("result3.png",img_1)
# #(b)
# img_2 = img_1.copy()
# img_2 = modify_contrast(3,img_2)
# cv2.imwrite("result4.png",img_2)
# #(c)
def draw_hist(img,name):
    tot_pixel = img.shape[0]*img.shape[1]
    hist = [0 for i in range(256)]
    for row in img:
        for pixel in row:
            hist[pixel]+=1
    plt.figure(figsize=(20,12))
    plt.bar(range(0,256),hist)
    plt.xlim([-2,257])
    if name!='':
        plt.savefig(name)
    return hist
# res_3 = cv2.imread("./result3.png",cv2.IMREAD_GRAYSCALE)
# res_4 = cv2.imread("./result4.png",cv2.IMREAD_GRAYSCALE)
# ori_hist = draw_hist(img,'hist_sam_2.png')
# res_3_hist = draw_hist(res_3,'hist_res_3.png')
# res_4_hist = draw_hist(res_4,'hist_res_4.png')
# #(d)
# def GHE(img,hist,tot_pixel,pixel_range):
    
#     PDF = [num/tot_pixel for num in hist]
#     CDF = [0 for i in range(len(PDF))]
#     prob = 0
    
#     for i,p in enumerate(PDF):
#         prob+=(p*pixel_range)
#         CDF[i] = int(prob) #原pixel轉新pixel
    
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             img[i][j] = CDF[img[i][j]]
    
    
#     return img
# res_5 = img.copy()
# res_6 = res_3.copy()
# res_7 = res_4.copy()
# res_5 = GHE(res_5,ori_hist,tot_pixel,255)
# res_6 = GHE(res_6,res_3_hist,tot_pixel,255)
# res_7 = GHE(res_7,res_4_hist,tot_pixel,255)
# a = draw_hist(res_5,'hist_res_5.png')
# a = draw_hist(res_6,'hist_res_6.png')
# a = draw_hist(res_7,'hist_res_7.png')
# cv2.imwrite("result5.png",res_5)
# cv2.imwrite("result6.png",res_6)
# cv2.imwrite("result7.png",res_7)
# #(e)
# def cal_hist(img):    
#     hist = [0 for i in range(256)]
#     for row in img:
#         for pixel in row:
#             hist[pixel]+=1
#     return hist
# def cal_CDF(hist,tot_pixel,pixel_range):
    
#     PDF = [num/tot_pixel for num in hist]
#     CDF = [0 for i in range(len(PDF))]
#     prob = 0
    
#     for i,p in enumerate(PDF):
#         prob+=(p*pixel_range)
#         CDF[i] = int(prob) #原pixel轉新pixel
#     return CDF

# def cal_CDF_img(img,hist,tot_pixel,pixel_range):
    
#     PDF = [num/tot_pixel for num in hist]
#     CDF = [0 for i in range(len(PDF))]
#     prob = 0
    
#     for i,p in enumerate(PDF):
#         prob+=(p*pixel_range)
#         CDF[i] = int(prob) #原pixel轉新pixel
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             img[i][j] = CDF[img[i][j]]
    
#     return CDF
 
# def LHE(img,size,pixel_range): #上下左右分別代表中心點向外考慮的pixel

#     up,down,right,left = size[0],size[0],size[1],size[1]
#     new_img = img.copy()
#     tot_pixel = (up+down+1)*(right+left+1)
#     ver_footprint = up+down
#     hor_footprint = left+right

#     # for i in range(0,len(img),ver_footprint+1):
#     #     for j in range(0,len(img[i]),hor_footprint+1):
#     #         local_img = img[i:i+ver_footprint+1,j:j+hor_footprint+1]
#     #         local_hist = cal_hist(local_img)
#     #         local_CDF = cal_CDF_img(local_img,local_hist,tot_pixel,pixel_range)


#     #針對每個pixel，一次只改一個pixel，速度很慢
#     for i in range(up):
#         new_img = np.insert(new_img,0,new_img[0],axis=0)
#     for i in range(down):
#         new_img = np.insert(new_img,-1,new_img[-1],axis=0)
#     for j in range(left):
#         new_img = np.insert(new_img,0,new_img[:,0],axis=1)
#     for j in range(right):
#         new_img = np.insert(new_img,-1,new_img[:,-1],axis=1)
    
#     for i in range(0,len(img)):
#         for j in range(0,len(img[i])):
#             local_hist = cal_hist(new_img[i:i+up+down+1,j:j+left+right+1])
#             local_CDF = cal_CDF(local_hist,tot_pixel,pixel_range)
#             img[i][j] = local_CDF[img[i][j]]
    
#     return img

# res_8 = img.copy()
# res_9 = res_3.copy()
# res_10 = res_4.copy()
# size = [1,1]
# res_8 = LHE(res_8,size,255)
# res_9 = LHE(res_9,size,255)
# res_10 = LHE(res_10,size,255)
# cv2.imwrite("result8.png",res_8)
# cv2.imwrite("result9.png",res_9)
# cv2.imwrite("result10.png",res_10)
# a = draw_hist(res_8,'hist_res_8.png')
# a = draw_hist(res_9,'hist_res_9.png')
# a = draw_hist(res_10,'hist_res_10.png')

# #(f)
# def transfer_function(img,tot_pixel,intensity,power1,power2,thr1,thr2):

#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             #暗部調亮，清晰化
#             if img[i][j]<thr1:
#                 img[i][j] = int(img[i][j]*intensity)
#             #使用power-off，模糊化亮部
#             img[i][j] = int(img[i][j]**power1)
#     img_hist = draw_hist(res_11,'')
#     img = GHE(img,img_hist,tot_pixel,255)
#     #降低亮部對比
#     for i in range(len(img)):
#         for j in range(len(img[i])):
#             if img[i][j]>=thr2:
#                 new_val = int(img[i][j]*power2)
#                 if new_val>255:
#                     img[i][j] = 255
#                 else:
#                     img[i][j] = new_val
#     return img

# tot_pixel = 600*800
# res_11 = img.copy()
# res_11 = transfer_function(res_11,tot_pixel,2.5,0.5,0.8,10,200)
# img_hist = draw_hist(res_11,'hist_res_11.png')
# cv2.imwrite("result11.png",res_11)

#P2

#(a)

def gaussian_blur(img,b,f):
    new_img = img.copy()
    if f == 1:
        filter_1D = np.asarray([[1,b,1]])
        filter_2D = np.matmul(filter_1D.transpose(),filter_1D)
        filter_2D = np.multiply((1/(b+2)**2),filter_2D)
        up,down,right,left = 1,1,1,1
        for i in range(up):
            new_img = np.insert(new_img,0,new_img[0],axis=0)
        for i in range(down):
            new_img = np.insert(new_img,-1,new_img[-1],axis=0)
        for j in range(left):
            new_img = np.insert(new_img,0,new_img[:,0],axis=1)
        for j in range(right):
            new_img = np.insert(new_img,-1,new_img[:,-1],axis=1)
    else:
        filter_1D = np.asarray([[1,4,6,4,1]])
        filter_2D = np.matmul(filter_1D.transpose(),filter_1D)
        filter_2D = np.multiply((1/256),filter_2D)
        up,down,right,left = 2,2,2,2
        for i in range(up):
            new_img = np.insert(new_img,0,new_img[0],axis=0)
        for i in range(down):
            new_img = np.insert(new_img,-1,new_img[-1],axis=0)
        for j in range(left):
            new_img = np.insert(new_img,0,new_img[:,0],axis=1)
        for j in range(right):
            new_img = np.insert(new_img,-1,new_img[:,-1],axis=1)

    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            filter_area = new_img[i:i+up+down+1,j:j+left+right+1].copy()
            filter_area = np.multiply(filter_area,filter_2D)
            new_val = np.sum(filter_area)
            if new_val>255:
                new_val = 255
            img[i,j] = int(new_val)
    return img

def median_blur(img,size):

    up,down,right,left = size[0],size[0],size[1],size[1]
    new_img = img.copy()
    for i in range(up):
        new_img = np.insert(new_img,0,new_img[0],axis=0)
    for i in range(down):
        new_img = np.insert(new_img,-1,new_img[-1],axis=0)
    for j in range(left):
        new_img = np.insert(new_img,0,new_img[:,0],axis=1)
    for j in range(right):
        new_img = np.insert(new_img,-1,new_img[:,-1],axis=1)

    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            filter_area = new_img[i:i+up+down+1,j:j+left+right+1].copy()
            filter_area = sorted(filter_area.flatten())
            new_val = filter_area[int(len(filter_area)/2)]
            img[i,j] = int(new_val)
    
    return img

def mean_blur(img,size):
    
    up,down,right,left = size[0],size[0],size[1],size[1]
    new_img = img.copy()
    for i in range(up):
        new_img = np.insert(new_img,0,new_img[0],axis=0)
    for i in range(down):
        new_img = np.insert(new_img,-1,new_img[-1],axis=0)
    for j in range(left):
        new_img = np.insert(new_img,0,new_img[:,0],axis=1)
    for j in range(right):
        new_img = np.insert(new_img,-1,new_img[:,-1],axis=1)

    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            filter_area = new_img[i:i+up+down+1,j:j+left+right+1].copy()
            new_val = np.sum(filter_area)/filter_area.size
            img[i,j] = int(new_val)
    
    return img

            

sam_3 = cv2.imread("./SampleImage/sample3.png",cv2.IMREAD_GRAYSCALE)
sam_4 = cv2.imread("./SampleImage/sample4.png",cv2.IMREAD_GRAYSCALE)
sam_5 = cv2.imread("./SampleImage/sample5.png",cv2.IMREAD_GRAYSCALE)
sam_4_1 = sam_4.copy()
sam_4_2 = sam_4.copy()
sam_4_3 = sam_4.copy()
sam_5_1 = sam_5.copy()
sam_5_2 = sam_5.copy()
sam_5_3 = sam_5.copy()
size = [1,1]
b = 2
f = 1
res_12_g = gaussian_blur(sam_4_1,b,f)
res_12_mean = mean_blur(sam_4_2,size)
res_12_median = median_blur(sam_4_3,size)
size = [1,1]
b = 2
f = 1
res_13_g = gaussian_blur(sam_5_1,b,f)
res_13_mean = mean_blur(sam_5_2,size)
res_13_median = median_blur(sam_5_3,size)
cv2.imwrite("result12.png",res_12_g)
cv2.imwrite("result12_mean.png",res_12_mean)
cv2.imwrite("result12_median.png",res_12_median)
cv2.imwrite("result13_g.png",res_13_g)
cv2.imwrite("result13_mean.png",res_13_mean)
cv2.imwrite("result13.png",res_13_median)
#(b)
def MSE(ori_img,new_img,tot_pixel):
    tot_val = 0
    for i in range(len(ori_img)):
        for j in range(len(ori_img[i])):
            val = int(ori_img[i,j])-int(new_img[i,j])
            tot_val = tot_val+val**2
    return tot_val/tot_pixel
def PSNR(mse):
    return 10*np.log10(255**2/mse)
tot_pixel = 313*320
res_12_5 = cv2.imread("./result12_5.png",cv2.IMREAD_GRAYSCALE)
res_13_5 = cv2.imread("./result13_5.png",cv2.IMREAD_GRAYSCALE)
print("MSE:")
print("sample4:"+str(MSE(sam_3,sam_4,tot_pixel)))
print("result12:"+str(MSE(sam_3,res_12_g,tot_pixel)))
print("result12_5:"+str(MSE(sam_3,res_12_5,tot_pixel)))
print("sample5:"+str(MSE(sam_3,sam_5,tot_pixel)))
print("result13:"+str(MSE(sam_3,res_13_median,tot_pixel)))
print("result13_5:"+str(MSE(sam_3,res_13_5,tot_pixel)))
print("PSNR:")
print("sample4:"+str(PSNR(MSE(sam_3,sam_4,tot_pixel))))
print("result12:"+str(PSNR(MSE(sam_3,res_12_g,tot_pixel))))
print("result12_5:"+str(PSNR(MSE(sam_3,res_12_5,tot_pixel))))
print("sample5:"+str(PSNR(MSE(sam_3,sam_5,tot_pixel))))
print("result13:"+str(PSNR(MSE(sam_3,res_13_median,tot_pixel))))
print("result13_5:"+str(PSNR(MSE(sam_3,res_13_5,tot_pixel))))

# for i in range(1,14):
#     name = "result"+str(i)+".png"
#     l = "./"+name
#     p = cv2.imread(l,cv2.IMREAD_GRAYSCALE)
#     cv2.imshow(name,p)
# for i in range(3,12):
#     name = "hist_res_"+str(i)+".png"
#     l = "./"+name
#     p = cv2.imread(l,cv2.IMREAD_GRAYSCALE)
#     cv2.imshow(name,p)
name = "hist_sam_2.png"
l = "./"+name
p = cv2.imread(l,cv2.IMREAD_GRAYSCALE)
cv2.waitKey(0)