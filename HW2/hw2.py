import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as matimg
from PIL import Image
#P1

#(a)
def sobel(img,h_sobel,v_sobel,width,height,thr):

    edge_img = img.copy()
    grad_img = img.copy()

    for i in range(height//2):
        img = np.insert(img,0,img[0],axis=0)
    for i in range(height//2):
        img = np.insert(img,-1,img[-1],axis=0)
    for j in range(width//2):
        img = np.insert(img,0,img[:,0],axis=1)
    for j in range(width//2):
        img = np.insert(img,-1,img[:,-1],axis=1)
    angle_img = []

    for i in range(0,len(edge_img)):
        angle_list=[]
        for j in range(len(edge_img[0])):
            local_img = img[i:i+height,j:j+width]
            gx = np.multiply(h_sobel,local_img).sum()
            gy = np.multiply(v_sobel,local_img).sum()
            if gx == 0:
                angle = 90
            else:
                angle =math.degrees(math.atan(gy/gx))
            if (angle%360<112.5 and angle%360>67.5) or (angle%360>247.5 and angle%360<292.5):
                angle_list.append(0) #上下
            elif (angle%360<22.5 or angle%360>337.5) or (angle%360<202.5 and angle%360>157.5):
                angle_list.append(1) #左右
            elif (angle%360<157.5 and angle%360>112.5) or (angle%360>292.5 and angle%360<337.5):
                angle_list.append(2) #左上
            else:
                angle_list.append(3)#右上
            g = (gx**2+gy**2)**0.5
            if g>=thr:
                edge_img[i][j] = 255
            else:
                edge_img[i][j] = 0
            if g>=255:
                grad_img[i][j] = 255
            else:
                grad_img[i][j] = int(g)
        angle_img.append(angle_list)

    return edge_img,grad_img,angle_img

width = 3
height = 3
v_sobel=[]
for i in range(height):
    if i ==0:
        l=[1 for j in range(width)]
        l[width//2] = 2
    elif i == height-1:
        l = [-1 for j in range(width)]
        l[width//2]= -2
    else:
        l = [0 for j in range(width)]
    v_sobel.append(l)
v_sobel=np.asarray(v_sobel)
h_sobel = v_sobel.T
thr =150
img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
res_2,res_1,_ = sobel(img_1,h_sobel=h_sobel,v_sobel=v_sobel,width=width,height=height,thr=thr)
cv2.imshow('result1',res_1)
cv2.imshow('result2',res_2)
cv2.imwrite("result2.png",res_2)
cv2.imwrite("result1.png",res_1)
# for i in range(50,255,50):
#     thr = i
#     edge_name = "sobel_edge_img_1_thr_"+str(i)+".png"
#     edge_path = "./"+edge_name
#     grad_name = "sobel_grad_img_1_thr_"+str(i)+".png"
#     grad_path = "./"+grad_name
#     img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
#     res_1,res_2,_ = sobel(img_1,h_sobel=h_sobel,v_sobel=v_sobel,width=width,height=height,thr=thr)
#     cv2.imshow(edge_name,res_1)
#     cv2.imshow(grad_name,res_2)
#     cv2.imwrite(edge_path,res_1)
#     cv2.imwrite(grad_path,res_2)
# cv2.waitKey(0)
#(b)
def gaussian_blur(img,w,h,b):
    new_img = img.copy()
    up,down = h//2,h//2
    left,right = w//2,w//2

    for i in range(up):
        new_img = np.insert(new_img,0,new_img[0],axis=0)
    for i in range(down):
        new_img = np.insert(new_img,-1,new_img[-1],axis=0)
    for j in range(left):
        new_img = np.insert(new_img,0,new_img[:,0],axis=1)
    for j in range(right):
        new_img = np.insert(new_img,-1,new_img[:,-1],axis=1)
    filter_2D=[]
    tot = 0
    for i in range(-up,up+1,1):
        filter_1D=[]
        for j in range(-left,left+1,1):
            g_val = (1/(2*math.pi*b**2))*math.exp(-(i**2+j**2)/(2*b**2))
            tot+=g_val
            filter_1D.append(g_val)
        filter_2D.append(filter_1D)
    filter_2D = np.asarray(filter_2D)
    filter_2D = np.multiply(filter_2D,1/tot)
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            filter_area = new_img[i:i+up+down+1,j:j+left+right+1].copy()
            filter_area = np.multiply(filter_area,filter_2D)
            new_val = np.sum(filter_area)
            if new_val>255:
                new_val = 255
            img[i,j] = int(new_val)
    return img

def Canny(img,Th,Tl):
    #1.gaussian blur
    w=3
    h=3
    b=1.8
    img = gaussian_blur(img,w,h,b)
    #2.cal grad
    width = 5
    height = 5
    v_sobel=[]
    for i in range(height):
        if i ==0:
            l=[1 for j in range(width)]
            l[width//2] = 2
        elif i == height-1:
            l = [-1 for j in range(width)]
            l[width//2]= -2
        else:
            l = [0 for j in range(width)]
        v_sobel.append(l)
    v_sobel=np.asarray(v_sobel)
    h_sobel = v_sobel.T
    _,grad_img,angle_img = sobel(img,h_sobel=h_sobel,v_sobel=v_sobel,width=width,height=height,thr=220)
    #3.non-maximal suppression
    old_grad =grad_img.copy()
    for i in range(len(grad_img)):
        for j in range(len(grad_img[0])):
            angle = angle_img[i][j]
            val = grad_img[i][j]
            if angle == 0:
                if i-1>=0 and i+1<len(grad_img):
                    g1 = grad_img[i-1][j]
                    g2 = grad_img[i+1][j]
                    if g1<val and g2<val:
                        continue
                    else:
                        grad_img[i][j] = 0
                        continue
            elif angle == 1:
                if j-1>=0 and j+1<len(grad_img):
                    g1 = grad_img[i][j-1]
                    g2 = grad_img[i][j+1]
                    if g1<val and g2<val:
                        continue
                    else:
                        grad_img[i][j]=0
                        continue
            elif angle == 2:
                if i-1>=0 and j-1>=0 and i+1<len(grad_img) and j+1<len(grad_img):
                    g1 = grad_img[i-1][j-1]
                    g2 = grad_img[i+1][j+1]
                    if g1<val and g2<val:
                        continue
                    else:
                        grad_img[i][j]=0
                        continue
            else:
                if i-1>=0 and j-1>=0 and i+1<len(grad_img) and j+1<len(grad_img):
                    g1 = grad_img[i-1][j+1]
                    g2 = grad_img[i+1][j-1]
                    if g1<val and g2<val:
                        continue
                    else:
                        grad_img[i][j]=0
                        continue
    #4.hysteretic thresholding
    new_grad = grad_img.copy()
    for i in range(len(grad_img)):
        for j in range(len(grad_img[0])):
            if grad_img[i][j]>=Th:
                new_grad[i][j] = 255
            elif Th>grad_img[i][j]>Tl:
                new_grad[i][j] = 100
            else:
                new_grad[i][j] = 0
    #5.connect component method
    
    def connect_search(img,h_list,w_list,visit_array): #只檢查周圍
        for h in h_list:
            for w in w_list:
                if 0<h<len(img) and 0<w<len(img[0]):
                    if visit_array[h][w] == 0:
                        if img[h][w] == 100:
                            visit_array[h][w] = 1
                            new_h_list = [h-1,h,h+1]
                            new_w_list = [w-1,w,w+1]
                            connect_search(img,new_h_list,new_w_list,visit_array)

    visit_array = [[0 for i in range(len(new_grad[0]))] for j in range(len(new_grad))]
    for i in range(len(visit_array)):
        for j in range(len(visit_array[0])):
            if visit_array[i][j] == 0:
                if new_grad[i][j] == 255:
                    visit_array[i][j] = 1
                    h_list = [i+k for k in range(-1,2)]
                    w_list = [j+k for k in range(-1,2)]
                    connect_search(new_grad,h_list,w_list,visit_array=visit_array)
                    
    for i in range(len(visit_array)):
        for j in range(len(visit_array[0])):
            if visit_array[i][j] == 0:
                new_grad[i][j] = 0
            else:
                new_grad[i][j] = 255

    return new_grad

img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
res_3 = Canny(img=img_1,Th=200,Tl=150)
cv2.imshow("result_3",res_3)
cv2.imwrite("./result3.png",res_3)

#(c)
def draw_hist(img,name):
    tot_pixel = img.shape[0]*img.shape[1]
    hist = [0 for i in range(256)]
    for row in img:
        for pixel in row:
            hist[pixel]+=1
    plt.figure(figsize=(20,12))
    plt.bar(range(0,256),hist)
    plt.xlim([-2,257])
    # if name!='':
    #     plt.savefig(name)
    return hist

def LOG(img,w,h,b,thr):

    new_img = img.copy()
    up,down = h//2,h//2
    left,right = w//2,w//2

    for i in range(up):
        new_img = np.insert(new_img,0,new_img[0],axis=0)
    for i in range(down):
        new_img = np.insert(new_img,-1,new_img[-1],axis=0)
    for j in range(left):
        new_img = np.insert(new_img,0,new_img[:,0],axis=1)
    for j in range(right):
        new_img = np.insert(new_img,-1,new_img[:,-1],axis=1)
    filter_2D=[]
    tot = 0
    for i in range(-up,up+1,1):
        filter_1D=[]
        for j in range(-left,left+1,1):
            g_val = (1/(math.pi*b**4))*(1-(i**2+j**2)/(2*b**2))*math.exp(-(i**2+j**2)/(2*b**2))
            tot+=g_val
            filter_1D.append(g_val)
        filter_2D.append(filter_1D)
    filter_2D = np.asarray(filter_2D)
    filter_2D = np.multiply(filter_2D,1/tot)
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            filter_area = new_img[i:i+up+down+1,j:j+left+right+1].copy()  
            filter_area = np.multiply(filter_area,filter_2D)
            new_val = np.sum(filter_area)
            if new_val>255:
                new_val = 255
            img[i,j] = int(new_val)
    img_hist =draw_hist(img,"log_hist.png")
    idx = img_hist.index(max(img_hist))
    
    for i in range(len(img)):
        for j in range(len(img[0])):
            if idx-thr<=img[i][j]<=idx+thr:
                img[i][j] = 255
            else:
                img[i][j] = 0
    new_img = img.copy()
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 255:
                cross_flag =False
                h_list = [i+k for k in range(-1,2)]
                w_list = [j+k for k in range(-1,2)]
                for h in h_list:
                    for w in w_list:
                        if 0<h<len(img) and 0<w<len(img[0]):
                            if img[h][w] == 0:
                                cross_flag = True
                            break
                if cross_flag == False:
                    new_img[i][j] = 0
    return new_img
img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
w=7
h=7
b=1.4
thr = 10
res_4 = LOG(img=img_1,w=w,h=h,b=b,thr=thr)
cv2.imwrite("result4.png",res_4)
cv2.imshow('result4',res_4)


#(d)
def edge_crispening(img,w,h,b,c):
    LP = img.copy()
    LP = gaussian_blur(LP,w=w,h=h,b=b)
    for i in range(len(img)):
        for j in range(len(img[0])):
            new_val = ((c/(2*c-1))*img[i][j]-((1-c)/(2*c-1))*LP[i][j])
            if new_val>255:
                img[i][j] = 255
            elif new_val<0:
                img[i][j] = 0
            else:
                img[i][j] = int(new_val)
    return img

w=7
h=7
b=1.4
c = 0.6
img_2 = cv2.imread("./SampleImage/sample2.png",cv2.IMREAD_GRAYSCALE)
img_2 = edge_crispening(img=img_2,w=w,h=h,b=b,c=c)
cv2.imshow('result5',img_2)
cv2.imwrite("result5.png",img_2)

# (e)

# Q(2)

# (a)

def translation(img,y_shift,x_shift,bound_x,bound_y):
    M =[[1,0,x_shift],[0,1,y_shift],[0,0,1]]
    M = np.linalg.inv(np.asarray(M))
    new_img = np.asarray([[255 for i in range(len(img[0]))] for j in range(len(img))])
    new_img = new_img.astype('uint8')
    for i in range(len(img)):#轉換後圖y軸 j
        for j in range(len(img[0])):#轉換後圖x軸 k
            if i >=bound_y or j <=bound_x:
                new_img[i][j] = img[i][j]
                continue
            xk = j
            yj = len(img)-1-i
            a = np.asarray([xk,yj,1])
            val = np.dot(M,[xk,yj,1])#[uq,vp,1]
            q = int(val[0]) #寬
            p = int(len(img)-1-val[1]) #高
            if i<=bound_y and j>=bound_x:
                if p>bound_y or q<bound_x:
                    continue
                else:
                    if 0<=p<len(img) and 0<=q<len(img[0]):
                        new_img[i][j] = img[p][q]
            else:
                new_img[i][j] = img[i][j]
    return new_img

def rotation(img,theta,bound_x,bound_y,t_x,t_y):
    M = [[math.cos(math.radians(theta)),-math.sin(math.radians(theta)),0],
         [math.sin(math.radians(theta)),math.cos(math.radians(theta)),0],
         [0,0,1]]
    M = np.asarray(M)
    M = np.linalg.inv(M)
    new_img = np.asarray([[255 for i in range(len(img[0]))] for j in range(len(img))])
    new_img = new_img.astype('uint8')
    for i in range(bound_y):#轉換後圖y軸 j
        for j in range(len(img[0])):#轉換後圖x軸 k
            if j<=bound_x or i>=bound_y:
                if i<len(img) and j<len(img[0]):
                    new_img[i][j] = img[i][j]
                continue
            xk = j
            yj = len(img)-1-i
            a = np.asarray([xk,yj,1])
            val = np.dot(M,[xk,yj,1])#[uq,vp,1]
            q = int(val[0]) #寬
            p = int(len(img)-1-val[1]) #高
            if 0<i+t_y<len(img) and 0<j+t_x<len(img[0]):
                    if p>bound_y or q<bound_x:
                        continue
                    if 0<=p<len(img) and 0<=q<len(img[0]):
                        if i+t_y<bound_y and j+t_x>bound_x:
                            new_img[i+t_y][j+t_x] = img[p][q]
                    else:
                        continue
    return new_img

def scaling(img,x_scal,y_scal,bound_x,bound_y_up,bound_y_down):
    M = [[x_scal,0,0],
         [0,y_scal,0],
         [0,0,1]]
    M = np.asarray(M)
    M = np.linalg.inv(M)
    new_img = np.asarray([[255 for i in range(len(img[0]))] for j in range(len(img))])
    new_img = new_img.astype('uint8')
    t_x = None
    t_y = None
    for i in range(len(img)):#轉換後圖y軸 j
        for j in range(len(img[0])):#轉換後圖x軸 k
            if i > bound_y_down or i <bound_y_up or j <bound_x:
                new_img[i][j] = img[i][j]
                continue
            xk = j
            yj = bound_y_down-i
            if yj<0:
                continue
            a = np.asarray([xk,yj,1])
            val = np.dot(M,[xk,yj,1])#[uq,vp,1]
            if t_y == None:
                t_y = a[1]-val[1]
            t_x = a[0]-val[0]
            uq = val[0]+t_x
            vp = val[1]+t_y
            q = int(uq) #寬
            p = int(bound_y_down-vp) #高
            if p<bound_y_up or p>bound_y_down or q<bound_x:
                continue
            else:
                if 0<=p<len(img) and 0<=q<len(img[0]):
                    new_img[i][j] = img[p][q]
                else:
                    continue
    return new_img

img = cv2.imread('./SampleImage/sample3.png',cv2.IMREAD_GRAYSCALE)

x_shift = -40
y_shift = 110
bound_x = 200
bound_y = 599
new_img = translation(img=img,x_shift=x_shift,y_shift=y_shift,bound_x=bound_x,bound_y=bound_y)

x_scal = 1
y_scal = 2.1
bound_x = 200
bound_y_up = 200
bound_y_down = 599
new_img = scaling(img=new_img,x_scal=x_scal,y_scal=y_scal,bound_x=bound_x,bound_y_up=bound_y_up,bound_y_down=bound_y_down)

theta = -25
bound_x = 180
bound_y = 800
t_x=-85
t_y=-130
new_img = rotation(img=new_img,theta=theta,bound_x=bound_x,bound_y=bound_y,t_x=t_x,t_y=t_y)

x_scal = 1
y_scal = 1.1
bound_x = 180
bound_y_up = 200
bound_y_down = 599
new_img = scaling(img=new_img,x_scal=x_scal,y_scal=y_scal,bound_x=bound_x,bound_y_up=bound_y_up,bound_y_down=bound_y_down)
cv2.imshow('result8',new_img)
cv2.imwrite('./result8.png',new_img)

#(b)
def dis(x,y,mid_x,mid_y):
    return ((x-mid_x)**2+(y-mid_y)**2)**0.5

def fisheye(img,mid_x,mid_y,r,deg):
    new_img = img.copy()
    for i in range(len(img)):
        for j in range(len(img[0])):
            x = j
            y = i
            d = dis(x,y,mid_x,mid_y)
            if d <= r:
                u = int(mid_x+((x - mid_x) * (d / r)**deg))
                v = int(mid_y+((y - mid_y) * (d / r)**deg))
                new_img[y, x] = img[v,u]
            else:
                new_img[y, x] = img[y, x]   
    return new_img

img = cv2.imread('./SampleImage/sample5.png',cv2.IMREAD_GRAYSCALE)
mid_x = 150
mid_y = 175
r = 90
deg = 1.2
img = cv2.imread('./SampleImage/sample5.png',cv2.IMREAD_GRAYSCALE)
img = fisheye(img,mid_x,mid_y,r,deg)  
cv2.imshow("result9",img)
cv2.imwrite("./result9.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()





    
    