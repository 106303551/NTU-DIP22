import cv2
import numpy as np
import random
#Q1.a
def img_pad(img,height,width):

    for i in range(height//2):
        img = np.insert(img,0,img[0],axis=0)
    for i in range(height//2):
        img = np.insert(img,-1,img[-1],axis=0)
    for j in range(width//2):
        img = np.insert(img,0,img[:,0],axis=1)
    for j in range(width//2):
        img = np.insert(img,-1,img[:,-1],axis=1)

    return img



def preserve_same_values(img,arrays):
    result = img.copy()
    for i in range(len(result)):
        for j in range(len(result[i])):
            values = set()
            for arr in arrays:
                values.add(arr[i][j])
            if len(values) == 1:
                result[i][j] = values.pop()
            else:
                result[i][j] = 0
    return result

def struct_cal(img,x_bias,y_bias): #OK

    new_img = img.copy()
    h,w = img.shape
    for i in range(len(img)):
        for j in range(len(img[i])):
            old_i = i-y_bias
            old_j = j-x_bias
            if 0<=old_i<h and 0<=old_j<w:
                new_img[i][j] = img[old_i][old_j]
            else:
                new_img[i][j] = 0
    return new_img

def erosion(img,structure):
    struct_list= []
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            flag = structure[i][j]
            if flag == 1:
                x_bias = j-(len(structure)//2)
                y_bias = i-(len(structure[i])//2)
                struct_img = struct_cal(img,x_bias,y_bias)
                struct_list.append(struct_img)
    result = preserve_same_values(img,struct_list)
    return result

def preserve_all_values(img,arrays):
    result = img.copy()
    for i in range(len(img)):
        for j in range(len(img[i])):
            for arr in arrays:
                value = arr[i][j]
                if value == 255:
                    result[i][j] = value
    return result

def dilation(img,structure):
    struct_list= []
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            flag = structure[i][j]
            if flag == 1:
                x_bias = j-(len(structure)//2)
                y_bias = i-(len(structure[i])//2)
                struct_img = struct_cal(img,x_bias,y_bias)
                struct_list.append(struct_img)

    result = preserve_all_values(img,struct_list)
    return result

img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
structure = [[1,1,1],[1,1,1],[1,1,1]]
img_1 = img_pad(img_1,len(structure),len(structure[0]))
result1 = img_1.copy()
num = 1
for i in range(num):
    result1= erosion(result1,structure)
for i in range(len(img_1)):
    for j in range(len(img_1[i])):
        if img_1[i][j] == result1[i][j]:
            img_1[i][j] = 0
result1 = img_1
cv2.imshow("reuslt1",result1)
cv2.imwrite("./result1.png",result1)

#Q1.b

def hole_fill(img,G,structure,x,y):

    spot_list=[[x,y]]
    #dilation 正常
    struc_list=[]
    bias_list=[]
    count = 0
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            if structure[i][j] == 1:
                struc_G = G.copy()
                x_bias = j-(len(structure)//2)
                y_bias = i-(len(structure[i])//2)
                if x_bias ==-1:
                    struc_G = np.insert(struc_G,-1,struc_G[:,-1],axis=1)
                elif x_bias ==1:
                    struc_G = np.insert(struc_G,0,struc_G[:,0],axis=1)
                if y_bias ==-1:
                    struc_G = np.insert(struc_G,-1,struc_G[-1],axis=0)
                elif y_bias ==1:
                    struc_G = np.insert(struc_G,0,struc_G[0],axis=0)
            else:
                continue
            struc_list.append(struc_G)
            bias_list.append([x_bias,y_bias])
    while(len(spot_list)!=0):
        count+=1
        #創新的G並且padding
        spot = spot_list.pop(0)
        x,y = spot[0],spot[1]
        #去除重疊到object的部分
        for k,struc_G in enumerate(struc_list):
            x_bias,y_bias = bias_list[k][0],bias_list[k][1] 
            for i in range(len(struc_G)):
                for j in range(len(struc_G[i])):
                    if x_bias<=0:
                        real_x = x+(j+x_bias-1)
                    else:
                        real_x = x+(j-x_bias)
                    if y_bias<=0:
                        real_y = y+(i+y_bias-1)
                    else:
                        real_y = y+(i-y_bias)
                    if 0<=real_y<len(img) and 0<=real_x<len(img[0]):
                        if img[real_y][real_x]== 0 and struc_G[i][j] == 255:
                            img[real_y][real_x]=255
                            spot_list.append([real_x,real_y])

    return img

img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
structure = [[0,1,0],[1,1,1],[0,1,0]]
img_1 = img_pad(img_1,len(structure),len(structure[0]))
G_0 = np.zeros((3,3))
G_0[1][1] = 255
result2 = hole_fill(img_1,G_0,structure,0,0)
img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
img_1 = img_pad(img_1,len(structure),len(structure[0]))
for i in range(len(img_1)):
    for j in range(len(img_1[i])):
        if result2[i][j] == 0:
            img_1[i][j] = 255
cv2.imshow("reuslt2",img_1)
cv2.imwrite("./result2.png",img_1)

#Q1.c
def hoshen_kopelman(matrix):

    label_map = np.zeros(matrix.shape)
    label_dict = {}
    neighbors = []
    label = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            if matrix[i, j] != 255:
                continue

            neighbors = []
            if i > 0 and matrix[i - 1, j] == 255:
                neighbors.append(label_map[i - 1, j])
            if j > 0 and matrix[i, j - 1] == 255:
                neighbors.append(label_map[i, j - 1])
            if i>0 and j<len(matrix[0])-1 and matrix[i-1,j+1] == 255:
                neighbors.append(label_map[i-1,j+1])
            if j>0 and i>0 and matrix[i-1,j-1] == 255:
                neighbors.append(label_map[i-1,j-1])

            #無neighbor，更新label
            if not neighbors:
                label += 1
                label_map[i, j] = label
                label_dict[label] = label
            #有neighbor，找最小label
            else:
                min_label = min(neighbors)
                #先找出nieghbor真正對應到的label
                for nei in neighbors:                 
                    while(True):
                        nei = label_dict[nei]
                        if nei == label_dict[nei]:
                            break
                    if nei<min_label:
                        min_label = nei
                label_map[i, j] = min_label

                #將neighbor都指至min label
                for neighbor in neighbors:
                    if neighbor != min_label:
                        label_dict[neighbor] = int(min_label)
    
    #更新label_dict
    keys = label_dict.keys()
    labels = set()
    flag =True
    while(flag == True):
        flag = False
        for k in keys:
            new_k = label_dict[k]
            while label_dict[k] != label_dict[new_k]:
                flag = True
                label_dict[k] = label_dict[new_k]
                new_k = label_dict[k]
    for k in keys:
        labels.add(label_dict[k])
    val_labels=set()
    #將label dict更新至label map
    for i in range(label_map.shape[0]):
        for j in range(label_map.shape[1]):
            if matrix[i, j] == 255:
                while label_map[i, j] != label_dict[label_map[i, j]]:
                    label_map[i, j] = label_dict[label_map[i, j]]
                val_labels.add(label_map[i,j])
        
    return label_map, label_dict,labels


img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
label_img,l_dict,labels = hoshen_kopelman(img_1)

#Q1.d
def random_structure(structure,width):

    for i in range(width):
        structure_list=[]
    for j in range(width):
        val = random.randint(0,1)
        if i ==width//2 and j== width//2:
            val = 1
        structure_list.append(val)
    structure.append(structure_list)

    return structure

def opening(img,structure):
    result = img.copy()
    result = erosion(result,structure)
    result = dilation(result,structure)
    return result

def closing(img,structure):
    result = img.copy()
    result = dilation(result,structure)
    result = erosion(result,structure)
    return result

img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)       
structure = [[0,1,0],[0,1,0],[0,1,0]]  
img_1 = img_pad(img_1,len(structure),len(structure[0]))
result = opening(img_1,structure)
cv2.imshow("result3",result)
cv2.imwrite("result3.png",result)

img_1 = cv2.imread("./SampleImage/sample1.png",cv2.IMREAD_GRAYSCALE)
structure = [[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]]
img_1 = img_pad(img_1,len(structure),len(structure[0]))
result = closing(img_1,structure)
cv2.imshow("result4",result)
cv2.imwrite("result4.png",result)

#Q2.a
def law_conv(ori_img,pad_img,stride,H):
    img = ori_img.copy()
    filter = H
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            filter_area = pad_img[i:i+stride+1,j:j+stride+1].copy()
            filter_area = np.multiply(filter_area,filter)
            new_val = np.sum(filter_area)
            if new_val>255:
                new_val = 255
            img[i,j] = int(new_val)
    return img
def energy_computation(ori_img,pad_img,w):
    #img = np.zeros(ori_img.shape)
    img = ori_img.copy()
    max = 0
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            val = 0
            for m in range(w):
                for n in range(w):
                    val += pad_img[i+m,j+n]
            val=val/(w**2)
            img[i,j] = int(val)
            if max<val:
                max = val
    if max>255:
        scale = 255/max
        img = img[i,j]*scale
        img = img.astype(np.uint8)

    return img
img_2 = cv2.imread("./SampleImage/sample2.png",cv2.IMREAD_GRAYSCALE)
width = 3
stride= 2
pad_img = img_pad(img_2,width,width)
img_2 = cv2.imread("./SampleImage/sample2.png",cv2.IMREAD_GRAYSCALE)

L=1/6*np.asarray([1,2,1])
L=L.reshape(3,1)
E=1/2*np.asarray([-1,0,1])
E=E.reshape(3,1)
S=1/2*np.asarray([1,-2,1])
S=S.reshape(3,1)

l=[L,E,S]
H_list=[]
img_list=[]

#正式用
for l1 in l:
    for l2 in l:
        H=np.matmul(l1,l2.T)
        H_list.append(H)
feature_list=[]
for i in range(len(H_list)):
    file_name = "./texture_conv/image_" + str(i) + ".jpg"
    result = law_conv(img_2,pad_img,stride,H_list[i])
    img_list.append(result)
    cv2.imwrite(file_name,result)

#省時間用
# for i in range(9):
#     file_name = "./texture_conv/image_" + str(i) + ".jpg"
#     img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
#     img_list.append(img)

w=13
feature_list=[]

#正式用
for i in range(len(img_list)):
    file_name = "./texture_feature/image_" + str(i) + ".jpg"
    img = img_list[i]
    pad_img = img_pad(img,w,w)
    result = energy_computation(img,pad_img,w)
    feature_list.append(result)
    cv2.imwrite(file_name,result)

#省時間用
# for i in range(9):
#     file_name = "./texture_feature/image_" + str(i) + ".jpg"
#     img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
#     img_list.append(img)
#     feature_list.append(img)

#Q2.b

def kmeans(data, n_clusters, max_iter,center):

    X = np.reshape(data, (-1, data.shape[-1]))

    for i in range(max_iter):

        distances = np.linalg.norm(X[:, np.newaxis, :] - center, axis=-1)

        labels = np.argmin(distances, axis=-1)

        for j in range(n_clusters):
            center[j] = np.mean(X[labels == j], axis=0)

    final_labels = np.argmin(np.linalg.norm(X[:, np.newaxis, :] - center, axis=-1), axis=-1)
    final_labels = np.reshape(final_labels, data.shape[:-1])

    return final_labels, center

k=2
val=[]
for i in range(k):
    val.append(int(255/k*(i+1)))
X=np.stack(feature_list,axis=-1)
center = np.asarray([X[0,0],X[300,450]])
label,center = kmeans(X,k,100,center)
label = label.astype(np.uint8)
for i in range(len(label)):
    for j in range(len(label[i])):
        label[i,j]=val[label[i,j]]
cv2.imshow("result5",label)
cv2.imwrite("result5.png",label)

#Q2.c

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
size=[10,10]
img_3 = cv2.imread("./result5.png",cv2.IMREAD_GRAYSCALE)
img = median_blur(img_3,size)
cv2.imshow("result6.png",img)
cv2.imwrite("result6.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()