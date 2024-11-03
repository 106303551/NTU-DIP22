from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
import igraph as ig
from argparse import ArgumentParser, Namespace
from border_matting import BorderMatting as bm
import warnings
import time
warnings.filterwarnings('ignore') 
#init
bg={'val':1}
fg={'val':2}
prob_bg={'val':3}
prob_fg={'val':4}
RECTANGLE_COLOR = (0,255,0)
BG_COLOR = (0,0,0) #black
FG_COLOR = (255,255,255) #white
PROB_BG_COLOR = (255,0,0) #blue
PROB_FG_COLOR = (0,0,255) #red

def parse_args() -> Namespace:

    parser = ArgumentParser()
    parser.add_argument(
        "--img_file",
        type=str,
        help="Path to img",
        default="sample1.jpg"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="output's name",
        default="output.jpg"
    )
    args = parser.parse_known_args()[0]
    return args

class GrabCut:
    def __init__(self,img,mask,rect=None,gmm_components=5,iter_num=3):
        self.img = np.asarray(img, dtype=np.float64)
        self.ori_img = img
        self.rows,self.cols,_=img.shape
        self.img_idx = np.arange(self.rows * self.cols,dtype=np.uint32).reshape(self.rows, self.cols)
        self.mask = mask.copy()
        self.flat_mask = mask.reshape(-1)
        self.rect = rect
        if rect is not None:
            self.mask[self.rect[1]:self.rect[1]+self.rect[3],self.rect[0]:self.rect[0]+self.rect[2]]=prob_fg['val']
        self.gmm_components = gmm_components
        self.f_gmm = None
        self.b_gmm = None
        self.comp_idxs = np.empty((self.rows, self.cols), dtype=np.uint32)
        self.f_idx=[]
        self.b_idx=[]
        self.beta = 0
        self.gamma = 50
        self.iter_num = iter_num
        #for graph-cut
        self.gc_graph = None
        self.gc_graph_capacity = None #Edge capacities
        self.gc_source = self.cols*self.rows #"object" terminal S
        self.gc_sink = self.gc_source+1 #"background" terminal T
        self.update_mask(mask)
        self.calc_beta_smoothness()
        self.init_gmm()

    def update_mask(self,fg_bg_mask):
        b = np.where(np.logical_or(self.mask == 0,fg_bg_mask == bg['val']))
        f = np.where(fg_bg_mask == fg['val'])
        self.mask[f] = fg['val']
        self.mask[b] = bg['val']
        self.flat_mask = self.mask.reshape(-1)

    def calc_beta_smoothness(self):
        print('--calc beta and smoothness--')
        left_ur = self.img[1:, :-1] - self.img[:-1, 1:]
        left_d = self.img[:, 1:] - self.img[:, :-1]
        left_u = self.img[1:, :] - self.img[:-1, :]
        left_ul = self.img[1:, 1:] - self.img[:-1, :-1]

        self.beta = np.sum(np.square(left_d)) + np.sum(np.square(left_ul)) + np.sum(np.square(left_u)) + np.sum(np.square(left_ur))
        neighbor_num = int((left_d.size+left_ul.size+left_u.size+left_ur.size)/3)
        self.beta = 1/(2*self.beta)*neighbor_num

        # print('Beta:', self.beta)
        # Smoothness
        self.left_V = self.gamma * np.exp(-self.beta * np.sum(np.square(left_d), axis=2))
        self.upleft_V = self.gamma * np.exp(-self.beta * np.sum(np.square(left_ul), axis=2))/ np.sqrt(2)
        self.up_V = self.gamma * np.exp(-self.beta * np.sum(np.square(left_u), axis=2))
        self.upright_V = self.gamma * np.exp(-self.beta * np.sum(np.square(left_ur), axis=2))/ np.sqrt(2)
        

    def update_idx(self):
        print('--update idx--')
        self.f_idx = np.where(np.logical_or(self.mask == fg['val'],self.mask == prob_fg['val']))
        self.b_idx = np.where(np.logical_or(self.mask == bg['val'],self.mask== prob_bg['val']))

    def init_gmm(self):
        print('--init_gmm--')
        self.f_gmm = GaussianMixture(n_components=self.gmm_components,max_iter=1)
        self.b_gmm = GaussianMixture(n_components=self.gmm_components,max_iter=1)
    def learn_gmm(self):
        print('--learn gmm--')
        self.comp_idxs[self.f_idx] = self.f_gmm.fit_predict(self.img[self.f_idx])
        self.comp_idxs[self.b_idx] = self.b_gmm.fit_predict(self.img[self.b_idx])
    
    #graph-cut
    def construct_gc_graph(self):
        print('--建構graph--')
        #找出三種state的pixel的idx
        flat_bg_idx = np.where(self.flat_mask == bg['val'])
        flat_fg_idx = np.where(self.flat_mask == fg['val'])
        flat_pr_idx = np.where(np.logical_or(self.flat_mask == prob_bg['val'], self.flat_mask == prob_fg['val']))

        # print('背景pixel數量: %d, 前景pixel數量: %d, 未定狀態pixel數量: %d' % (len(flat_bg_idx[0]), len(flat_fg_idx[0]), len(flat_pr_idx[0])))

        edges = []
        self.gc_graph_capacity = []

        # t-links(根據graph-cut定義之link權重)
        edges.extend(
            list(zip([self.gc_source] * flat_pr_idx[0].size, flat_pr_idx[0])))
        _D = -self.b_gmm.score_samples(self.img.reshape(-1,3)[flat_pr_idx])
        self.gc_graph_capacity.extend(_D.tolist())

        edges.extend(
            list(zip([self.gc_sink] * flat_pr_idx[0].size, flat_pr_idx[0])))
        _D = -self.f_gmm.score_samples(self.img.reshape(-1,3)[flat_pr_idx])
        self.gc_graph_capacity.extend(_D.tolist())

        edges.extend(
            list(zip([self.gc_source] * flat_bg_idx[0].size, flat_bg_idx[0])))
        self.gc_graph_capacity.extend([0] * flat_bg_idx[0].size)

        edges.extend(
            list(zip([self.gc_sink] * flat_bg_idx[0].size, flat_bg_idx[0])))
        self.gc_graph_capacity.extend([9 * self.gamma] * flat_bg_idx[0].size)

        edges.extend(
            list(zip([self.gc_source] * flat_fg_idx[0].size, flat_fg_idx[0])))
        self.gc_graph_capacity.extend([9 * self.gamma] * flat_fg_idx[0].size)

        edges.extend(
            list(zip([self.gc_sink] * flat_fg_idx[0].size, flat_fg_idx[0])))
        self.gc_graph_capacity.extend([0] * flat_fg_idx[0].size)


        # n-links(neighborhood)(根據graph-cut定義之link權重)
        self.img_idx = np.arange(self.rows * self.cols,dtype=np.uint32).reshape(self.rows, self.cols)
        #mask用於考慮edge case
        mask1 = self.img_idx[:, 1:].reshape(-1)
        mask2 = self.img_idx[:, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.left_V.reshape(-1).tolist())

        mask1 = self.img_idx[1:, 1:].reshape(-1)
        mask2 = self.img_idx[:-1, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.upleft_V.reshape(-1).tolist())

        mask1 = self.img_idx[1:, :].reshape(-1)
        mask2 = self.img_idx[:-1, :].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.up_V.reshape(-1).tolist())

        mask1 = self.img_idx[1:, :-1].reshape(-1)
        mask2 = self.img_idx[:-1, 1:].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.upright_V.reshape(-1).tolist())
        #+2代表加入source及sink
        self.gc_graph = ig.Graph(self.cols * self.rows + 2)
        self.gc_graph.add_edges(edges)

    def segmentation(self):
        print('--segmentation--')
        mincut = self.gc_graph.st_mincut(self.gc_source, self.gc_sink, self.gc_graph_capacity)
        pr_indexes = np.where(np.logical_or(self.mask == prob_bg['val'], self.mask == prob_fg['val']))
        self.img_idx = np.arange(self.rows * self.cols,dtype=np.uint32).reshape(self.rows, self.cols)
        self.mask[pr_indexes] = np.where(np.isin(self.img_idx[pr_indexes], mincut.partition[0]),prob_fg['val'], prob_bg['val'])
        self.update_idx()

    '''
    def show_img(self):
        output_mask = np.where((self.mask==fg['val'])+(self.mask==prob_fg['val']),255,0).astype('uint8')
        output_mask[np.where(self.mask==bg['val'])] = 0
        output = cv2.bitwise_and(self.ori_img,self.ori_img,mask=output_mask)
        cv2.imshow('img',output)
    '''

    def return_img(self):
        output_mask = np.where((self.mask==fg['val'])+(self.mask==prob_fg['val']),255,0).astype('uint8')
        output_mask[np.where(self.mask==bg['val'])] = 0
        output = cv2.bitwise_and(self.ori_img,self.ori_img,mask=output_mask)
        return output
    
    def execute_all(self):
        self.update_idx()
        self.learn_gmm()
        self.construct_gc_graph()
        self.segmentation()

    def border_matting(self):
        newmask = np.copy(self.mask)
        newmask[self.mask == 1] = 0
        newmask[self.mask == 2] = 4
        newmask[self.mask == 3] = 0
        newmask[self.mask == 4] = 4
        test = bm(self.ori_img, newmask).run()
        out = np.zeros_like(self.ori_img)
        m,n,k = self.ori_img.shape
        for i in range(m):
            for j in range(n):
                out[i][j][0] = test[i][j] * self.ori_img[i][j][0]
                out[i][j][1] = test[i][j] * self.ori_img[i][j][1]
                out[i][j][2] = test[i][j] * self.ori_img[i][j][2]
        return out

    # def save(self):
    #     output_mask = np.where((self.ori_mask==1)+(self.mask==3),255,0).astype('uint8')
    #     output = cv2.bitwise_and(self.ori_img,self.ori_img,mask=output_mask)
    #     cv2.imwrite(args.output_file,output)

process_record_list = []

def draw(event,x,y,flag,param):
    if variable_saving_dict['process_state'] == 'init rectangle':
        if event == cv2.EVENT_LBUTTONDOWN:
            variable_saving_dict['rectangle_begin'] = True
            variable_saving_dict['rectangle_begin_x'] = x
            variable_saving_dict['rectangle_begin_y'] = y
        elif flag == cv2.EVENT_FLAG_LBUTTON:
            if variable_saving_dict['rectangle_begin']:
                begin_x = variable_saving_dict['rectangle_begin_x']
                begin_y = variable_saving_dict['rectangle_begin_y']
                variable_saving_dict['rectangle_end_x'] = x
                variable_saving_dict['rectangle_end_y'] = y
                rectangle = (min(begin_x,x),min(begin_y,y),abs(begin_x-x),abs(begin_y-y))
                variable_saving_dict['rectangle'] = rectangle
                img = variable_saving_dict['tmp_img'].copy()
                img = cv2.rectangle(img,(begin_x,begin_y),(x, y),RECTANGLE_COLOR,2)
                cv2.imshow('img',img)
        if event == cv2.EVENT_LBUTTONUP:
            if variable_saving_dict['rectangle_begin']:
                variable_saving_dict['rectangle_begin'] = False
                begin_x = variable_saving_dict['rectangle_begin_x']
                begin_y = variable_saving_dict['rectangle_begin_y']
                variable_saving_dict['rectangle_end_x'] = x
                variable_saving_dict['rectangle_end_y'] = y
                rectangle = (min(begin_x,x),min(begin_y,y),abs(begin_x-x),abs(begin_y-y))
                variable_saving_dict['rectangle'] = rectangle
                img = variable_saving_dict['tmp_img'].copy()
                img = cv2.rectangle(img,(begin_x,begin_y),(x, y),RECTANGLE_COLOR,2)
                cv2.imshow('img',img)
    if variable_saving_dict['process_state'] == 'draw brush' or variable_saving_dict['process_state'] == 'init brush':
        if event == cv2.EVENT_LBUTTONDOWN:
            variable_saving_dict['brush_begin'] = True
            img = variable_saving_dict['tmp_img']
            mask = variable_saving_dict['tmp_mask']
            cv2.circle(img,(x,y),variable_saving_dict['brush_size'],variable_saving_dict['brush_color'],-1)
            cv2.circle(mask,(x,y),variable_saving_dict['brush_size'],variable_saving_dict['brush_index'],-1)
            if variable_saving_dict['process_state'] == 'init brush':
                if 'rectangle' in variable_saving_dict:
                    img = img.copy()
                    begin_x = variable_saving_dict['rectangle_begin_x']
                    begin_y = variable_saving_dict['rectangle_begin_y']
                    end_x = variable_saving_dict['rectangle_end_x']
                    end_y = variable_saving_dict['rectangle_end_y']
                    img = cv2.rectangle(img,(begin_x,begin_y),(end_x,end_y),RECTANGLE_COLOR,2)
            cv2.imshow('img',img)
        elif flag == cv2.EVENT_FLAG_LBUTTON:
            if variable_saving_dict['brush_begin']:
                img = variable_saving_dict['tmp_img']
                mask = variable_saving_dict['tmp_mask']
                cv2.circle(img,(x,y),variable_saving_dict['brush_size'],variable_saving_dict['brush_color'],-1)
                cv2.circle(mask,(x,y),variable_saving_dict['brush_size'],variable_saving_dict['brush_index'],-1)
                if variable_saving_dict['process_state'] == 'init brush':
                    if 'rectangle' in variable_saving_dict:
                        img = img.copy()
                        begin_x = variable_saving_dict['rectangle_begin_x']
                        begin_y = variable_saving_dict['rectangle_begin_y']
                        end_x = variable_saving_dict['rectangle_end_x']
                        end_y = variable_saving_dict['rectangle_end_y']
                        img = cv2.rectangle(img,(begin_x,begin_y),(end_x,end_y),RECTANGLE_COLOR,2)
                cv2.imshow('img',img)
        elif event == cv2.EVENT_LBUTTONUP:
            if variable_saving_dict['brush_begin']:
                variable_saving_dict['brush_begin'] = False
                img = variable_saving_dict['tmp_img']
                mask = variable_saving_dict['tmp_mask']
                cv2.circle(img,(x,y),variable_saving_dict['brush_size'],variable_saving_dict['brush_color'],-1)
                cv2.circle(mask,(x,y),variable_saving_dict['brush_size'],variable_saving_dict['brush_index'],-1)
                if variable_saving_dict['process_state'] == 'init brush':
                    if 'rectangle' in variable_saving_dict:
                        img = img.copy()
                        begin_x = variable_saving_dict['rectangle_begin_x']
                        begin_y = variable_saving_dict['rectangle_begin_y']
                        end_x = variable_saving_dict['rectangle_end_x']
                        end_y = variable_saving_dict['rectangle_end_y']
                        img = cv2.rectangle(img,(begin_x,begin_y),(end_x,end_y),RECTANGLE_COLOR,2)
                cv2.imshow('img',img)
                

if __name__ == '__main__':
    args=parse_args()
    ori_img = cv2.imread(args.img_file)
    process_record_list.append(ori_img.copy())
    #目前設定width固定800
    ratio =ori_img.shape[0]/ori_img.shape[1]
    new_width =800
    new_height = int(new_width*ratio)
    ori_img = cv2.resize(ori_img, (new_width, new_height))

    img = ori_img.copy()
    rect_img = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #設定視窗大小
    cv2.resizeWindow('img', new_width, new_height)
    cv2.imshow('img',ori_img)
    cv2.setMouseCallback('img',draw)

    variable_saving_dict = {
        'process_state':'init rectangle',
        'now_img':img.copy(),
        'tmp_img':img.copy(),
        'now_mask':mask.copy(),
        'tmp_mask':mask.copy(),
        'brush_size':3,
        'ori_imgsize':img.shape,
        'ratio':ratio
    }
    gc = None
    while(1):
        k = cv2.waitKey(0)
        if k == 27:  # esc - exit
            if gc:
                cv2.imwrite(args.output_file,process_record_list[-1])
            break
        #use brush
        if k == ord('b'):
            if variable_saving_dict['process_state'] != 'draw brush':
                variable_saving_dict['process_state'] = 'init brush'
                print('brush mode')
                if 'brush_color' not in variable_saving_dict:
                    variable_saving_dict['brush_color'] = BG_COLOR
                    variable_saving_dict['brush_index'] = 1
                    print('brush:bg')
        if k == ord('v'):
            if variable_saving_dict['process_state'] != 'draw brush':
                variable_saving_dict['process_state'] = 'init rectangle'
                print('rect mode')
        if k == ord('n'):  # n - end draw and ask model to resolve
            if variable_saving_dict['process_state'] != 'draw brush':
                if 'rectangle' in variable_saving_dict:
                    print('--------------------')
                    variable_saving_dict['brush_color'] = BG_COLOR
                    variable_saving_dict['brush_index'] = 1
                    variable_saving_dict['process_state'] = 'draw brush'
                    mask = variable_saving_dict['tmp_mask']
                    rect = variable_saving_dict['rectangle']
                    gc = GrabCut(img,mask,rect,5,0)
                    gc.execute_all()
                    masked_image = gc.return_img().copy()
                    showed_image = cv2.addWeighted(ori_img,0.25,masked_image,0.75,0)
                    cv2.imshow('img',showed_image)
                    process_record_list.append(showed_image.copy())
                    variable_saving_dict['now_img'] = showed_image.copy()
                    variable_saving_dict['now_mask'] = gc.mask.copy()
                    variable_saving_dict['tmp_img'] = showed_image.copy()
                    variable_saving_dict['tmp_mask'] = gc.mask.copy()
                    print('done')
                    print('--------------------')
                    print('start user editing')
                    print('brush:bg')
            else:
                print('--------------------')
                mask = variable_saving_dict['tmp_mask']
                process_record_list.append(variable_saving_dict['tmp_img'].copy())
                gc.update_mask(mask)
                gc.execute_all()
                masked_image = gc.return_img().copy()
                showed_image = cv2.addWeighted(ori_img,0.25,masked_image,0.75,0)
                cv2.imshow('img',showed_image)
                process_record_list.append(showed_image.copy())
                variable_saving_dict['now_img'] = showed_image.copy()
                variable_saving_dict['now_mask'] = gc.mask.copy()
                variable_saving_dict['tmp_img'] = showed_image.copy()
                variable_saving_dict['tmp_mask'] = gc.mask.copy()
                print('done')
                print('--------------------')
        elif k == ord('1'):  # 1 for bg
            if variable_saving_dict['process_state'] != 'init rectangle':
                print("brush:bg")
                variable_saving_dict['brush_color'] = BG_COLOR
                variable_saving_dict['brush_index'] = bg['val']
        elif k == ord('2'):  # 2 for fg
            if variable_saving_dict['process_state'] != 'init rectangle':
                print("brush:fg")
                variable_saving_dict['brush_color'] = FG_COLOR
                variable_saving_dict['brush_index'] = fg['val']
        elif k == ord('3'):  # 3 for probable bg
            if variable_saving_dict['process_state'] != 'init rectangle':
                print("brush:prob bg")
                variable_saving_dict['brush_color'] = PROB_BG_COLOR
                variable_saving_dict['brush_index'] = prob_bg['val']
        elif k == ord('4'):  # 4 for probable fg
            if variable_saving_dict['process_state'] != 'init rectangle':
                print("brush:prob fg")
                variable_saving_dict['brush_color'] = PROB_FG_COLOR
                variable_saving_dict['brush_index'] = prob_fg['val']
        elif k == ord('<'):  # up for thin brush
            if variable_saving_dict['process_state'] != 'init rectangle':
                variable_saving_dict['brush_size'] = max(1,variable_saving_dict['brush_size']-1)
                print("brush size:"+str(variable_saving_dict['brush_size']))
        elif k == ord('>'):  # > for thick brush
            if variable_saving_dict['process_state'] != 'init rectangle':
                variable_saving_dict['brush_size'] = min(10,variable_saving_dict['brush_size']+1)
                print("brush size:"+str(variable_saving_dict['brush_size']))
        elif k == ord('s'): # s - 儲存當前brush
            print('--save state--')
            variable_saving_dict['now_img'] = variable_saving_dict['tmp_img'].copy()
            variable_saving_dict['now_mask'] = variable_saving_dict['tmp_mask'].copy()
            if 'rectangle' in variable_saving_dict:
                variable_saving_dict['saved_rectangle'] = variable_saving_dict['rectangle']
        elif k == ord('z'): # z - 回到上次儲存結果
            print('--last record--')
            variable_saving_dict['tmp_img'] = variable_saving_dict['now_img'].copy()
            variable_saving_dict['tmp_mask'] = variable_saving_dict['now_mask'].copy()
            img = variable_saving_dict['tmp_img']
            if variable_saving_dict['process_state'] != 'draw brush':
                if 'saved_rectangle' in variable_saving_dict:
                    variable_saving_dict['rectangle'] = variable_saving_dict['saved_rectangle']
                    variable_saving_dict['begin_x'] = variable_saving_dict['rectangle'][0]
                    variable_saving_dict['begin_y'] = variable_saving_dict['rectangle'][1]
                    variable_saving_dict['end_x'] = variable_saving_dict['rectangle'][2]
                    variable_saving_dict['end_y'] = variable_saving_dict['rectangle'][3]
                    rect = variable_saving_dict['rectangle']
                    img = img.copy()
                    img = cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),RECTANGLE_COLOR,2)
                elif 'rectangle' in variable_saving_dict:
                    del variable_saving_dict['rectangle']
            cv2.imshow('img',img)
        elif k == ord('m'):
            if variable_saving_dict['process_state'] == 'draw brush':
                #matted_image = gc.border_matting()
                #showed_image = cv2.addWeighted(ori_img,0.25,matted_image,0.75,0)
                showed_image = gc.border_matting()
                cv2.imshow('img',showed_image)
                process_record_list.append(showed_image.copy())
                variable_saving_dict['now_img'] = showed_image.copy()
                variable_saving_dict['now_mask'] = gc.mask.copy()
                variable_saving_dict['tmp_img'] = showed_image.copy()
                variable_saving_dict['tmp_mask'] = gc.mask.copy()
                print('done')
        elif k == ord('r'): # r - 重置圖片
            print('--restart pic--')
            print('--------------------')
            variable_saving_dict = {
                'process_state': 'draw rectangle',
                'now_img':ori_img.copy(),
                'tmp_img':ori_img.copy(),
                'now_mask':mask.copy(),
                'tmp_mask':mask.copy()
                }
            process_record_list = [process_record_list[0]]
            cv2.imshow('img',ori_img)
        elif k == ord('a'): # a - 顯示過程
            print('--show process--')
            for i in range(len(process_record_list)):
                print('--show step--'+str(i+1))
                cv2.imshow('record',process_record_list[i])
                cv2.waitKey(1000)
            print('--press any key to close record--')
            cv2.waitKey(0)
            cv2.destroyWindow('record')
            print('--close record--')
        elif k == ord('h'): # h - 顯示指令
            print('--commands                                     --')
            print('--b: brush mode(only in first step)            --')
            print('--v: rect mode(only in first step)             --')
            print('--n: generate next picture                     --')
            print('--1: change to background brush                --')
            print('--2: change to foreground brush                --')
            print('--3: change to possible background brush       --')
            print('--4: change to possible foreground brush       --')
            print('--<: thin the brush(minimum is 1)              --')
            print('-->: thick the brush(maximum is 10)            --')
            print('--s: save current mark                         --')
            print('--z: back to the last draw record              --')
            print('--m: discard all draws and do border matting   --')
            print('--   (it will take a few minutes)              --')
            print('--r: restart                                   --')
            print('--a: show previous process                     --')
            print('--h: show commands                             --')
            print('--esc: exit and output image                   --')



