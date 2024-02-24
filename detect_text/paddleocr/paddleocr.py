from detect_text.paddleocr.PPOCR_api import GetOcrApi
from detect_text.paddleocr.PPOCR_visualize import visualize
import detect_text.paddleocr.text as Text
import matplotlib.pyplot as plt
from os.path import join as pjoin
import time
import json
import numpy as np
import cv2
ocr = GetOcrApi(r"D:\papers\paddleocr\PaddleOCR-json\PaddleOCR-json_v.1.3.1\PaddleOCR-json_v.1.3.1\PaddleOCR-json.exe")
def ResponseGenerator(TestImagePath):
    #TestImagePath = "D:\papers\pdd-darkpattern\informcollectingnote.jpg"
    # 初始化识别器对象，传入 PaddleOCR-json.exe 的路径。请改成你自己的路径
    print(f'初始化OCR成功，进程号为{ocr.ret.pid}')
    #print('\n测试图片路径：{"D:\papers\pdd-darkpattern\informcollectingnote.jpg"}')
    # 示例1：识别本地图片
    res = ocr.run(TestImagePath)
    print(f'\n response from paddle ocr（original information）：\n{res}')
    #print(f'\n示例1-图片路径识别结果（格式化输出）：')
    #ocr.printResult(res)
    return res

def ExtractTextblock(TestImagePath):
    #ocr = GetOcrApi(r"D:\papers\paddleocr\PaddleOCR-json\PaddleOCR-json_v.1.3.1\PaddleOCR-json_v.1.3.1\PaddleOCR-json.exe")
    print(f'初始化OCR成功，进程号为{ocr.ret.pid}')
    # OCR识别图片，获取文本块
    getObj = ocr.run(TestImagePath)
    ocr.exit()  # 结束引擎子进程
    if not getObj["code"] == 100:
        print('识别失败！！')
        exit()
    textBlocks = getObj["data"]  # 提取文本块数据
    print('显示图片！')
    visualize(textBlocks, TestImagePath).show()
    # 程序阻塞，直到关闭图片浏览窗口才继续往下走。如果长时间不动，注释掉上面这行再跑
    # 示例2：显示更详细的信息
    vis = visualize(textBlocks, TestImagePath)
    print('获取图片！')
    # 禁用包围盒，获取原图片的 PIL Image 对象
    visImg1 = vis.get(isBox=False)
    # 启用文本和序号、禁用原图（显示透明背景），获取 PIL Image 对象
    visImg2 = vis.get(isText=True, isOrder=True, isSource=False)
    # 获取两个图片的左右对比，左边是原图，右边是单独的文本框
    vis = visualize.createContrast(visImg1, visImg2)
    # 显示该对比
    vis.show()
    # 接下来可以还用PIL库对visImg进一步处理。
    # 保存到本地
    start = int(time.perf_counter())
    print(f"保存图片到 {os.path.dirname('./data/output/ocr')}\\{start}.png ")
    vis.save(f"{os.path.dirname('./data/output/ocr/')}\\可视化结果.png", isText=True)
    
    print('程序结束。')


class Text:
    def __init__(self, id, content, location):
        self.id = id
        self.content = content
        self.location = location

        self.width = self.location['right'] - self.location['left']
        self.height = self.location['bottom'] - self.location['top']
        self.area = self.width * self.height
        self.word_width = self.width / len(self.content)
    def __str__(self):
        return str(self.id)+","+str(self.content)+","+str(self.location)
    '''
    ********************************
    *** Relation with Other text ***
    ********************************
    '''
    def is_justified(self, ele_b, direction='h', max_bias_justify=4):
        '''
        Check if the element is justified
        :param max_bias_justify: maximum bias if two elements to be justified
        :param direction:
             - 'v': vertical up-down connection
             - 'h': horizontal left-right connection
        '''
        l_a = self.location
        l_b = ele_b.location
        # connected vertically - up and below
        if direction == 'v':
            # left and right should be justified
            if abs(l_a['left'] - l_b['left']) < max_bias_justify and abs(l_a['right'] - l_b['right']) < max_bias_justify:
                return True
            return False
        elif direction == 'h':
            # top and bottom should be justified
            if abs(l_a['top'] - l_b['top']) < max_bias_justify and abs(l_a['bottom'] - l_b['bottom']) < max_bias_justify:
                return True
            return False

    def is_on_same_line(self, text_b, direction='h', bias_gap=4, bias_justify=4):
        '''
        Check if the element is on the same row(direction='h') or column(direction='v') with ele_b
        :param direction:
             - 'v': vertical up-down connection
             - 'h': horizontal left-right connection
        :return:
        '''
        l_a = self.location
        l_b = text_b.location
        # connected vertically - up and below
        if direction == 'v':
            # left and right should be justified
            if self.is_justified(text_b, direction='v', max_bias_justify=bias_justify):
                # top and bottom should be connected (small gap)
                if abs(l_a['bottom'] - l_b['top']) < bias_gap or abs(l_a['top'] - l_b['bottom']) < bias_gap:
                    return True
            return False
        elif direction == 'h':
            # top and bottom should be justified
            if self.is_justified(text_b, direction='h', max_bias_justify=bias_justify):
                # top and bottom should be connected (small gap)
                if abs(l_a['right'] - l_b['left']) < bias_gap or abs(l_a['left'] - l_b['right']) < bias_gap:
                    return True
            return False

    def is_intersected(self, text_b, bias):
        l_a = self.location
        l_b = text_b.location
        left_in = max(l_a['left'], l_b['left']) + bias
        top_in = max(l_a['top'], l_b['top']) + bias
        right_in = min(l_a['right'], l_b['right'])
        bottom_in = min(l_a['bottom'], l_b['bottom'])

        w_in = max(0, right_in - left_in)
        h_in = max(0, bottom_in - top_in)
        area_in = w_in * h_in
        if area_in > 0:
            return True

    '''
    ***********************
    *** Revise the Text ***
    ***********************
    '''
    def merge_text(self, text_b):
        text_a = self
        top = min(text_a.location['top'], text_b.location['top'])
        left = min(text_a.location['left'], text_b.location['left'])
        right = max(text_a.location['right'], text_b.location['right'])
        bottom = max(text_a.location['bottom'], text_b.location['bottom'])
        self.location = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
        self.width = self.location['right'] - self.location['left']
        self.height = self.location['bottom'] - self.location['top']
        self.area = self.width * self.height

        left_element = text_a
        right_element = text_b
        if text_a.location['left'] > text_b.location['left']:
            left_element = text_b
            right_element = text_a
        self.content = left_element.content + ' ' + right_element.content
        self.word_width = self.width / len(self.content)

    def shrink_bound(self, binary_map):
        bin_clip = binary_map[self.location['top']:self.location['bottom'], self.location['left']:self.location['right']]
        height, width = np.shape(bin_clip)

        shrink_top = 0
        shrink_bottom = 0
        for i in range(height):
            # top
            if shrink_top == 0:
                if sum(bin_clip[i]) == 0:
                    shrink_top = 1
                else:
                    shrink_top = -1
            elif shrink_top == 1:
                if sum(bin_clip[i]) != 0:
                    self.location['top'] += i
                    shrink_top = -1
            # bottom
            if shrink_bottom == 0:
                if sum(bin_clip[height-i-1]) == 0:
                    shrink_bottom = 1
                else:
                    shrink_bottom = -1
            elif shrink_bottom == 1:
                if sum(bin_clip[height-i-1]) != 0:
                    self.location['bottom'] -= i
                    shrink_bottom = -1

            if shrink_top == -1 and shrink_bottom == -1:
                break

        shrink_left = 0
        shrink_right = 0
        for j in range(width):
            # left
            if shrink_left == 0:
                if sum(bin_clip[:, j]) == 0:
                    shrink_left = 1
                else:
                    shrink_left = -1
            elif shrink_left == 1:
                if sum(bin_clip[:, j]) != 0:
                    self.location['left'] += j
                    shrink_left = -1
            # right
            if shrink_right == 0:
                if sum(bin_clip[:, width-j-1]) == 0:
                    shrink_right = 1
                else:
                    shrink_right = -1
            elif shrink_right == 1:
                if sum(bin_clip[:, width-j-1]) != 0:
                    self.location['right'] -= j
                    shrink_right = -1

            if shrink_left == -1 and shrink_right == -1:
                break
        self.width = self.location['right'] - self.location['left']
        self.height = self.location['bottom'] - self.location['top']
        self.area = self.width * self.height
        self.word_width = self.width / len(self.content)

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def visualize_element(self, img, color=(0, 0, 255), line=1, show=False):
        loc = self.location
        cv2.rectangle(img, (loc['left'], loc['top']), (loc['right'], loc['bottom']), color, line)
        if show:
            print(self.content)
            cv2.imshow('text', img)
            cv2.waitKey()
            cv2.destroyWindow('text')
