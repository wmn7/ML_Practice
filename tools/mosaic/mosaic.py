import os
import numpy as np
from PIL import Image, ImageOps
import time
from multiprocessing import Pool
import math
import sys
from colorsys import rgb_to_hsv, hsv_to_rgb


class mosaic(object):
    """定义计算图片的平均hsv值
    """
    def __init__(self, IN_DIR, OUT_DIR, SLICE_SIZE, REPATE, OUT_SIZE):
        self.IN_DIR = IN_DIR  # 原始的图像素材所在文件夹
        self.OUT_DIR = OUT_DIR  # 输出素材的文件夹, 这些都是计算过hsv和经过resize之后的图像
        self.SLICE_SIZE = SLICE_SIZE  # 图像放缩后的大小
        self.REPATE = REPATE  # 同一张图片可以重复使用的次数
        self.OUT_SIZE = OUT_SIZE  # 最终图片输出的大小

    def resize_pic(self, in_name, size):
        """转换图像大小
        """
        img = Image.open(in_name)
        img = ImageOps.fit(img, (size, size), Image.ANTIALIAS)
        return img

    def get_avg_color(self, img):
        """计算图像的平均hsv
        """
        width, height = img.size
        pixels = img.load()
        if type(pixels) is not int:
            data = []  # 存储图像像素的值
            for x in range(width):
                for y in range(height):
                    cpixel = pixels[x, y]  # 获得每一个像素的值
                    data.append(cpixel)
            h = 0
            s = 0
            v = 0
            count = 0
            for x in range(len(data)):
                r = data[x][0]
                g = data[x][1]
                b = data[x][2]  # 得到一个点的GRB三色
                count += 1
                hsv = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                h += hsv[0]
                s += hsv[1]
                v += hsv[2]

            hAvg = round(h / count, 3)
            sAvg = round(s / count, 3)
            vAvg = round(v / count, 3)

            if count > 0:  # 像素点的个数大于0
                return (hAvg, sAvg, vAvg)
            else:
                raise IOError("读取图片数据失败")
        else:
            raise IOError("PIL 读取图片数据失败")


class create_image_db(mosaic):
    """创建所需要的数据
    """
    def __init__(self, IN_DIR, OUT_DIR, SLICE_SIZE, REPATE, OUT_SIZE):
        super(create_image_db, self).__init__(IN_DIR, OUT_DIR, SLICE_SIZE,
                                              REPATE, OUT_SIZE)

    def get_image_paths(self):
        """获取文件夹内图像的地址
        """
        paths = []
        suffixs = ['png', 'jpg']
        for file_ in os.listdir(self.IN_DIR):
            suffix = file_.split('.', 1)[1]  # 获得文件后缀
            if suffix in suffixs:  # 通过后缀判断是否是图片
                paths.append(self.IN_DIR + file_)  # 添加图像路径
            else:
                print("非图片:%s" % file_)
        if len(paths) > 0:
            print("一共找到了%s" % len(paths) + "张图片")
        else:
            raise IOError("未找到任何图片")

        return paths

    def convert_image(self, path):
        """转换图像大小, 同时计算一个图像的平均hsv值.
        """
        img = self.resize_pic(path, self.SLICE_SIZE)
        color = self.get_avg_color(img)
        img.save(str(self.OUT_DIR) + str(color) + ".png")

    def convert_all_images(self):
        """将所有图像进行转换
        """
        paths = self.get_image_paths()
        print("正在生成马赛克块...")
        pool = Pool()  # 多进程处理
        pool.map(self.convert_image, paths)  # 对已有的图像进行处理, 转换为对应的色块
        pool.close()
        pool.join()


class creat_mosaic(mosaic):
    """创建马赛克图片
    """
    def __init__(self, IN_DIR, OUT_DIR, SLICE_SIZE, REPATE, OUT_SIZE):
        super(creat_mosaic, self).__init__(IN_DIR, OUT_DIR, SLICE_SIZE, REPATE,
                                           OUT_SIZE)

    def read_img_db(self):
        """读取所有的图片
        """
        img_db = []  # 存储color_list
        for file_ in os.listdir(self.OUT_DIR):
            if file_ == 'None.png':
                pass
            else:
                file_ = file_.split('.png')[0]  # 获得文件名
                file_ = file_[1:-1].split(',')  # 获得hsv三个值
                file_ = [float(i) for i in file_]
                file_.append(0)  # 最后一位计算图像使用次数
                img_db.append(file_)
        return img_db

    def find_closiest(self, color, list_colors):
        """寻找与像素块颜色最接近的图像
        """
        FAR = 10000000
        for cur_color in list_colors:  # list_color是图像库中所以图像的平均hsv颜色
            n_diff = np.sum((color - np.absolute(cur_color[:3]))**2)
            if cur_color[3] <= self.REPATE:  # 同一个图片使用次数不能太多
                if n_diff < FAR:  # 修改最接近的颜色
                    FAR = n_diff
                    cur_closer = cur_color
        cur_closer[3] += 1
        return "({}, {}, {})".format(cur_closer[0], cur_closer[1],
                                     cur_closer[2])  # 返回hsv颜色

    def make_puzzle(self, img):
        """制作拼图
        """
        img = self.resize_pic(img, self.OUT_SIZE)  # 读取图片并修改大小
        color_list = self.read_img_db()  # 获取所有的颜色的list

        width, height = img.size  # 获得图片的宽度和高度
        print("Width = {}, Height = {}".format(width, height))
        background = Image.new('RGB', img.size,
                               (255, 255, 255))  # 创建一个空白的背景, 之后向里面填充图片
        total_images = math.floor(
            (width * height) / (self.SLICE_SIZE * self.SLICE_SIZE))  # 需要多少小图片
        now_images = 0  # 用来计算完成度
        for y1 in range(0, height, self.SLICE_SIZE):
            for x1 in range(0, width, self.SLICE_SIZE):
                try:
                    # 计算当前位置
                    y2 = y1 + self.SLICE_SIZE
                    x2 = x1 + self.SLICE_SIZE
                    # 截取图像的一小块, 并计算平均hsv
                    new_img = img.crop((x1, y1, x2, y2))
                    color = self.get_avg_color(new_img)
                    # 找到最相似颜色的照片
                    close_img_name = self.find_closiest(color, color_list)
                    close_img_name = self.OUT_DIR + str(
                        close_img_name) + '.png'  # 图片的地址
                    paste_img = Image.open(close_img_name)
                    # 计算完成度
                    now_images += 1
                    now_done = math.floor((now_images / total_images) * 100)
                    r = '\r[{}{}]{}%'.format("#" * now_done,
                                             " " * (100 - now_done), now_done)
                    sys.stdout.write(r)
                    sys.stdout.flush()
                    background.paste(paste_img, (x1, y1))
                except IOError:
                    print('创建马赛克块失败')
        # 保持最后的结果
        background.save('out_without_background.jpg')
        img = Image.blend(background, img, 0.5)
        img.save('out_with_background.jpg')
        return True


if __name__ == "__main__":
    filePath = os.path.dirname(os.path.abspath(__file__))  # 获取当前的路径
    start_time = time.time()  # 程序开始运行时间, 记录一共运行了多久
    # 创建马赛克块, 创建素材库
    createdb = create_image_db(IN_DIR=os.path.join(filePath, 'images/'),
                               OUT_DIR=os.path.join(filePath, 'outputImages/'),
                               SLICE_SIZE=100,
                               REPATE=20,
                               OUT_SIZE=5000)
    createdb.convert_all_images()
    # 创建拼图 (这里使用绝对路径)
    createM = creat_mosaic(IN_DIR=os.path.join(filePath, 'images/'),
                           OUT_DIR=os.path.join(filePath, 'outputImages/'),
                           SLICE_SIZE=100,
                           REPATE=20,
                           OUT_SIZE=5000)
    out = createM.make_puzzle(img=os.path.join(filePath, 'Zelda.jpg'))
    # 打印时间
    print("耗时: %s" % (time.time() - start_time))
    print("已完成")
