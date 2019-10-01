from PIL import Image
import fire

ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

# 将256灰度映射到70个字符上-即输入一个像素点,输出一个符号
def get_char(r,g,b,alpha = 256):
    if alpha == 0:
        return ' '
    length = len(ascii_char)
    # 将彩色转换为灰度
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)

    unit = (256.0 + 1)/length
    return ascii_char[int(gray/unit)]

def image_process(input_file, output_file='res.txt', pic_height=200, pic_weight=200):
    # 加载图片
    im = Image.open(input_file)
    # 调整图片大小
    im = im.resize((pic_weight,pic_height), Image.NEAREST)
    # 保存缩小后的图片
    im.save('temp.jpg')
    txt = ""
    # 对每个像素点进行操作
    for i in range(pic_height):
        for j in range(pic_weight):
            txt += get_char(*im.getpixel((j,i)))
            txt += ' '
        txt += '\n'

    print(txt)

    #字符画输出到文件
    if output_file:
        with open(output_file,'w') as f:
            f.write(txt)
    else:
        with open("output.txt",'w') as f:
            f.write(txt)   

if __name__ == '__main__':
    """
    input_file: 图像的名称
    output_file: 保存文件的名称
    """
    fire.Fire(image_process)
