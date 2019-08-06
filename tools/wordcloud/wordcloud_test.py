from os import path
from wordcloud import WordCloud
from PIL import Image
import numpy as np

d = path.dirname(__file__)

# Read the whole text.
text = open(path.join(d, 'constitution.txt'), encoding='utf8').read()

# 导入字体文件
font_path = path.join(d, 'HYC6GFM.TTF')

# 生成普通的wordcloud
wordcloud = WordCloud(font_path=font_path, margin=1, random_state=1, max_words=300, 
                        width=1000, height=700, background_color='white').generate(text)

wordcloud.to_file('wordcloud.jpg')

# 生成带有mask的图片
mask = np.array(Image.open(path.join(d, "62.jpg")))
wordcloud = WordCloud(font_path=font_path, mask=mask, margin=1, random_state=1, 
                      background_color='white').generate(text)

wordcloud.to_file('wordcloud_mask.jpg')