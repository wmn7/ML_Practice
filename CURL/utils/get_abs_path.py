'''
@Author: WANG Maonan
@Date: 2023-03-13 20:29:14
@Description: 相对路径转换为绝对路径
@LastEditTime: 2023-03-13 20:29:14
'''
import os
import logging

class getAbsPath(object):

    def __init__(self, file_abspath:str=None) -> None:
        self.file_abspath = file_abspath # 当前文件的完整路径
        self.logger = logging.getLogger(__name__)

    def __call__(self, file_relpath:str):
        """将相对路径转换为绝对路径
        """
        if self.file_abspath == None:
            self.logger.warning('没有设置当前文件路径')
            self.file_abspath = os.path.abspath(__file__)
        
        folder_abspath = os.path.dirname(self.file_abspath)
        return os.path.join(folder_abspath, file_relpath)