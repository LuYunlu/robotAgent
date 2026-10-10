import hashlib
import os.path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document

from utils.logger_handler import logger


def get_file_md5_hex(filepath:str) :
    """
        计算文件的MD5哈希值，返回十六进制字符串
        :param filepath: 文件的绝对/相对路径
        :return: 成功返回32位MD5十六进制字符串，失败返回None
    """
    if not os.path.exists(filepath):
        logger.error(f"[md5计算] 文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算] 路径{filepath}不是文件")
        return

    md5_obj=hashlib.md5()

    chunk_size=4096  # 4KB分片，可根据需求调整
    try:
        with open(filepath,"rb") as f:  # 必须以二进制模式打开
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
        md5_hex=md5_obj.hexdigest()
        return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}的md5失败。{str(e)}")


def listdir_with_allowed_type(path:str, allowed_types:tuple[str]):
    """
       返回允许的文件后缀
    """
    files=[]
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] {path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path,f))
    return tuple(files)

def csv_loader(filepath:str, source_column=None, encoding="utf-8", csv_args=None) -> list[Document]:
    loader=CSVLoader(
        filepath,
        source_column=source_column,
        encoding=encoding,
        csv_args=csv_args
    )
    return loader.load()

def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath:str) -> list[Document]:
    return TextLoader(filepath,encoding="utf-8").load()