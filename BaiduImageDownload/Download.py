import numpy as np
import hashlib
import requests
import json
import cv2
import os

#更改工作路径到当前
curDir = os.path.dirname(os.path.abspath(__file__))
#os.chdir(curDir)

def evalMd5(sentence,charset='utf8'):
    '''
    计算一段字符串的md5
    :param sentence: 字符串
    :param charset: 字符集
    :return: md5值
    '''
    #将字符串编码成bytes
    if type(sentence) != bytes:
        sentence = sentence.encode(charset)
    md5 = hashlib.md5(sentence).hexdigest()
    return md5

def resizeImg(oldPath,size,newPath):
    '''
    重定图片尺寸
    :param oldPath: 图片路径
    :param size: 重定大小
    :param newPath: 图片保存路径
    :return: None      
    '''
    oldPath = oldPath.replace('\\','/')
    newPath = newPath.replace('\\','/')
    oldImg = cv2.imdecode(np.fromfile(oldPath,dtype=np.uint8),-1)
    try:
        newImg = cv2.resize(oldImg,size,) #为图片重新指定尺寸
        cv2.imwrite(newPath,newImg)
        cv2.imencode('.'+newPath.split('.')[-1],newImg)[1].tofile(newPath)
    except:
        #图片格式不对发生错误，删除
        os.remove(oldPath)

def download(keyWord,imgNumber,imgSize=None):
    '''
    下载图片到关键词文件夹
    :param keyWord: 关键词
    :param imgNumber: 图片数量
    :param imgSize: 图片重定大小
    :return: None   
    '''
    #创建关键词文件夹
    dirname = os.path.join(curDir,keyWord)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    #开始爬图片
    url = 'https://image.baidu.com/search/acjson'#图片网址
    same = 0#重复下载数
    error = 0#错误数
    passNum = 0#无链接数
    for i in range(30,30*10000+30,30):
        param = {
            'tn': 'resultjson_com','ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'queryWord': keyWord,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': -1,
            'z': '',
            'ic': 0,
            'word': keyWord,
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': 0,
            'istype': 2,
            'qc': '',
            'nc': 1,
            'fr': '',
            'pn': i,
            'rn': 30,
            'gsm': '1e',
            '1488942260214': ''
            }
        #所有图片地址列表
        data = requests.get(url,params=param).text.replace('\\','\\\\')
        try:
            data = json.loads(data)['data']
        except:
            #json数据可能不合法,直接跳过
            error += 1
            if error >=20:
                return None
            continue
        
        for item in data:
            imgUrl = item.get("middleURL")#图片地址
            if passNum>=20:
                return None
            if imgUrl is None:
                passNum+=1
                continue
            suffix = imgUrl.split('.')[-1]#图片后缀
            imgContent = requests.get(imgUrl).content#图片内容
            imgMd5 = evalMd5(imgContent)#图片md5
            imgPath = os.path.join(dirname,'%s.%s'%(imgMd5,suffix))#图片路径
            oldFinish = len(os.listdir(dirname))
            open(imgPath, 'wb').write(imgContent)#写入
            #重定尺寸
            if imgSize:
                resizeImg(imgPath,imgSize,imgPath)
            newFinish = len(os.listdir(dirname))
            print('key:%s goal:%d finish:%d'%(keyWord,imgNumber,newFinish))
            #图片数达标,退出
            if newFinish >= imgNumber:
                return None
            #重复下载图片达到100次，说明已经下载完所有图片，退出
            if newFinish == oldFinish:
                same+=1
            if same >= 20:
                return

if __name__ == "__main__":
    for keyWord in ['机器人']:
        download(keyWord,imgNumber,imgSize)