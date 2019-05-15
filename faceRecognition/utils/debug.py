# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : debug.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-10 16:24:49
"""
import datetime
import logging
import sys
import time

'''
    url:https://cuiqingcai.com/6080.html
    level级别：debug、info、warning、error以及critical
'''
# logging.basicConfig(level=logging.DEBUG,
#                     filename='output.log',
#                     datefmt='%Y/%m/%d %H:%M:%S',
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def RUN_TIME(deta_time):
    '''
    返回毫秒,deta_time.seconds获得秒数=1000ms，deta_time.microseconds获得微妙数=1/1000ms
    :param deta_time: ms
    :return:
    '''
    time_ = deta_time.seconds * 1000 + deta_time.microseconds / 1000.0
    return time_


def TIME():
    return datetime.datetime.now()


if __name__ == '__main__':
    T0 = TIME()
    # do something
    time.sleep(5)
    T1 = TIME()
    print("rum time:{}ms".format(RUN_TIME(T1 - T0)))

    logger.info('This is a log info')
    logger.debug('Debugging')
    logger.warning('Warning exists')
    logger.error('Finish')
