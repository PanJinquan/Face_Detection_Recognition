import datetime


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
if __name__=='__main__':
    T0 = datetime.datetime.now()
    # do something
    T1 = datetime.datetime.now()
    print("rum time:{}".format(RUN_TIME(T1 - T0)))
