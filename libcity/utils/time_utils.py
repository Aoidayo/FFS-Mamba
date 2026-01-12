import time
import datetime


def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    # 10-22-2025_16-17-53
    cur = cur.strftime('%m-%d-%Y_%H-%M-%S')
    return cur

def timestamp_datetime(secs):
    dt = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime(secs))
    return dt


def datetime_timestamp(dt):
    s = time.mktime(time.strptime(dt, '%Y-%m-%dT%H:%M:%SZ'))
    return int(s)
