import logging
import os
import sys
from libcity.utils import get_local_time


"""

Config 参数 / 
    exp_id: 在pipeline中生成的exp_id
    model: model name，与libcity/model 同步，命令行指定(替换config-file)
    dataset: city-name，optional[chengdu,xian,porto}
    log_level: info.debug..

"""


def get_logger(config, is_output_file=True):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    log_dir = './libcity/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}-{}-{}-{}.log'.format(
        config['line'],
        config['exp_id'],
        config['model'], config['dataset'], get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)



    logger = logging.getLogger()
    log_level = config.get('log_level', 'INFO')
    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # [1] file handler
    if is_output_file:
        file_handler = logging.FileHandler(logfilepath)
        file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    if is_output_file:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger
