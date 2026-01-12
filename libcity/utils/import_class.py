import importlib

def get_executor(config, model, data_feature):
    """
    according the config['executor'] to create the executor

    Args:
        config(ConfigParser): config
        model(AbstractModel): model
        data_feature(dict): data_feature

    Returns:
        AbstractExecutor: the loaded executor
    """
    try:
        return getattr(importlib.import_module('libcity.executor.' + config['line']),
                       config['executor'])(config, model, data_feature)
    except AttributeError:
        raise AttributeError('executor is not found')

def get_executor_no_gat(config, model):
    """
    according the config['executor'] to create the executor

    Args:
        config(ConfigParser): config
        model(AbstractModel): model
        data_feature(dict): data_feature

    Returns:
        AbstractExecutor: the loaded executor
    """
    try:
        return getattr(importlib.import_module('libcity.executor.' + config['line']),
                       config['executor'])(config, model)
    except AttributeError:
        raise AttributeError('executor is not found')



def get_model(config, data_feature):
    """
    according the config['model'] to create the model

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the dataset

    Returns:
        AbstractModel: the loaded model
    """
    try:
        return getattr(importlib.import_module('libcity.model.'+ config['line']),
                       config['model'])(config, data_feature)
    except AttributeError:
        raise AttributeError('model is not found')

def get_model_no_gat(config, road_vocab_size=None):
    """
    according the config['model'] to create the model

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the dataset

    Returns:
        AbstractModel: the loaded model
    """
    try:
        if road_vocab_size is None:
            return getattr(importlib.import_module('libcity.model.'+ config['line']),
                           config['model'])(config)
        else:
            return getattr(importlib.import_module('libcity.model.' + config['line']),
                           config['model'])(config, road_vocab_size)
    except AttributeError:
        raise AttributeError('model is not found')


def get_evaluator(config, data_feature):
    """
    according the config['evaluator'] to create the evaluator

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the dataset

    Returns:
        AbstractEvaluator: the loaded evaluator
    """
    try:
        return getattr(importlib.import_module('libcity.evaluator.' + config['line']),
                       config['evaluator'])(config, data_feature)
    except AttributeError:
        raise AttributeError('evaluator is not found')

def get_evaluator_no_gat(config):
    """
    according the config['evaluator'] to create the evaluator

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the dataset

    Returns:
        AbstractEvaluator: the loaded evaluator
    """
    try:
        return getattr(importlib.import_module('libcity.evaluator.' + config['line']),
                       config['evaluator'])(config)
    except AttributeError:
        raise AttributeError('evaluator is not found')

def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset
    Config:
        dataset_class: 与libcity/data下面的datasetClass同步

    Args:
        config(ConfigParser): config

    Returns:
        AbstractDataset: the loaded dataset
    """
    try:
        print(config['dataset_class'])
        return getattr(importlib.import_module('libcity.dataset.' + config['line']),
                       config['dataset_class'])(config)
    except Exception as e:
        print(e)
        raise AttributeError('dataset_class is not found')

def get_config(config):
    try:
        return getattr(importlib.import_module('libcity.config.' + config['task']),
                       config.get('config_file'))
    except Exception as e:
        print(e)
        raise AttributeError('dataset_class is not found')