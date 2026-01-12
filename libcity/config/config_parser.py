import os
import json
import torch
from libcity.utils.import_class import get_config


class ConfigParser(object):

    def __init__(self, task, model, dataset, config_file=None,
                 saved_model=True, train=True, other_args=None):
        self.config = {}
        self.config['config_file'] = config_file
        # 优先级：model的默认dataset、executor、evaluator > _parse_commandline_config > {xxcity}.json > libcity/config
        # 1、命令行参数（包括命令行输入和argument_list中的default值），可能会被后面config_file和load_default_config中的参数给覆盖
        self._parse_commandline_config(task, model, dataset, saved_model, train, other_args)
        # 2、repoRoot/{xxcity}.json 可能会被libcity/config中的默认配置覆盖
        # self._parse_config_file(config_file)
        # 3、libcity/config中的默认配置
        # self._load_libcity_config()

        # 3、libcity/config中的默认配置
        self._load_libcity_all_config()
        self._init_device()

        # init exp_id
        if self.config.get('exp_id') is None:
            # Make a new experiment ID
            import random
            exp_id = int(random.SystemRandom().random() * 1000000)
            self.config['exp_id'] = exp_id

    def _parse_commandline_config(self, task, model, dataset,
                               saved_model=True, train=True, other_args=None):
        '''
        加载 命令行参数（包括命令行输入和argument_list中的default值）
        Args:
            task:
            model:
            dataset:
            saved_model:
            train:
            other_args:

        Returns:

        '''
        if task is None:
            raise ValueError('the parameter task should not be None!')
        if model is None:
            raise ValueError('the parameter model should not be None!')
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        self.config['task'] = task
        self.config['model'] = model
        self.config['dataset'] = dataset
        self.config['saved_model'] = saved_model
        self.config['train'] = train
        if other_args is not None:
            for key in other_args:
                self.config[key] = other_args[key]


    def _load_libcity_all_config(self):
        if self.config['task'] == 'bertlm_pretrain':
            if self.config['model'] == 'BERTContrastiveLM':
                # 预训练的 所有配置
                libcity_all_config = get_config(self)
                for key in libcity_all_config:
                    if key not in self.config:
                        self.config[key] = libcity_all_config[key]
        elif self.config['task'] == 'bertlm_downstream':
            if self.config['model'] == 'LinearTTE':
                # 预训练的 所有配置
                libcity_all_config = get_config(self)
                for key in libcity_all_config:
                    if key not in self.config:
                        self.config[key] = libcity_all_config[key]
        elif self.config['task'] == 'pm_pretrain':
            # 预训练的 所有配置
            libcity_all_config = get_config(self)
            for key in libcity_all_config:
                if key not in self.config:
                    self.config[key] = libcity_all_config[key]
        elif self.config['task'] == 'ffs_pretrain':
            # 预训练的 所有配置
            libcity_all_config = get_config(self)
            for key in libcity_all_config:
                if key not in self.config:
                    self.config[key] = libcity_all_config[key]
        else:
            libcity_all_config = get_config(self)
            for key in libcity_all_config:
                if key not in self.config:
                    self.config[key] = libcity_all_config[key]


    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)
        if use_gpu:
            torch.cuda.set_device(gpu_id)
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if torch.cuda.is_available() and use_gpu else "cpu")

    def get(self, key, default=None):
        # usage： config.get('exp_id', None)
        return self.config.get(key, default)

    def __getitem__(self, key):
        # usage：config['exp_id'] = exp_id
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        # # usage：config['exp_id'] = exp_id
        self.config[key] = value

    def __contains__(self, key):
        # 'exp_id' in config
        return key in self.config

    def __iter__(self):
        # for key in config:
        return self.config.__iter__()

    # def _parse_config_file(self, config_file):
    #     '''
    #     repoRoot/{xxcity}.json 可能会被libcity/config中的默认配置覆盖
    #     Args:
    #         config_file: eg. porto1
    #
    #     Returns:
    #
    #     '''
    #     if config_file is not None:
    #         if os.path.exists('./{}.json'.format(config_file)):
    #             with open('./{}.json'.format(config_file), 'r') as f:
    #                 x = json.load(f)
    #                 for key in x:
    #                     # 不覆盖 _parse_commandline_config 中参数
    #                     if key not in self.config:
    #                         self.config[key] = x[key]
    #         else:
    #             raise FileNotFoundError(
    #                 'Config file {}.json is not found. Please ensure \
    #                 the config file is in the root dir and is a JSON \
    #                 file.'.format(config_file))
    #
    # def _load_libcity_config(self):
    #     '''
    #     libcity/config中的默认配置 参数
    #     '''
    #     if self.config['task'] == 'bertlm_pretrain':
    #         if self.config['model'] == 'BERTContrastiveLM':
    #             self.config['dataset_class'] = 'BertLMContrastiveSplitDataset'
    #             self.config['executor'] = 'ContrastiveSplitMLMExecutor'
    #             self.config['evaluator'] = 'ClassificationEvaluator'
    #         if self.config['model'] == 'LinearClassify':
    #             self.config['dataset_class'] = 'TrajClassifyDataset'
    #             self.config['executor'] = 'TrajClassifyExecutor'
    #             self.config['evaluator'] = 'TwoClassificationEvaluator'
    #             if 'classify_label' in self.config and self.config['classify_label'] == 'usrid':
    #                 self.config['evaluator'] = 'MultiClassificationEvaluator'
    #         if self.config['model'] == 'LinearETA':
    #             self.config['dataset_class'] = 'ETADataset'
    #             self.config['executor'] = 'ETAExecutor'
    #             self.config['evaluator'] = 'RegressionEvaluator'
    #         if self.config['model'] == 'LinearSim':
    #             self.config['dataset_class'] = 'SimilarityDataset'
    #             self.config['executor'] = 'SimilarityExecutor'
    #             self.config['evaluator'] = 'SimilarityEvaluator'
    #
    #
    #     '''
    #     最低优先级参数
    #     '''
    #     default_file_list = []
    #     # model
    #     default_file_list.append('model/{}.json'.format(self.config['model']))
    #     # dataset
    #     default_file_list.append('dataset/{}.json'.format(self.config['dataset_class']))
    #     # executor
    #     default_file_list.append('executor/{}.json'.format(self.config['executor']))
    #     # evaluator
    #     default_file_list.append('evaluator/{}.json'.format(self.config['evaluator']))
    #     for file_name in default_file_list:
    #         with open('./libcity/config/{}'.format(file_name), 'r') as f:
    #             x = json.load(f)
    #             for key in x:
    #                 if key not in self.config:
    #                     self.config[key] = x[key]
