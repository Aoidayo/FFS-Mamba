from libcity.config import ConfigParser
from libcity.utils import get_executor, get_model, get_evaluator, get_dataset, \
    ensure_dir, set_random_seed, \
    get_logger, get_model_no_gat, get_evaluator_no_gat, get_executor_no_gat


config = ConfigParser(
    task="ffs_downstream",
    model="FFSTTE_AUG",
    dataset="xian",
    config_file="xian_tte", #['chengdu_small_4_gpsview_in_tte_aug_20w']
    saved_model=True,
    train=True,
    other_args=None
)
logger = get_logger(config, is_output_file=True)
logger.info('Begin pretrain-pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(config.get('task'), config.get("model"), config.get("dataset"), config.get("exp_id")))
logger.info("⚙️ Config Here...")
logger.info(config.config)
seed = config.get('seed', 0)
set_random_seed(seed)

dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
# roadgat_data = dataset.get_roadgat_data()
model_cache_file = config.get('model_cache_file', './libcity/cache/{}/{}/model_cache/{}_{}_{}.pt'.format(
    config['line'], config['exp_id'], config['exp_id'], config['model'], config['dataset']))
model = get_model_no_gat(config)
executor = get_executor_no_gat(config, model)

initial_ckpt = config.get("initial_ckpt", None)
pretrain_path = config.get("pretrain_path", None)
if config['train']:
    executor.train(train_data, valid_data, test_data)
    if config['saved_model']:
        executor.save_model(model_cache_file)
    executor.load_model(config.get("initial_ckpt"))
    # executor.load_model_with_epoch(17)
    # executor.evaluate(test_data)
else:
    # assert os.path.exists(model_cache_file) or initial_ckpt is not None or pretrain_path is not None
    # if initial_ckpt is None and pretrain_path is None:
    #     executor.load_model_state(model_cache_file)
    if initial_ckpt is not None:
        executor.load_model_with_tar(config.get("initial_ckpt"))
    executor.evaluate(test_data)

