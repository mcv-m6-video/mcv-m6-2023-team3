import trackeval

default_eval_config = trackeval.Evaluator.get_default_eval_config()

default_eval_config['DISPLAY_LESS_PROGRESS'] = False

default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()

default_dataset_config['GT_FOLDER'] = './code/data_dir/'
default_dataset_config['TRACKERS_FOLDER'] = './code/track'
#default_dataset_config['CLASSES_TO_EVAL']= ['pedestrian', 'cars']
default_dataset_config['BENCHMARK']= "data_dir"
default_dataset_config['DO_PREPROC']= False

default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}


evaluator = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
metrics_list = []
for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
    if metric.get_name() in metrics_config['METRICS']:
        metrics_list.append(metric(metrics_config))
if len(metrics_list) == 0:
    raise Exception('No metrics selected for evaluation')
print(evaluator.evaluate(dataset_list, metrics_list, show_progressbar = True))