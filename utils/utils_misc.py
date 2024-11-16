import os
import os.path as osp
import configparser
import logging

def get_root_dir():
    return osp.dirname(osp.dirname(osp.abspath(__file__)))

def get_eval_report(labels, preds):
    from sklearn.metrics import classification_report
    return classification_report(labels, preds, output_dict=True)

def print_results(results, epoch, args=None, dataset_split_name="Train"):
    logging.info(f"\n{dataset_split_name} Results for epoch {epoch}:")
    logging.info(f"Accuracy: {results['accuracy']:.4f}")
    logging.info(f"Macro F1: {results['macro avg']['f1-score']:.4f}")
    logging.info(f"Weighted F1: {results['weighted avg']['f1-score']:.4f}")

def set_args_from_config(args, config):
    for section in config.sections():
        for key, value in config[section].items():
            if hasattr(args, key):
                try:
                    # Try to convert to appropriate type
                    if value.lower() == 'true':
                        setattr(args, key, True)
                    elif value.lower() == 'false':
                        setattr(args, key, False)
                    elif '.' in value:
                        setattr(args, key, float(value))
                    else:
                        setattr(args, key, int(value))
                except ValueError:
                    setattr(args, key, value)
    return args

def save_results_to_tsv(results_train, results_eval, experiment_name, args):
    import pandas as pd
    df_train = pd.DataFrame(results_train).T
    df_eval = pd.DataFrame(results_eval).T
    
    output_dir = osp.join(args.outdir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    df_train.to_csv(osp.join(output_dir, f'{experiment_name}_train.tsv'), sep='\t')
    df_eval.to_csv(osp.join(output_dir, f'{experiment_name}_eval.tsv'), sep='\t')
