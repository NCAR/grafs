import argparse
from glob import glob

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from pygrafs.util.Config import Config
from MLTrainer import MLTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',help="Config file")
    parser.add_argument("--train","-t",action='store_true',help="Train models.")
    parser.add_argument("--cross","-c",action='store_true',help="Cross-validate models.")
    parser.add_argument("--eval","-e",action='store_true',help="Evaluate models on test data.")
    parser.add_argument("--fore","-f",action='store_true',help="Create forecast from trained models.")
    args = parser.parse_args()
    config = Config(args.config)
    if args.train:
        train_models(config)
    if args.cross:
        cross_validate_models(config)
    return


def train_models(config):
    mlt = MLTrainer(config.data_path,
                    config.data_format,
                    config.input_columns,
                    config.output_column)
    if hasattr(config,'query'):
        mlt.load_data_files(config.query)
    else:
        mlt.load_data_files()
    for m, model_name in enumerate(config.model_names):
        mlt.train_model(model_name,config.model_objects[m])
    mlt.show_feature_importances()
    mlt.save_models(config.ml_model_path)
    return


def cross_validate_models(config):
    mlt = MLTrainer(config.data_path,
                    config.data_format,
                    config.input_columns,
                    config.output_column)
    if hasattr(config,'query'):
        mlt.load_data_files(config.query)
    else:
        mlt.load_data_files()
    all_predictions = {}
    for m, model_name in enumerate(config.model_names):
        all_predictions[model_name] = mlt.cross_validate_model(model_name,config.model_objects[m],config.n_folds)
    pred_frame = pd.DataFrame(all_predictions)
    pred_frame['obs'] = mlt.all_data[config.output_column]
    for col in config.pred_columns:
        pred_frame[col] = mlt.all_data[col]
    pred_frame.to_csv(config.cv_pred_file,columns=config.pred_columns + config.model_names + ['obs'],float_format="%0.3f")
    return


if __name__ == "__main__":
    main()
