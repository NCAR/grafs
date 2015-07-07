import argparse

import pandas as pd
from pygrafs.libs.util.Config import Config
from pygrafs.libs.model.MLTrainer import MLTrainer
from pygrafs.libs.model.MLForecaster import MLForecaster


def main():
    parser = argparse.ArgumentParser(description="Use machine learning models on processed data.")
    parser.add_argument('config', help="Config file")
    parser.add_argument("-t", "--train", action='store_true', help="Train models.")
    parser.add_argument("-c", "--cross", action='store_true', help="Cross-validate models.")
    parser.add_argument("-s", "--site", action='store_true', help="Evaluate models at site.")
    parser.add_argument("-f", "--fore", action='store_true', help="Create forecast from trained models.")
    args = parser.parse_args()
    required_attributes = ['data_path', 'data_format', 'input_columns', 'output_column',
                           'model_names', 'model_objects']
    config = Config(args.config, required_attributes=required_attributes)
    if args.train:
        train_models(config)
    if args.cross:
        cross_validate_models(config)
    if args.site:
        site_validation(config)
    if args.fore:
        print "Forecasting"
        forecast_models(config)
    return


def train_models(config):
    """
    Train a set of machine learning models on the input data.

    :param config: Config object containing parameters for data and models.
    """
    mlt = MLTrainer(config.data_path,
                    config.data_format,
                    config.input_columns,
                    config.output_column)
    if hasattr(config, 'query'):
        mlt.load_data_files(exp=config.expression, query=config.query)
    else:
        mlt.load_data_files()
    for m, model_name in enumerate(config.model_names):
        print(model_name)
        mlt.train_model(model_name, config.model_objects[m])
    mlt.save_models(config.ml_model_path)
    mlt.show_feature_importance()


def cross_validate_models(config):
    """
    Perform a cross-validation procedure to evaluate model parameter settings.

    :param config: Config object containing model parameter information.
    :return:
    """
    if hasattr(config, 'diff_column'):
        diff_column = config.diff_column
    else:
        diff_column = None
    mlt = MLTrainer(config.data_path,
                    config.data_format,
                    config.input_columns,
                    config.output_column,
                    diff_column=diff_column)
    if hasattr(config, 'query'):
        mlt.load_data_files(query=config.query)
    else:
        mlt.load_data_files()
    all_predictions = {}
    for m, model_name in enumerate(config.model_names):
        all_predictions[model_name] = mlt.cross_validate_model(model_name, config.model_objects[m], config.n_folds)
    pred_frame = pd.DataFrame(all_predictions)
    pred_frame['obs'] = mlt.all_data[config.output_column]
    for col in config.pred_columns:
        pred_frame[col] = mlt.all_data[col]
    pred_frame.to_csv(config.cv_pred_file,
                      columns=config.pred_columns + config.model_names + ['obs'],
                      float_format="%0.3f")
    return


def site_validation(config):
    diff_column = None
    query = None
    expression = ""
    if hasattr(config, "diff_column"):
        diff_column = config.diff_column
    if hasattr(config, "query"):
        query = config.query
    if hasattr(config, "expression"):
        expression = config.expression
    mlt = MLTrainer(config.data_path,
                    config.data_format,
                    config.input_columns,
                    config.output_column,
                    diff_column=diff_column)
    mlt.load_data_files(expression, query)
    predictions = mlt.site_validation(config.model_names, config.model_objects, config.pred_columns, config.split_day)
    predictions.to_csv(config.site_pred_file, float_format="%0.3f", na_rep="nan", index=False)


def forecast_models(config):
    """
    Load data from a forecast file and apply machine learning model to the forecasts.

    :param config: Config object
    :return:
    """
    mlf = MLForecaster(config.data_path,
                       config.data_format,
                       config.input_columns,
                       config.output_column)
    mlf.load_data(exp=config.expression)
    for model_name in config.model_names:
        mlf.load_model(config.ml_model_path + model_name + ".pkl")
    mlf.make_predictions(config.pred_columns, config.pred_path, config.pred_format)
    return

if __name__ == "__main__":
    main()
