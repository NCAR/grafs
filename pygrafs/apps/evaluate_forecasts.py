#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')
import argparse
import pandas as pd
from pygrafs.libs.util.Config import Config
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Calculate verification statistics and plot them.")
    parser.add_argument("config", help="Name of config file")
    args = parser.parse_args()
    required_attributes = ['forecast_file', 'score_names', 'model_names', 'score_functions', 'valid_hour_var']
    config = Config(args.config, required_attributes=required_attributes)
    forecast_data = load_forecast_file(config.forecast_file)
    total_scores = calc_scores_by_model(forecast_data, config)
    print(total_scores)
    hour_scores = calc_scores_by_valid_hour(forecast_data, config)
    for score_name, stats in hour_scores.items():
        print(score_name)
        print(stats)
    plot_scores_by_valid_hour(hour_scores, config)
    return


def load_forecast_file(filename):
    """
    Load forecast output from CSV file.

    :param filename: Full path and name of file.
    :return: A pandas DataFrame containing the predictions, observations, and metadata.
    """
    data = pd.read_csv(filename)
    data = data.loc[data['av_dswrf_sfc'] > 0]
    return data


def calc_scores_by_model(forecast_data, config):
    """
    Calculate the score functions specified in the config file.

    :param forecast_data:
    :param config:
    :return: DateFrame with scores organized by model.
    """
    scores = pd.DataFrame(index=config.model_names, columns=config.score_names)
    for m, model in enumerate(config.model_names):
        for s, score in enumerate(config.score_names):
            scores.ix[model, score] = config.score_functions[s](forecast_data[model], forecast_data[config.obs_var])
    return scores


def calc_scores_by_valid_hour(forecast_data, config):
    """
    Calculates scores for all unique valid hours in the prediction file.

    :param forecast_data: Input prediction DataFrame
    :param config: Config object with required parameters
    :return: dict of DataFrames containing each score.
    """
    scores = {}
    valid_hours = sorted(forecast_data[config.valid_hour_var].unique())
    for s, score in enumerate(config.score_names):
        scores[score] = pd.DataFrame(index=config.model_names, columns=valid_hours)
        for model in config.model_names:
            for hour in valid_hours:
                vh_indices = forecast_data[config.valid_hour_var] == hour
                scores[score].ix[model, hour] = config.score_functions[s](forecast_data.loc[vh_indices, model],
                                                                          forecast_data.loc[vh_indices, config.obs_var])
    return scores


def plot_scores_by_valid_hour(scores, config):
    """
    Create line plots of the scores of each model by valid hour.

    :param scores: DataFrame containing the valid hour scores.
    :param config: Config object
    """
    for score_name, score_data in scores.iteritems():
        plt.figure(figsize=(6, 4))
        for model in score_data.index:
            if model in ["av_dswrf_sfc_f","add_diff"]:
                model_label = "NAM"
            else:
                model_label = model
            plt.plot(score_data.columns.values, score_data.ix[model, :], label=model_label)
        plt.xlim(score_data.columns.values.min(),score_data.columns.values.max())
        plt.xticks(score_data.columns.values)
        plt.xlabel("Valid Hour (PST)")
        plt.ylabel(score_name + " ({0})".format(config.units))
        plt.title(score_name + " Hourly Comparison")
        plt.legend(loc=0, fontsize=9)
        plt.savefig(config.figure_path + score_name.lower().replace(" ", "_") + "_valid_hours.png",
                    bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
