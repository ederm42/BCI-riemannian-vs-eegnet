import warnings
import os
from datetime import datetime

import matplotlib.pyplot as plt
import mne
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats
import dabest

import moabb
import moabb.analysis
import moabb.analysis.plotting
from moabb.datasets import \
    BNCI2014001, BNCI2014004, Zhou2016, PhysionetMI, MunichMI, Shin2017A, \
    BNCI2014008, BNCI2014009, BNCI2015003, bi2015a, x , bi2012, bi2014a, \
    MAMEM3, MAMEM2, SSVEPExo, Nakanishi2015, Lee2019_SSVEP, Wang2016


from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import LeftRightImagery, MotorImagery, P300, SSVEP, FilterBankSSVEP
import moabb.pipelines

import classification_pipelines

mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")

RESULTS_PATH = os.path.join(os.getcwd(), "results")

FIT_PARAMS_DEFAULT = {
    "kerasclassifier__batch_size": 16,
    "kerasclassifier__epochs": 300,
    "kerasclassifier__verbose": 2
}


PARADIGM_DATA_MILR = {
    "paradigm": LeftRightImagery(resample=128),
    "pipelines": [
        classification_pipelines.get_mi_pipelines,
        classification_pipelines.get_eegnet_pipelines
    ],
    "eegnet_types": [
        "default"  # , "deep", "shallow"
    ],
    "datasets": [
        # [Dataset, chans, samples, nb_classes]
        [BNCI2014001(), 22, 513, 2],
        [BNCI2014004(), 3, 577, 2],
        [PhysionetMI(), 64, 385, 2],
        [MunichMI(), 128, 896, 2],
        [Shin2017A(accept=True), 30, 1281, 2],
    ],
    "subjects": [
        1, 2, 3, 4, 5
    ],
    "evaluation_methods": [
        WithinSessionEvaluation,
        CrossSubjectEvaluation,
        CrossSessionEvaluation,
    ]
}


PARADIGM_DATA_P300 = {
    "paradigm": P300(resample=128),
    "pipelines": [
        classification_pipelines.get_erp_pipelines,
        classification_pipelines.get_eegnet_pipelines
    ],
    "eegnet_types": [
        "default"  # , "deep", "shallow"
    ],
    "datasets": [
        # [Dataset, chans, samples, nb_classes]
        [BNCI2014008(), 8, 128, 2],
        [BNCI2014009(), 16, 103, 2],
        [BNCI2015003(), 8, 103, 2],
        [bi2015a(), 32, 128, 2],
        [bi2014a(), 16, 128, 2]
    ],
    "subjects": [
        1, 2, 3, 4, 5
    ],
    "evaluation_methods": [
        WithinSessionEvaluation,
        CrossSubjectEvaluation,
        CrossSessionEvaluation,
    ]
}

PARADIGM_DATA_SSVEP = {
    "paradigm": SSVEP(resample=128, fmin=5),
    "pipelines": [
        classification_pipelines.get_ssvep_pipelines,
        classification_pipelines.get_eegnet_pipelines
    ],
    "eegnet_types": [
        "ssvep"  #, "default", "deep", "shallow", "ssvep"
    ],
    "datasets": [
        # [Dataset, chans, samples, nb_classes]
        [SSVEPExo(), 8, 256, 4],
        [Nakanishi2015(), 8, 532, 12],
        [MAMEM3(), 14, 385, 5],
    ],
    "subjects": [
        1, 2, 3, 4, 5
    ],
    "evaluation_methods": [
        WithinSessionEvaluation,
        CrossSubjectEvaluation,
        CrossSessionEvaluation,
    ]
}

PARADIGM_DATA_SSVEP_FB = {
    "paradigm": FilterBankSSVEP(resample=128, n_classes=None),
    "pipelines": [
        classification_pipelines.get_ssvep_fb_pipelines,
        classification_pipelines.get_eegnet_pipelines
    ],
    "eegnet_types": [
        "ssvep_fb"  #, "default", "ssvep_fb"
    ],
    "datasets": [
        # [Dataset, chans, samples, nb_classes]
        [SSVEPExo(), 24, 256, 4],
        [Nakanishi2015(), 96, 532, 12],
        [MAMEM3(), 14 * 5, 385, 5],
    ],
    "subjects": [
        1, 2, 3, 4, 5, 6, 7, 8
    ],
    "evaluation_methods": [
        WithinSessionEvaluation,
        CrossSubjectEvaluation,
        CrossSessionEvaluation,
    ]
}

"""
def process_ssvep_old(evaluation_methods=None):
    if evaluation_methods is None:
        evaluation_methods = [WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation]

    DATASETS_SSVEP = [
        # [Dataset, chans, samples, nb_classes]
        [MAMEM3(), 14, 385, 5]
        # [SSVEPExo(), 8, 513, 4]
    ]

    for dataset_ssvep in DATASETS_SSVEP:
        datasets = [dataset_ssvep[0]]
        chans = dataset_ssvep[1]
        samples = dataset_ssvep[2]
        nb_classes = dataset_ssvep[3]

        # paradigm = SSVEP(resample=128, fmin=5, fmax=20, n_classes=nb_classes)
        paradigm = FilterBankSSVEP(filters=None, n_classes=nb_classes)
        pipelines = {}
        pipelines.update(classification_pipelines.get_ssvep_pipelines())
        # pipelines.update(classification_pipelines.get_eegnet_pipelines(chans*3, samples, nb_classes, eegnet_types=["ssvep_fb"]))

        for evaluation_method in evaluation_methods:
            results = evaluate(evaluation_method, paradigm, pipelines, datasets, subjects=[1, 2, 3])
            plot_results(results)
            save_results(results, paradigm, evaluation_method, datasets[0])
"""

def process_paradigm(paradigm_dict):
    if paradigm_dict["evaluation_methods"] is None:
        evaluation_methods = [WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation]
    else:
        evaluation_methods = paradigm_dict["evaluation_methods"]

    paradigm = paradigm_dict["paradigm"]
    print(f"Processing Paradigm {paradigm}")
    for dataset_info in paradigm_dict["datasets"]:
        dataset_list = [dataset_info[0]]
        print(f" - Processing Dataset {dataset_info[0]}")

        # assemble pipelines
        pipelines = {}
        for pipeline_getter_func in paradigm_dict["pipelines"]:
            print(f" - - Getting Pipeline from function {pipeline_getter_func}")
            pipeline_getter_kwargs = {
                "chans": dataset_info[1],
                "samples": dataset_info[2],
                "nb_classes": dataset_info[3],
                "eegnet_types": paradigm_dict["eegnet_types"]
            }
            pipelines.update(pipeline_getter_func(**pipeline_getter_kwargs))

        # evaluate
        for evaluation_method in evaluation_methods:
            print(f" - - Evaluating with {evaluation_method}")
            results = evaluate(evaluation_method, paradigm, pipelines, dataset_list, subjects=paradigm_dict["subjects"])
            # plot_results(results)
            if results is not None:
                save_results(results, paradigm, evaluation_method, dataset_info[0])


def evaluate(evaluation_method, paradigm, pipelines, datasets, subjects=None):
    if subjects is not None:
        for d in datasets:
            d.subject_list = subjects

    # datasets are automatically removed if this is not checked
    for dataset in datasets:
        if not evaluation_method.is_valid(evaluation_method, dataset):
            print(f" - - Evaluaton Method {evaluation_method} skipped for dataset {dataset}")
            return None

    # When `overwrite` is set to False, the results from the previous tutorial are reused and
    # only the new pipelines are evaluated.
    try:
        evaluation = evaluation_method(
            paradigm=paradigm, datasets=datasets, overwrite=True, fit_params=FIT_PARAMS_DEFAULT
        )
        results = evaluation.process(pipelines)
        return results
    except Exception as e:
        print(e)
        return pd.DataFrame(list())


def plot_results(results, plot_title=None, plot_filepath=None):
    results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]
    g = sns.catplot(
        kind="bar",
        x="score",
        y="subj",
        hue="pipeline",
        col="dataset",
        height=12,
        aspect=0.5,
        data=results,
        orient="h",
        palette="viridis",
    )

    if plot_title is not None:
        g.fig.suptitle(plot_title)
        g.fig.subplots_adjust(top=.9)

    if plot_filepath is not None:
        plt.savefig(plot_filepath)
    plt.show()


def save_results(results, paradigm, evaluation_method, dataset):
    paradigm_str = paradigm.__class__.__name__
    evaluation_method_str = evaluation_method.__name__
    dataset_str = dataset.__class__.__name__
    current_datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")

    result_filename = "{0}_{1}_{2}_{3}.csv".format(paradigm_str, evaluation_method_str,
                                                   dataset_str, current_datetime_str)

    results.to_csv(path_or_buf=os.path.join(RESULTS_PATH, result_filename), sep=";")


def plot_results_from_path(path, subdirectories, filename_endswith="merged.csv", sep=";"):
    for subdirectory in subdirectories:
        current_path = os.path.join(path, subdirectory)
        for filename in os.listdir(current_path):
            if filename.endswith(filename_endswith):
                current_file_path = os.path.join(current_path, filename)
                results = pd.read_csv(current_file_path, sep=sep)
                plot_results(results, plot_title=filename, plot_filepath=current_file_path.replace(".csv", ".png"))
            else:
                continue


def download_all_datasets_to_local(paradigms=[]):
    for paradigm in paradigms:
        for dataset in paradigm.datasets:
            data = paradigm.get_data(dataset=dataset)


def permutation_testing(results_path, paradigm_folder, evaluation_method, evaluation_code, sep=";"):
    def statistic(_x, _y, axis):
        return np.mean(_x, axis=axis) - np.mean(_y, axis=axis)

    #  The exact p value is obtained by measuring how many values of the null distribution are equal or higher than the correctly labelled accuracy.

    # x is always eegnet, y is always rg
    x = []
    y = []
    full_results_df = pd.DataFrame({})
    current_path = os.path.join(results_path, paradigm_folder)
    for filename in os.listdir(current_path):
        if evaluation_method in filename:
            current_file_path = os.path.join(current_path, filename)
            results = pd.read_csv(current_file_path, sep=sep)
            results['group'] = results.apply(lambda row: f"EEGNet\n{evaluation_code}" if "EEGNet" in row.pipeline else f"Riemann\n{evaluation_code}", axis=1)
            data_eegnet = results[results['pipeline'].str.contains("EEGNet") == True]
            data_rg = results[results['pipeline'].str.contains("EEGNet") == False]
            full_results_df = pd.concat([full_results_df, results])
            x.extend(data_eegnet["score"].to_list())
            y.extend(data_rg["score"].to_list())
        else:
            continue
    if not x or not y:
        return None, (x, y), pd.DataFrame({})
    res = scipy.stats.permutation_test((x, y), statistic, vectorized=True, n_resamples=99999, alternative='greater')
    #db = dabest.load(
    #    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in {"EEGNet": x, "Riemann": y}.items()])),
    #    idx=("Riemann", "EEGNet"),
    #    resamples=99999
    #)
    #db_plt = db.mean_diff.plot()
    #db_plt.show()
    #perm_test = dabest.PermutationTest(y, x, effect_size="mean_diff", is_paired=False)
    return res, (x, y), full_results_df


def run_all_permutation_testing():
    paragidm_folders = ["michal"]  # ["MILR", "P300", "SSVEP"]
    evaluations = ["WithinSessionEvaluation", "CrossSessionEvaluation", "CrossSubjectEvaluation"]
    evaluation_codes = {
        "WithinSessionEvaluation": "WSession",
        "CrossSessionEvaluation": "XSession",
        "CrossSubjectEvaluation": "XSubj"
    }
    results = {}
    for paragidm_folder in paragidm_folders:
        results[paragidm_folder] = {}
        db_dict = {}
        db_idx_tuple = ()
        full_results_df = pd.DataFrame({})
        for evaluation in evaluations:
            evaluation_code = evaluation_codes[evaluation]
            res, xy, results_df = permutation_testing(os.path.join(RESULTS_PATH, "300_epochs"), paragidm_folder, evaluation, evaluation_code)
            full_results_df = pd.concat([full_results_df, results_df])
            db_dict[f"EEGNet\n{evaluation_code}"] = xy[0]
            db_dict[f"Riemann\n{evaluation_code}"] = xy[1]
            if res:
                db_idx_tuple = ((f"Riemann\n{evaluation_code}", f"EEGNet\n{evaluation_code}"),) + db_idx_tuple
                results[paragidm_folder][evaluation] = {}
                results[paragidm_folder][evaluation]["statistic"] = res.statistic
                results[paragidm_folder][evaluation]["pvalue"] = res.pvalue
        if len(xy) > 1:  # if False:
            db = dabest.load(
                full_results_df,
                idx=db_idx_tuple,
                x="group", y="score",
                resamples=10000,
            )
            db_plt = db.mean_diff.plot(
                color_col="pipeline",
                custom_palette="Accent",
            )
            plt.tight_layout()
            db_plt.tight_layout()
            db_plt.suptitle(paragidm_folder)
            db_plt.savefig(os.path.join(RESULTS_PATH, f"{paragidm_folder}.png"), bbox_inches='tight')
            db_plt.show()

        if len(xy) > 1:
            # new
            # remove pipelines
            full_results_df = full_results_df[~full_results_df.pipeline.str.contains("CSP")]
            #full_results_df = full_results_df[~full_results_df.pipeline.str.contains("ERPCov")]

            # rename pipelines
            full_results_df.loc[full_results_df['pipeline'].str.contains('EEGNet'), 'pipeline'] = 'EEGNet'
            #full_results_df.loc[full_results_df['pipeline'].str.contains('RMDM'), 'pipeline'] = 'RMDM'
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('LogReg', 'LR')
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('Tgsp', 'TGSP')
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace("Cov\+", "")
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('X', 'XC+')
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('ERP', 'ERPC+')

            unique_riemann_pipelines = [pipeline for pipeline in full_results_df["pipeline"].unique() if "EEGNet" not in pipeline]
            db_idx_tuple_pipeline_wise = ("EEGNet", )  # if "SSVEP" in paragidm_folder else ("EEGNet_default", )
            for unique_riemann_pipeline in unique_riemann_pipelines:
                db_idx_tuple_pipeline_wise = db_idx_tuple_pipeline_wise + (unique_riemann_pipeline, )
            for evaluation_code_key, evaluation_code_value in evaluation_codes.items():
                # print(f"{paragidm_folder}, {evaluation_code_key}")
                evaluation_result_df = full_results_df[full_results_df['group'].str.contains(evaluation_code_value)]

                if evaluation_result_df.empty:
                    continue
                evaluation_result_df['unique_subject'] = evaluation_result_df.apply(lambda row: row.dataset + "_" + str(row.subject), axis=1)

                db = dabest.load(
                    evaluation_result_df,
                    idx=db_idx_tuple_pipeline_wise,
                    x="pipeline", y="score",
                    resamples=10000,
                )
                for index, row in db.mean_diff.statistical_tests.iterrows():
                    print(f"{paragidm_folder} & {evaluation_code_key.replace('Evaluation', '')} & {row['test']} & {row['control_N']} "
                          f"& {round(row['difference'], 4)} & {round(row['bca_low'], 4)}, {round(row['bca_high'], 4)} "
                          f"& {round(row['pvalue_permutation'], 4)}")

                db_plt = db.mean_diff.plot(
                    color_col="unique_subject",
                    custom_palette="Accent",
                )
                plt.tight_layout()
                db_plt.tight_layout()
                #db_plt.gca().get_legend().remove()
                db_plt.suptitle(f"{paragidm_folder} - {evaluation_code_key}")
                db_plt.savefig(os.path.join(RESULTS_PATH, f"{paragidm_folder}_{evaluation_code_value}.png"), bbox_inches='tight')
                db.mean_diff.statistical_tests.to_csv(os.path.join(RESULTS_PATH, f"{paragidm_folder}_{evaluation_code_value}_statistics.csv"), sep=";")
                db_plt.show()
                # print("here")

    return results



if __name__ == '__main__':
    process_paradigm(PARADIGM_DATA_MILR)
    process_paradigm(PARADIGM_DATA_P300)
    process_paradigm(PARADIGM_DATA_SSVEP_FB)
    process_paradigm(PARADIGM_DATA_SSVEP)

    #
    # fig, color_dict = moabb.analysis.plotting.score_plot(
    #     pd.read_csv(
    #         os.path.join(
    #             RESULTS_PATH,
    #             "300_epochs",
    #             "MILR",
    #             "LeftRightImagery_WithinSessionEvaluation_BNCI2014004_20220528104351.csv"
    #         ), sep=";")
    # )
    # plt.show()

    run_all_permutation_testing()
    plot_michal()

    plot_results_from_path(RESULTS_PATH, ["300_epochs"], filename_endswith=".csv")
    print("done")

