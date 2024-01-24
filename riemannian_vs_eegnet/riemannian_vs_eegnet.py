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

from moabb.datasets import \
    BNCI2014001, BNCI2014004, Zhou2016, PhysionetMI, MunichMI, Shin2017A, \
    BNCI2014008, BNCI2014009, BNCI2015003, bi2015a, bi2012, bi2014a, \
    MAMEM3, MAMEM2, SSVEPExo, Nakanishi2015, Lee2019_SSVEP, Wang2016


from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import LeftRightImagery, MotorImagery, P300, SSVEP, FilterBankSSVEP
import moabb.pipelines

import classification_pipelines

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")

RESULTS_PATH = os.path.join(os.getcwd(), "results", "300_epochs")

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
        "default", "deep", "shallow"
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
        "default", "deep", "shallow"
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
        "default", "deep", "shallow"
    ],
    "datasets": [
        # [Dataset, chans, samples, nb_classes]
        [SSVEPExo(), 8, 256, 4],
        [Nakanishi2015(), 8, 532, 12],
        [MAMEM3(), 14, 385, 4],
    ],
    "subjects": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ],
    "evaluation_methods": [
        WithinSessionEvaluation,
        CrossSubjectEvaluation,
        # CrossSessionEvaluation,
    ]
}

PARADIGM_DATA_SSVEP_FB = {
    "paradigm": FilterBankSSVEP(resample=128, n_classes=None),
    "pipelines": [
        classification_pipelines.get_ssvep_fb_pipelines,
        classification_pipelines.get_eegnet_pipelines
    ],
    "eegnet_types": [
        "default_fb", "deep_fb", "shallow_fb"
    ],
    "datasets": [
        # [Dataset, chans, samples, nb_classes]
        [SSVEPExo(), 8 * 3, 256, 4],
        [Nakanishi2015(), 8 * 12, 532, 12],
        [MAMEM3(), 14 * 5, 385, 4],
    ],
    "subjects": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ],
    "evaluation_methods": [
        WithinSessionEvaluation,
        CrossSubjectEvaluation,
        # CrossSessionEvaluation,
    ]
}


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


def save_results(results, paradigm, evaluation_method, dataset):
    paradigm_str = paradigm.__class__.__name__
    evaluation_method_str = evaluation_method.__name__
    dataset_str = dataset.__class__.__name__
    current_datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")

    result_filename = "{0}_{1}_{2}_{3}.csv".format(paradigm_str, evaluation_method_str,
                                                   dataset_str, current_datetime_str)

    results.to_csv(path_or_buf=os.path.join(RESULTS_PATH, result_filename), sep=";")


def download_all_datasets_to_local(paradigms=[]):
    for paradigm in paradigms:
        for dataset in paradigm.datasets:
            data = paradigm.get_data(dataset=dataset)


if __name__ == '__main__':
    process_paradigm(PARADIGM_DATA_MILR)
    process_paradigm(PARADIGM_DATA_P300)
    process_paradigm(PARADIGM_DATA_SSVEP_FB)
