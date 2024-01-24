import warnings
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats
import dabest

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
warnings.filterwarnings("ignore")

RESULTS_PATH = os.path.join(os.getcwd(), "results", "300_epochs")

paradigm_folders = ["MILR", "P300", "SSVEP"]
evaluations = ["WithinSessionEvaluation", "CrossSessionEvaluation", "CrossSubjectEvaluation"]
evaluation_codes = {
    "WithinSessionEvaluation": "WSession",
    "CrossSessionEvaluation": "XSession",
    "CrossSubjectEvaluation": "XSubj"
}


def filter_df_by_evaluation_and_paradigm_pipeline(df, paradigm, evaluation, pipeline):
    df_filtered = df[(df["paradigm"] == paradigm) & (df["evaluation"] == evaluation) & (df["pipeline"] == pipeline)]
    df_filtered["pipeline"] = df_filtered['pipeline'].str.replace(pipeline, f"{pipeline}_{paradigm}_{evaluation}")
    return df_filtered


def remove_pipelines(full_results_df):
    full_results_df = full_results_df[~full_results_df.pipeline.str.contains("CSP")]
    full_results_df = full_results_df[~full_results_df.pipeline.str.contains("ssvep")]
    full_results_df = full_results_df[~full_results_df.pipeline.str.contains("LogMDM")]
    full_results_df = full_results_df[~full_results_df.pipeline.str.contains("ERPCov")]
    return full_results_df


def rename_pipelines(full_results_df):
    full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('EEGNet_', '')  # change here
    full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('_fb', '')  # change here
    full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('LogReg', 'LR')
    full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('SVC', 'SVM')
    full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('Tgsp', 'TGSP')
    full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('default', 'EEGNet')
    return full_results_df


def get_idx_tuple(full_results_df):
    unique_comparison_pipelines = [pipeline for pipeline in full_results_df["pipeline"].unique() if
                                   "EEGNet" not in pipeline]

    db_idx_tuple_pipeline_wise = ("EEGNet",)
    if "shallow" in unique_comparison_pipelines:
        db_idx_tuple_pipeline_wise = db_idx_tuple_pipeline_wise + ("shallow",)
        unique_comparison_pipelines.remove("shallow")
    if "deep" in unique_comparison_pipelines:
        db_idx_tuple_pipeline_wise = db_idx_tuple_pipeline_wise + ("deep",)
        unique_comparison_pipelines.remove("deep")
    for unique_riemann_pipeline in unique_comparison_pipelines:
        db_idx_tuple_pipeline_wise = db_idx_tuple_pipeline_wise + (unique_riemann_pipeline,)
    return db_idx_tuple_pipeline_wise


def permutation_testing(results_path, paradigm_folder, evaluation_method, evaluation_code, sep=";"):
    def statistic(_x, _y, axis):
        return np.mean(_x, axis=axis) - np.mean(_y, axis=axis)

    #  The exact p value is obtained by measuring how many values of the
    #  null distribution are equal or higher than the correctly labelled accuracy.
    x = []
    y = []
    full_results_df = pd.DataFrame({})
    current_path = os.path.join(results_path, paradigm_folder)
    for filename in os.listdir(current_path):
        if evaluation_method in filename:
            current_file_path = os.path.join(current_path, filename)
            results = pd.read_csv(current_file_path, sep=sep)
            results['group'] = results.apply(lambda row: f"EEGNet\n{evaluation_code}" if "default" in row.pipeline else f"Riemann\n{evaluation_code}", axis=1)  # change here
            data_eegnet = results[results['pipeline'].str.contains("default") == True]  # change here
            data_rg = results[results['pipeline'].str.contains("default") == False]  # change here
            full_results_df = pd.concat([full_results_df, results])
            x.extend(data_eegnet["score"].to_list())
            y.extend(data_rg["score"].to_list())
        else:
            continue
    if not x or not y:
        return None, (x, y), pd.DataFrame({})
    res = scipy.stats.permutation_test((x, y), statistic, vectorized=True, n_resamples=99999, alternative='greater')

    full_results_df["paradigm"] = paradigm_folder
    full_results_df["evaluation"] = evaluation_method
    full_results_df['unique_subject'] = full_results_df.apply(lambda row: row.dataset + "_" + str(row.subject), axis=1)
    return res, (x, y), full_results_df


def run_all_permutation_testing():
    results = {}
    for paradigm_folder in paradigm_folders:
        results[paradigm_folder] = {}
        full_results_df = pd.DataFrame({})
        for evaluation in evaluations:
            evaluation_code = evaluation_codes[evaluation]
            res, xy, results_df = permutation_testing(RESULTS_PATH, paradigm_folder, evaluation, evaluation_code)
            full_results_df = pd.concat([full_results_df, results_df])
            if res:
                results[paradigm_folder][evaluation] = {}
                results[paradigm_folder][evaluation]["statistic"] = res.statistic
                results[paradigm_folder][evaluation]["pvalue"] = res.pvalue

        if len(xy) > 1:
            full_results_df = remove_pipelines(full_results_df)
            full_results_df = rename_pipelines(full_results_df)
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace("Cov\+", "")
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('X', 'XCov+')
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('ERP', 'ERPC+')
            full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('_fb', '')

            db_idx_tuple_pipeline_wise = get_idx_tuple(full_results_df)

            for evaluation_code_key, evaluation_code_value in evaluation_codes.items():
                evaluation_result_df = full_results_df[full_results_df['group'].str.contains(evaluation_code_value)]

                if evaluation_result_df.empty:
                    continue
                evaluation_result_df['unique_subject'] = evaluation_result_df.apply(
                    lambda row: row.dataset + "_" + str(row.subject), axis=1)

                db = dabest.load(
                    evaluation_result_df,
                    idx=db_idx_tuple_pipeline_wise,
                    x="pipeline", y="score",
                    resamples=10000,
                )
                for index, row in db.mean_diff.statistical_tests.iterrows():
                    print(
                        f"{paradigm_folder} & {evaluation_code_key.replace('Evaluation', '')} & {row['test']} & {row['control_N']} "
                        f"& {round(row['difference'], 4)} & {round(row['bca_low'], 4)}, {round(row['bca_high'], 4)} "
                        f"& {round(row['pvalue_permutation'], 4)}")

                # Define the color palette for each dataset
                # You can adjust the colors or use other palettes as needed
                num_datasets = len(evaluation_result_df['dataset'].value_counts().index)
                colors = sns.color_palette("tab10", n_colors=num_datasets)

                unique_subjects = evaluation_result_df['unique_subject'].value_counts().index
                patients_per_dataset = \
                np.unique([label.split('_')[0] for label in unique_subjects], return_counts=True)[1][0]

                # Create a list of colors for each patient within each dataset
                all_colors = [sns.light_palette(color, n_colors=int(patients_per_dataset * 2), reverse=True).as_hex()[
                              :patients_per_dataset] for color in colors]

                # Flatten the list of lists into a single list
                all_colors_flat = [color for sublist in all_colors for color in sublist]

                sns.set(font_scale=1.2)
                sns.set_style(style='white')
                db_plt = db.mean_diff.plot(
                    color_col="unique_subject",
                    custom_palette=all_colors_flat,
                    # swarm_desat=0.65,
                )

                # Modify the legend to show only dataset names without patient numbers
                handles = db_plt.legend().legendHandles
                labels = [text.get_text() for text in db_plt.legend().get_texts()]

                # Maintain the order of appearance using a list
                unique_datasets = []
                for label in labels:
                    dataset = label.split('_')[0]
                    if dataset not in unique_datasets:
                        unique_datasets.append(dataset)

                new_handles = []
                for i in np.arange(0, num_datasets * patients_per_dataset, patients_per_dataset):
                    new_handles.append(handles[i])

                db_plt.axes[0].legend(new_handles, unique_datasets, title='Datasets', bbox_to_anchor=(1.05, 0),
                                      loc='center left')
                db_plt.legends = []
                # db_plt.axes[0].get_legend().set_visible(False)  # !!

                plt.tight_layout()
                db_plt.tight_layout()
                db_plt.suptitle(f"{paradigm_folder} - {evaluation_code_key}")
                db_plt.savefig(os.path.join(RESULTS_PATH, f"{paradigm_folder}_{evaluation_code_value}.png"),
                               bbox_inches='tight')
                db_plt.savefig(os.path.join(RESULTS_PATH, f"{paradigm_folder}_{evaluation_code_value}.pdf"),
                               bbox_inches='tight')
                db.mean_diff.statistical_tests.to_csv(
                    os.path.join(RESULTS_PATH, f"{paradigm_folder}_{evaluation_code_value}_statistics.csv"), sep=";")
                db_plt.show()

    return results


def run_all_permutation_testing_combined():
    results = {}
    full_results_df = pd.DataFrame({})
    for paradigm_folder in paradigm_folders:
        print(paradigm_folder)
        results[paradigm_folder] = {}
        for evaluation in evaluations:
            evaluation_code = evaluation_codes[evaluation]
            res, xy, results_df = permutation_testing(RESULTS_PATH, paradigm_folder, evaluation, evaluation_code)
            full_results_df = pd.concat([full_results_df, results_df])
            if res:
                results[paradigm_folder][evaluation] = {}
                results[paradigm_folder][evaluation]["statistic"] = res.statistic
                results[paradigm_folder][evaluation]["pvalue"] = res.pvalue

    if len(xy) > 1:
        full_results_df = remove_pipelines(full_results_df)
        full_results_df = rename_pipelines(full_results_df)
        full_results_df['pipeline'] = full_results_df['pipeline'].str.replace("Cov\+", "")
        full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('X', "")
        full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('ERP', "")
        full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('\+SVM', '')
        full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('\+LDA', '')
        full_results_df['pipeline'] = full_results_df['pipeline'].str.replace('\+LR', '')

        db_idx_tuple_pipeline_wise = get_idx_tuple(full_results_df)

        full_results_df['unique_subject'] = full_results_df.apply(
            lambda row: row.dataset + "_" + str(row.subject), axis=1)

        db = dabest.load(
            full_results_df,
            idx=db_idx_tuple_pipeline_wise,
            x="pipeline", y="score",
            resamples=10000,
        )
        for index, row in db.mean_diff.statistical_tests.iterrows():
            print(
                f"ALL & {row['test']} & {row['control_N']} "
                f"& {round(row['difference'], 4)} & {round(row['bca_low'], 4)}, {round(row['bca_high'], 4)} "
                f"& {round(row['pvalue_permutation'], 4)}")

        sns.set(font_scale=1.2)
        db_plt = db.mean_diff.plot(
            color_col="paradigm",  # "unique_subject",
            custom_palette="tab10",
            raw_marker_size=2.5,
            swarm_desat=0.65,
        )

        db_plt.legends = []
        # db_plt.axes[0].get_legend().set_visible(False)  # !!

        plt.tight_layout()
        db_plt.tight_layout()
        db_plt.suptitle(f"CNN vs. Riemannian")
        db_plt.savefig(os.path.join(RESULTS_PATH, f"ALL.png"), bbox_inches='tight')
        db_plt.savefig(os.path.join(RESULTS_PATH, f"ALL.pdf"), bbox_inches='tight')
        db.mean_diff.statistical_tests.to_csv(
            os.path.join(RESULTS_PATH, f"ALL_statistics.csv"), sep=";")
        db_plt.show()

        for x in ["paradigm", "evaluation"]:
            sns.set(font_scale=1.6)
            sns.set_style(style='white')
            ax = sns.boxplot(
                data=full_results_df,
                x=x,
                y="time",
                hue="pipeline",
                palette='Set2',
                boxprops={'alpha': 0.7},
                # hue_order=boxplot_order
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

            ax.set_xlabel('')
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.savefig(os.path.join(RESULTS_PATH, f"ALL_boxplot_{x}.png"), bbox_inches='tight', dpi=1200)
            plt.show()

    return results


if __name__ == '__main__':
    run_all_permutation_testing()
    run_all_permutation_testing_combined()
