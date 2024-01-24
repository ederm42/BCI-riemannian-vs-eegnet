import os
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
# mpl.use('PS')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

RESULTS_PATH = os.path.join(os.getcwd(), "results")

def merge_results(path, filename_contains, filename_endswith=".csv", sep=";"):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate through the files in the specified path
    for f in os.listdir(path):
        # Check if the file meets the conditions
        if f.endswith(filename_endswith) and filename_contains in f:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(path, f), sep=sep)

            # Add an "evaluation" column with the filename_contains string
            df["evaluation"] = filename_contains

            # Append the modified DataFrame to the list
            dfs.append(df)

    # Concatenate the list of DataFrames into a single DataFrame
    result_df = pd.concat(dfs, ignore_index=True)

    result_df = result_df[~result_df.pipeline.str.contains("ssvep")]
    result_df.replace({
        "_fb": "",
        "EEGNet_default": "EEGNet",
        "EEGNet_shallow": "ConvNet_shallow",
        "EEGNet_deep": "ConvNet_deep",
        "WithinSession": "within-session",
        "CrossSession": "cross-session",
        "CrossSubject": "cross-subject",
    }, inplace=True, regex=True)
    return result_df


def plot_cnn_comparison(path, paradigm_folders, comparison_type="score", ylim=None):
    evaluations = ["WithinSession", "CrossSession", "CrossSubject"]

    for paradigm_folder in paradigm_folders:
        print(paradigm_folder)

        result_dfs = []
        for evaluation in evaluations:
            try:
                result_df = merge_results(os.path.join(path, paradigm_folder), filename_contains=evaluation)
            except Exception as e:
                print(e)
                result_df = pd.DataFrame({})

            result_dfs.append(result_df)

        results_all = pd.concat(result_dfs, ignore_index=True)

        boxplot_order = ["EEGNet", "ConvNet_shallow", "ConvNet_deep"]
        missing_pipelines = set(results_all["pipeline"].unique()) - set(boxplot_order)
        for missing_pipeline in missing_pipelines:
            boxplot_order.append(missing_pipeline)

        #plt.ylim(0.0, 1.0)
        #plt.yticks(np.arange(0.0, 1.0, 0.1))
        plt.suptitle(f"{paradigm_folder.replace('_CNN', '')}", fontsize=20)

        sns.set(font_scale=1.6)
        sns.set_style(style='white')
        ax = sns.boxplot(
            data=results_all,
            x="evaluation",
            y=comparison_type,
            hue="pipeline",
            palette='tab10',
            boxprops={'alpha': 0.7},
            hue_order=boxplot_order
        )
        if ylim:
            ax.set(ylim=ylim)
            ax.set(yticks=np.arange(0.2, 1.2, 0.2))

        ax.set_xlabel('')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(RESULTS_PATH, "plots", paradigm_folder + "_" + comparison_type + ".jpg"), bbox_inches='tight', dpi=1200)
        plt.savefig(os.path.join(RESULTS_PATH, "plots", paradigm_folder + "_" + comparison_type + ".pdf"), bbox_inches='tight', format='pdf')
        plt.savefig(os.path.join(RESULTS_PATH, "plots", paradigm_folder + "_" + comparison_type + ".eps"), bbox_inches='tight', format='eps')
        plt.show()

        print("done")


def plot_cnn_epochs_comparison(path, epoch_folders, ylim=None):
    evaluations = ["WithinSession", "CrossSession", "CrossSubject"]
    paradigm_folders = ["MILR_CNN", "P300_CNN", "SSVEP_FB_CNN"]

    result_dfs_all = []
    for epoch_folder in epoch_folders:
        result_dfs_epoch = []
        for paradigm_folder in paradigm_folders:
            result_dfs_paradigm = []
            for evaluation in evaluations:
                try:
                    print(f"{epoch_folder} {paradigm_folder} {evaluation}")
                    result_df = merge_results(os.path.join(path, epoch_folder, paradigm_folder), filename_contains=evaluation)
                except Exception as e:
                    print(e, f" - in: {epoch_folder}, {paradigm_folder}, {evaluation}")
                    result_df = pd.DataFrame({})

                result_dfs_paradigm.append(result_df)

            results_all_paradigm = pd.concat(result_dfs_paradigm, ignore_index=True)
            results_all_paradigm['paradigm'] = paradigm_folder
            result_dfs_epoch.append(results_all_paradigm)

        results_all_epoch = pd.concat(result_dfs_epoch, ignore_index=True)
        results_all_epoch['epochs'] = epoch_folder.split("_")[0]
        result_dfs_all.append(results_all_epoch)

    results_all = pd.concat(result_dfs_all, ignore_index=True)


    boxplot_order = ["EEGNet", "ConvNet_shallow", "ConvNet_deep"]
    missing_pipelines = set(results_all["pipeline"].unique()) - set(boxplot_order)
    for missing_pipeline in missing_pipelines:
        boxplot_order.append(missing_pipeline)

    # plt.ylim(0.0, 1.0)
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.suptitle(f"Overall Model Performances", fontsize=20)

    sns.set(font_scale=1.6)
    sns.set_style(style='whitegrid')
    palette = sns.color_palette("tab10", 3, desat=0.75)
    ax = sns.pointplot(
        data=results_all,
        x="epochs",
        y="score",
        hue="pipeline",
        palette=palette,  # 'tab10',
        # boxprops={'saturation': 1},
        linestyles='--',
        dodge=True,
        errorbar="sd",
        # markers='^',
        hue_order=boxplot_order,
        alpha=0.1,
    )

    ax.legend(loc='lower right')
    # ax.legend().set_visible(False)

    if ylim:
        ax.set(yticks=np.arange(0.60, 0.851, 0.05))

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(os.path.join(RESULTS_PATH, epoch_folders[-1], "plots", "epochs_comparison.jpg"), bbox_inches='tight', dpi=1200)
    plt.savefig(os.path.join(RESULTS_PATH, epoch_folders[-1], "plots", "epochs_comparison.pdf"), bbox_inches='tight', format='pdf')
    plt.savefig(os.path.join(RESULTS_PATH, epoch_folders[-1], "plots", "epochs_comparison.eps"), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_cnn_epochs_comparison(path=os.path.join(os.getcwd(), "results"), epoch_folders=["100_epochs", "200_epochs", "300_epochs_final"], ylim=(0.6, 1.1))
