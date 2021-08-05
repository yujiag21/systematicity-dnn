import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


def create_visualizations(args):
    for f in os.listdir(args.results_dir):
        if f.endswith("metric.csv"):

            filename_args = f.split("_")
            exp_name = filename_args[0].capitalize()
            vocab = filename_args[1]
            test_set = filename_args[2].capitalize()
            png_filename = f.split(".")[0] + ".png"

            main_title = "{} Experiments {} Metrics ({})".format(exp_name, test_set, vocab)

            file_result_path = os.path.join(args.results_dir, f)
            file_fig_output_path = os.path.join(args.results_dir, png_filename)

            results_df = pd.read_csv(file_result_path)
            results_df = results_df.sort_values(by='epoch')

            fig, axs = plt.subplots(2, 2, sharex=True)
            fig.suptitle(main_title)
            fig.set_size_inches(15, 15)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.05)

            x_ticks = [x for x in range(results_df['epoch'].max() + 1)]

            axs[0, 0].plot(results_df['epoch'], results_df['eval_loss'])
            axs[0, 0].set_title("evaluation loss")
            axs[0, 0].set_xticks(x_ticks)

            axs[0, 1].plot(results_df['epoch'], results_df['f1'])
            axs[0, 1].set_title("f1")

            axs[1, 0].plot(results_df['epoch'], results_df['precision'])
            axs[1, 0].set_title("precision")
            axs[1, 0].set_xlabel("epoch")

            axs[1, 1].plot(results_df['epoch'], results_df['recall'])
            axs[1, 1].set_title("recall")
            axs[1, 1].set_xlabel("epoch")

            fig.savefig(file_fig_output_path, pad_inches=0)



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--results_dir', default='results',
                            help="Directory where all the results files are stored.")
    args = arg_parser.parse_args()
    create_visualizations(args)





