import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


def create_visualizations(args):
    for f in os.listdir(args.results_dir):
        if f.endswith(".csv") and f.startswith("pair_"):

            filename_args = f.split("_")
            type_data = filename_args[1]
            exp_name = filename_args[2].capitalize()
            vocab = filename_args[3]
            png_filename = f.split(".")[0] + ".png"

            main_title = "{} Experiments {} Metrics".format(exp_name, vocab)

            file_result_path = os.path.join(args.results_dir, f)
            file_fig_output_path = os.path.join(args.results_dir, png_filename)

            results_df = pd.read_csv(file_result_path)
            results_df = results_df.sort_values(by='epoch')
            column_names = results_df.columns.values.tolist()

            edit_dist_colns_src = list(filter(lambda x: "edit_distance_from_src" in x, column_names))
            edit_dist_colns_rev = list(filter(lambda x: "edit_distance_from_rev" in x, column_names))
            accuracy_colns = list(filter(lambda x: "accuracy" in x, column_names))

            fig, axs = plt.subplots(4, 2, sharex=True)
            fig.suptitle(main_title)
            fig.set_size_inches(20, 20)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.05)

            x_ticks = [x for x in range(results_df['epoch'].max() + 1)]

            axs[0, 0].plot(results_df['epoch'], results_df['eval_loss'])
            axs[0, 0].set_title("evaluation loss")
            axs[0, 0].set_xticks(x_ticks)

            axs[0, 1].plot(results_df['epoch'], results_df['overall_acc'])
            axs[0, 1].set_title("overall_acc")

            axs[1, 1].plot(results_df['epoch'], results_df[edit_dist_colns_src[0]])
            axs[1, 1].set_title(edit_dist_colns_src[0])

            axs[1, 0].plot(results_df['epoch'], results_df[edit_dist_colns_src[1]])
            axs[1, 0].set_title(edit_dist_colns_src[1])

            axs[2, 1].plot(results_df['epoch'], results_df[edit_dist_colns_rev[0]])
            axs[2, 1].set_title(edit_dist_colns_rev[0])

            axs[2, 0].plot(results_df['epoch'], results_df[edit_dist_colns_rev[1]])
            axs[2, 0].set_title(edit_dist_colns_rev[1])

            axs[3, 1].plot(results_df['epoch'], results_df[accuracy_colns[0]])
            axs[3, 1].set_title(accuracy_colns[0])

            axs[3, 0].plot(results_df['epoch'], results_df[accuracy_colns[1]])
            axs[3, 0].set_title(accuracy_colns[1])


            fig.savefig(file_fig_output_path, pad_inches=0)



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--results_dir', default='results',
                            help="Directory where all the results files are stored.")
    args = arg_parser.parse_args()
    create_visualizations(args)





