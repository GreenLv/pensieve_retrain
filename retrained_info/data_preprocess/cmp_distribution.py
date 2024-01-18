import os
import numpy as np
import matplotlib.pyplot as plt


TRAIN_DIR = './train_traces/'
TEST_DIR = './test_traces/'


def data_to_cdf(data):

    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n

    return x, y


def plot_cdf(train_trace_avg_bw, test_trace_avg_bw):

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    fig, ax = plt.subplots(figsize=(4, 3))

    x, y = data_to_cdf(train_trace_avg_bw)
    ax.plot(x, y, label='Training Set', linewidth=3)

    x, y = data_to_cdf(test_trace_avg_bw)
    ax.plot(x, y, label='Test Set', linewidth=3)

    ax.set_xlabel('Average Bandwidth (Mbps)', fontsize=16)
    ax.set_ylabel('CDF of Traces', fontsize=16)

    ax.set_ylim((0, 1))
    ax.set_xscale('log')

    ax.grid()
    ax.legend(fontsize=14)

    fig_path = './cmp_train_and_test_bw.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("save figure to {}".format(fig_path))


def read_data_and_calculate_distribution():
    # read traces info [timstamp, bandwidth (Mbps))] from TRAIN_DIR and TEST_DIR

    train_trace_avg_bw = []
    for train_trace in os.listdir(TRAIN_DIR):
        train_trace_path = os.path.join(TRAIN_DIR, train_trace)
        single_trace_bw = []
        with open(train_trace_path, 'r') as f:
            for line in f:
                timestamp, bw = line.strip().split()
                single_trace_bw.append(float(bw))
        train_trace_avg_bw.append(np.mean(single_trace_bw))

    test_trace_avg_bw = []
    for test_trace in os.listdir(TEST_DIR):
        test_trace_path = os.path.join(TEST_DIR, test_trace)
        single_trace_bw = []
        with open(test_trace_path, 'r') as f:
            for line in f:
                timestamp, bw = line.strip().split()
                single_trace_bw.append(float(bw))
        test_trace_avg_bw.append(np.mean(single_trace_bw))

    # calculate the distribution of bandwidth in training and test set
    print("Training set:")
    print("Avg. bandwidth: {:.2f} Mbps".format(np.mean(train_trace_avg_bw)))
    print("Std. bandwidth: {:.2f} Mbps".format(np.std(train_trace_avg_bw)))
    print("Min. bandwidth: {:.2f} Mbps".format(np.min(train_trace_avg_bw)))
    print("Max. bandwidth: {:.2f} Mbps".format(np.max(train_trace_avg_bw)))
    print("Test set:")
    print("Avg. bandwidth: {:.2f} Mbps".format(np.mean(test_trace_avg_bw)))
    print("Std. bandwidth: {:.2f} Mbps".format(np.std(test_trace_avg_bw)))
    print("Min. bandwidth: {:.2f} Mbps".format(np.min(test_trace_avg_bw)))
    print("Max. bandwidth: {:.2f} Mbps".format(np.max(test_trace_avg_bw)))

    # plot the distribution (CDF) of avg. bandwidth in training and test set in one figure
    plot_cdf(train_trace_avg_bw, test_trace_avg_bw)


if __name__ == '__main__':

    read_data_and_calculate_distribution()