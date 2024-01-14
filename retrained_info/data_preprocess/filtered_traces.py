import inspect
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


SRC_DIR = './raw_traces/'
DST_DIR = './filtered_traces/'

# TRACE_CLASSES = ['norway3g', 'belgium4g', 'lumos5g', 'soliswifi']
TRACE_CLASSES = ['norway3g', 'lumos4g', 'lumos5g', 'soliswifi']
TRACE_LABLES = ['3G', '4G', '5G', 'WiFi']
COLORS = ['C0', 'C1', 'C2', 'C3']
LINESTYLES = ['--', '-.', '-', ':']


def DEBUG(*objects):
    print('[{}]'.format(inspect.stack()[1][3]), end=' ')
    print(*objects)


def mkdir(dir):
    is_exists = os.path.exists(dir)
    if not is_exists:
        os.makedirs(dir)


def cal_bw_statistics(trace_path):

    all_bw = []
    with open(trace_path, 'rb') as f:
        for line in f:
            parse = line.split()
            all_bw.append(float(parse[1]))  # Mbps

    return np.mean(all_bw), np.std(all_bw), min(all_bw), max(all_bw)


def gen_data_summary(data, trace_class_idx, dst_dir):

    trace_class = TRACE_CLASSES[trace_class_idx]
    min_avg_bw = 1000000
    max_avg_bw = 0
    total_count = 0

    summary_txt = os.path.join(dst_dir, '{}_avg_bw_summary.txt'.format(trace_class))
    with open(summary_txt, 'w+') as f:
        # print to screen and file
        print('-' * 50)
        print("Trace class: [{}]".format(trace_class))
        print('Type\tCount\tAvg\tStd\tRange')

        f.write("Trace class: [{}]\n".format(trace_class))
        f.write('Type\tCount\tAvg\tStd\tRange\n')

        for key, bw_list in data.items():
            print('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f} ~ {:.2f}'.
                format(key, len(bw_list), np.mean(bw_list), np.std(bw_list), np.min(bw_list), np.max(bw_list)))
            f.write('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f} ~ {:.2f}\n'.
                format(key, len(bw_list), np.mean(bw_list), np.std(bw_list), np.min(bw_list), np.max(bw_list)))
            
            min_avg_bw = min(min_avg_bw, np.min(bw_list))
            max_avg_bw = max(max_avg_bw, np.max(bw_list))
            total_count += len(bw_list)

        print('Total trace count: {}'.format(total_count))
        print('Avg. BW range: [{:.2f}] ~ [{:.2f}] Mbps'.format(min_avg_bw, max_avg_bw))
        f.write('Total trace count: {}\n'.format(total_count))
        f.write('Avg. BW range: [{:.2f}] ~ [{:.2f}] Mbps\n'.format(min_avg_bw, max_avg_bw))
        DEBUG("write summary to {}".format(summary_txt))
    
    return max_avg_bw


def data_to_cdf(data):

    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n

    return x, y


def plot_cdf(data, max_value, trace_class_idx, dst_dir):

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    fig, ax = plt.subplots(figsize=(4, 3))
    fig_title = '{} Traces'.format(TRACE_LABLES[trace_class_idx])
    ax.set_title(fig_title, fontsize=16)

    # for loc_idx, loc_bw in enumerate(data):
    for key, bw_list in data.items():
        x, y = data_to_cdf(bw_list)
        ax.plot(x, y, label=key, linewidth=3)

    ax.set_xlabel('Average Bandwidth (Mbps)', fontsize=16)
    ax.set_ylabel('CDF of Traces', fontsize=16)

    # min_xlim = 1
    # max_xlim = max_value if max_value > 100 else 100
    # ax.set_xlim((min_xlim, max_xlim))
    ax.set_ylim((0, 1))
    # ax.set_xscale('log')

    ax.grid()
    ax.legend(labelspacing=0, fontsize=12)

    fig_path = os.path.join(dst_dir, '{}_avg_bw_cdf.png'.format(TRACE_CLASSES[trace_class_idx]))
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    DEBUG("save figure to {}".format(fig_path))


def read_trace_and_plot(src_dir=SRC_DIR, dst_dir=DST_DIR):

    mkdir(dst_dir)

    # 4 classes of traces, corresponding to 4 folders named by TRACE_CLASSES
    trace_class_list = os.listdir(src_dir)
    for trace_class in trace_class_list:
        if trace_class.strip() not in TRACE_CLASSES and not os.path.isdir(trace_class):
            continue

        trace_list = os.listdir(os.path.join(src_dir, trace_class))
        trace_class_idx = TRACE_CLASSES.index(trace_class)

        dst_trace_class_dir = os.path.join(dst_dir, trace_class)
        mkdir(dst_trace_class_dir)

        # each class of traces is further divided into several types, depending on the location or mobility
        trace_avg_bw = {}
        for trace in trace_list:
            if trace_class_idx == 0: # 3G
                key = trace.split('_')[1]
                if '.' in key:
                    key = key.split('.')[0]
            elif trace_class_idx == 1: # 4G
                # key = trace.split('_')[1]     # belgium4g
                key = trace.split('_')[2]       # lumos4g
            elif trace_class_idx == 2:  # 5G
                key = trace.split('_')[3]
            elif trace_class_idx == 3:  # WiFi
                key = trace.split('_')[1]
            if key not in trace_avg_bw:
                trace_avg_bw[key] = []

            trace_path = os.path.join(src_dir, trace_class, trace)
            bw_mean, bw_std, bw_min, bw_max = cal_bw_statistics(trace_path)
            if bw_mean <= 1.5:
                # leave out traces with low average bandwidth
                continue
            trace_avg_bw[key].append(bw_mean)

            # copy trace to dst_dir
            # dst_trace_path = os.path.join(dst_trace_class_dir, trace)
            # os.system('cp {} {}'.format(trace_path, dst_trace_path))
        
        # generate trace summary and plot CDF
        max_avg_bw = gen_data_summary(trace_avg_bw, trace_class_idx, dst_dir)

        plot_cdf(trace_avg_bw, max_avg_bw, trace_class_idx, dst_dir)


if __name__ == '__main__':

    read_trace_and_plot()
