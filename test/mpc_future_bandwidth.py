import numpy as np
import fixed_env2 as env
import load_trace
import matplotlib.pyplot as plt
import itertools


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 7
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
VIDEO_BIT_RATE = [1000,2500,5000,8000,16000,40000]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
REBUF_PENALTY = 40
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_sim_future7_mpc'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

#size_video1 = [3155849, 2641256, 2410258, 2956927, 2593984, 2387850, 2554662, 2964172, 2541127, 2553367, 2641109, 2876576, 2493400, 2872793, 2304791, 2855882, 2887892, 2474922, 2828949, 2510656, 2544304, 2640123, 2737436, 2559198, 2628069, 2626736, 2809466, 2334075, 2775360, 2910246, 2486226, 2721821, 2481034, 3049381, 2589002, 2551718, 2396078, 2869088, 2589488, 2596763, 2462482, 2755802, 2673179, 2846248, 2644274, 2760316, 2310848, 2647013, 1653424]
# size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
# size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
# size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
# size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
# size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
# size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

# BBB with new bitrate levels (up to 40Mbps)
size_video1 = [22434000,20626000,21751000,17875000,12887000,23933000,30165000,23660000,13371000,10627000,13074000,17703000,22364000,19659000,18587000,20750000,19080000,17757000,21908000,28929000,16862000,13737000,17457000,16681000,18906000,16467000,25271000,22098000,19670000,28608000,19168000,17492000,15950000,17945000,18753000,20596000,16754000,17489000,20376000,18977000,24040000,21554000,21265000,23651000,22756000,20727000,24790000,17168000,27341000,18817000,17767000,17773000,23043000,20880000,18891000,16393000,21692000,15690000,20039000,19882000]
size_video2 = [8444000,8970000,9409000,7106000,3707000,10135000,13626000,9621000,5522000,3901000,4471000,6506000,8832000,7963000,4419000,7781000,8011000,6799000,9334000,13938000,5914000,3997000,6289000,5897000,7367000,7123000,10532000,8640000,8270000,13080000,8103000,7103000,6332000,7361000,7697000,8306000,7026000,7274000,8283000,6385000,9779000,7900000,7677000,8543000,9926000,9490000,10142000,5740000,11303000,7109000,6759000,6664000,9735000,8745000,7363000,6167000,9728000,6433000,7601000,7253000]
size_video3 = [6374000,3606000,4593000,3070000,1421000,5140000,7076000,4754000,2798000,1896000,2028000,3130000,4348000,3986000,1666000,3424000,3966000,3208000,4712000,7650000,2766000,1552000,2924000,2187000,3308000,3657000,4915000,4215000,4403000,7712000,4355000,3553000,3136000,3729000,3927000,4171000,3457000,3732000,4222000,2458000,5062000,3775000,3551000,3779000,5383000,5240000,5363000,2625000,6063000,3535000,3176000,3148000,5069000,4586000,3592000,3066000,5528000,3150000,3573000,3164000]
size_video4 = [3629000,2411000,2895000,2259000,961000,2815000,4300000,3076000,2044000,1471000,1471000,2047000,2816000,2937000,860000,1691000,2293000,1825000,2826000,5002000,1809000,941000,1787000,1365000,1977000,2310000,3157000,2663000,2739000,4399000,2678000,2248000,2051000,2441000,2532000,2668000,2125000,2348000,2593000,1568000,3085000,2254000,2233000,2298000,3293000,3463000,3289000,1608000,3688000,2214000,2069000,2054000,3310000,3035000,2251000,1736000,3032000,1927000,2249000,2093000]
size_video5 = [1765000,1231000,1445000,1756000,331000,1043000,1851000,1473000,1120000,859000,803000,1071000,1446000,1344000,377000,736000,1117000,864000,1414000,2595000,955000,447000,961000,565000,868000,1119000,1453000,1286000,1406000,2268000,1400000,1125000,1004000,1246000,1285000,1372000,1002000,1204000,1310000,700000,1593000,1089000,1089000,1043000,1674000,1839000,1751000,759000,1882000,1101000,1022000,1052000,1795000,1658000,1113000,780000,1386000,910000,1083000,1038000]
size_video6 = [677000,527000,565000,669000,144000,433000,721000,577000,472000,359000,411000,455000,585000,496000,163000,299000,455000,350000,595000,1063000,379000,180000,380000,198000,331000,426000,496000,482000,580000,913000,608000,462000,405000,522000,544000,572000,387000,505000,539000,244000,634000,427000,431000,360000,677000,780000,755000,285000,759000,426000,391000,422000,784000,716000,436000,311000,543000,351000,411000,366000]


def get_chunk_size(quality, index):
    if ( index < 0 or index > 48 ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0:size_video6[index]}
    return sizes[quality]


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    video_count = 0

    # make chunk combination options
    for combo in itertools.product(range(A_DIM), repeat=MPC_FUTURE_CHUNK_COUNT):
        CHUNK_COMBO_OPTIONS.append(combo)

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        # log scale reward
        # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

        # reward = log_bit_rate \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        # reward = BITRATE_REWARD[bit_rate] \
        #          - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate] - BITRATE_REWARD[last_bit_rate])


        r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(past_bandwidth_ests) > 0 ):
            curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(past_errors) < 5 ):
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)


        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if ( TOTAL_VIDEO_CHUNKS - last_index < MPC_FUTURE_CHUNK_COUNT ):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = buffer_size
        #start = time.time()
        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int( bit_rate )

            net_env.reset_download_time()  # so that the next future download time starts from now

            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4

                # download_time = (get_chunk_size(chunk_quality, index)/1000000.) / future_bandwidth # this is MB/MB/s --> seconds

                download_time = net_env.get_download_time(get_chunk_size(chunk_quality, index))  # poke env to get future download time

                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                # bitrate_sum += BITRATE_REWARD[chunk_quality]
                # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
            
            reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (smoothness_diffs/1000.)
            # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)


            if ( reward >= max_reward ):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
                if ( best_combo != () ): # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data
        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            print "video count", video_count
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')


if __name__ == '__main__':
    main()

