import numpy as np
import os
import matplotlib.pyplot as plt



######################################################
# Upload.
######################################################
folder = "Sapphire 18-11-2025/"
datasets = []
files = os.listdir(folder)
for file in files:
    datasets.append(np.loadtxt(folder + file, skiprows = 1))


######################################################
# 2theta ranges.
######################################################
x_fulls = []
for dataset in range(len(datasets)):
    x_fulls.append([np.min(datasets[dataset][:, 0]), np.max(datasets[dataset][:, 0])])

x_ranges = [[37, 39], [41, 43], [65, 72], [42, 45], [24, 54], [42, 45]]


######################################################
# Plotting.
######################################################
def onclick(event): 
    print("button = %d, x = %d, y = %d, xdata = %f, ydata = %f" % (event.button, event.x, event.y, event.xdata, event.ydata))

def plot(dataset, file, x_range, cut = False, live = False):
    my_dpi = 192
    tnrfont = {'fontname':'Times New Roman'}
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.fancybox"] = False

    fig, ax = plt.subplots(num = "hist",
                            figsize = (4000/my_dpi, 2000/my_dpi),
                            dpi = my_dpi)
    
    plt.plot(dataset[:, 0], dataset[:, 1],
             color = "black",
             linewidth = 1,
             zorder = 1)
    if cut == True:
        plt.scatter(dataset[:, 0], dataset[:, 1],
                edgecolor = "black",
                facecolor = "white",
                s = 50,
                zorder = 2)
    x_min, x_max = x_range[0], x_range[1]
    if live == True:
        plt.plot(np.linspace(x_min, x_max, 2), np.ones((2)) * np.max(dataset[:, 1]) / 2,
            linestyle = "--",
            color = "black",
            linewidth = 1,
            zorder = 3)
    elif live == False:
        plt.text(0.75, 0.85,
                "GID XRD diffractogram\nof sapphire thin film",
                **tnrfont,
                fontsize = 52,
                horizontalalignment = 'center',
                verticalalignment = 'center',
                transform = ax.transAxes,
                bbox = dict(facecolor = 'white', alpha = 1))
    label_fontsize = 44
    ax.set_xlabel("$2 \\vartheta$ ($deg$)",
                fontsize = label_fontsize,
                **tnrfont)
    ax.set_ylabel("counts",
                fontsize = label_fontsize,
                **tnrfont)
    x_ticks = np.linspace(x_min, x_max, 10)
    x_ticks_names = []
    if cut == True:
        for tick in x_ticks:
            x_ticks_names.append(str(round(tick, 2)))
    elif cut == False:
        for tick in x_ticks:
            x_ticks_names.append(str(int(tick)))
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(x_ticks, x_ticks_names)
    y_min, y_max = np.min(dataset[:, 1]), np.max(dataset[:, 1])
    ax.set_ylim(y_min - y_max/100, y_max + y_max/100)
    ax.tick_params(axis = 'x', labelsize = 34)
    ax.tick_params(axis = 'y', labelsize = 34)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if live == True:
        print("Live session started.")
    plt.tight_layout()
    if live == False:
        if cut == True:
            plt.savefig("plot_" + file + "_cut.pdf",
                        dpi = my_dpi,
                        bbox_inches = "tight")
        elif cut == False:
            plt.savefig("plot_" + file + ".pdf",
                        dpi = my_dpi,
                        bbox_inches = "tight")
    if live == True:
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    plt.close()

def plot_everything(live):
    for dataset in range(len(datasets)):
        plot(datasets[dataset], files[dataset][:-3], x_fulls[dataset])
        plot(datasets[dataset], files[dataset][:-3], x_ranges[dataset], cut = True, live = live)
    ######################################################
    # Il 5° dataset è un figlio di puttana
    # perché ha due picchi,
    # so special treatment.
    ######################################################
    plot(datasets[4], files[4][:-3] + "_sx", [25, 27], cut = True, live = live)
    plot(datasets[4][len(datasets[4][:, 0])//2:, :], files[4][:-3] + "_dx", [52, 54], cut = True, live = live)


######################################################
# Peaks.
######################################################
def compute_peaks():
    peaks = []
    for dataset in range(len(datasets)):
        if dataset == 4:
            peaks.append([np.array(datasets[dataset][np.where(datasets[dataset][:, 1] == np.max(datasets[dataset][:, 1])), 0]).item(),
                        np.array(datasets[dataset][np.where(datasets[dataset][:, 1] == np.max(datasets[dataset][len(datasets[4][:, 0])//2::, 1])), 0]).item()])
        else:
            peaks.append(np.array(datasets[dataset][np.where(datasets[dataset][:, 1] == np.max(datasets[dataset][:, 1])), 0]).item())
    HWMH = [[37.766001, 37.856525],
            [41.730001, 41.776034],
            [68.160245, 68.261584],
            [43.396148, 43.461111],
            [[25.606153, 25.709256], [52.583072, 52.655565]],
            [43.299903, 43.342732]]
    peaks_err = []
    for dataset in range(len(peaks)):
        if dataset == 4:
            if np.abs(HWMH[dataset][0][0] - peaks[dataset][0]) > np.abs(HWMH[dataset][0][1] - peaks[dataset][0]) and np.abs(HWMH[dataset][1][0] - peaks[dataset][1]) > np.abs(HWMH[dataset][1][1] - peaks[dataset][1]):
                peaks_err.append([np.array(np.abs(HWMH[dataset][0][0] - peaks[dataset][0])).item(), np.array(np.abs(HWMH[dataset][1][0] - peaks[dataset][1])).item()])
            elif np.abs(HWMH[dataset][0][0] - peaks[dataset][0]) > np.abs(HWMH[dataset][0][1] - peaks[dataset][0]) and np.abs(HWMH[dataset][1][0] - peaks[dataset][1]) < np.abs(HWMH[dataset][1][1] - peaks[dataset][1]):
                peaks_err.append([np.array(np.abs(HWMH[dataset][0][0] - peaks[dataset][0])).item(), np.array(np.abs(HWMH[dataset][1][1] - peaks[dataset][1])).item()])
            elif np.abs(HWMH[dataset][0][0] - peaks[dataset][0]) < np.abs(HWMH[dataset][0][1] - peaks[dataset][0]) and np.abs(HWMH[dataset][1][0] - peaks[dataset][1]) > np.abs(HWMH[dataset][1][1] - peaks[dataset][1]):
                peaks_err.append([np.array(np.abs(HWMH[dataset][0][1] - peaks[dataset][0])).item(), np.array(np.abs(HWMH[dataset][1][0] - peaks[dataset][1])).item()])
            else:
                peaks_err.append([np.array(np.abs(HWMH[dataset][0][1] - peaks[dataset][0])).item(), np.array(np.abs(HWMH[dataset][1][1] - peaks[dataset][1])).item()])
        else:
            if np.abs(HWMH[dataset][0] - peaks[dataset]) > np.abs(HWMH[dataset][1] - peaks[dataset]):
                peaks_err.append(np.array(np.abs(HWMH[dataset][0] - peaks[dataset])).item())
            else:
                peaks_err.append(np.array(np.abs(HWMH[dataset][1] - peaks[dataset])).item())
    for dataset in range(len(datasets)):
        if dataset == 4:
            print("For sample " + files[dataset][:-3] + ", peaks are ({} +- {}) [deg] and ({} +- {}) [deg].".format(round(peaks[dataset][0], 2), round(peaks_err[dataset][0], 2), round(peaks[dataset][1], 2), round(peaks_err[dataset][1], 2)))
        else:
            print("For sample " + files[dataset][:-3] + ", peak is ({} +- {}) [deg].".format(round(peaks[dataset], 2), round(peaks_err[dataset], 2)))
    
    peaks_flat = []
    peaks_err_flat = []
    for sample in range(len(peaks)):
        if sample == 4:
            peaks_flat.append(peaks[sample][0])
            peaks_flat.append(peaks[sample][1])
            peaks_err_flat.append(peaks_err[sample][0])
            peaks_err_flat.append(peaks_err[sample][1])
        else:
            peaks_flat.append(peaks[sample])
            peaks_err_flat.append(peaks_err[sample])
    return np.array(peaks_flat), np.array(peaks_err_flat)


######################################################
# Hexagonal indexes to Miller indexes.
# A, C, M, N, R.
# (h, k, i, l)  ----> (h, k, l)
# h = h, k = k, i = - (h + k), l = l
######################################################
def compute_millers(peaks_flat, peaks_err_flat):
    hexs = np.array([[1, 1, 2, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 2, 3],
        [0, 1, -1, 2]])
    Millers = np.zeros((len(hexs), 3))
    for hex in range(len(hexs)):
        k = 0
        for index in range(len(Millers[hex])):
            if k == 2:
                Millers[hex, index] = hexs[hex, k + 1]
            else:
                Millers[hex, index] = hexs[hex, k]
                k += 1
    print("Hexagonals:\n", hexs)
    print("Millers:\n", Millers)
    ######################################################
    # Hexagonal indexes to Miller indexes.
    # Spatial group: R-3c:H
    # A: [2 -1 0] = 37.762°
    # C: [0 0 6] = 41.670°
    # M: [3 0 0] = 68.180°
    # N: [2 -1 3] = 43.338°
    # R: [1 0 -2] = 25.568°
    # R: [2 0 -4] = 52.534°
    # delta2theta = 0.1°
    ######################################################
    Millers_exp = np.array([37.762, 41.670, 68.180, 43.338, 25.568, 52.534, 43.338])
    Millers_exp_err = 0.1
    Millers_exp_rel = np.max(Millers_exp_err / Millers_exp * 100)
    print("The max value for expected Millers relative error is {}%.".format(round(Millers_exp_rel, 4)))
    peaks_rel = np.max(peaks_err_flat / peaks_flat * 100)
    print("The max value for experimental Millers relative error is {}%.".format(round(peaks_rel, 4)))
    coincidence_checks = 0
    for index in range(len(Millers_exp)):
        if (Millers_exp[index] - Millers_exp_err) - (peaks_flat[index] + peaks_err_flat[index]) < 0 or (peaks_flat[index] - peaks_err_flat[index]) - (Millers_exp[index] + Millers_exp_err) < 0:
            coincidence_checks += 1
    if coincidence_checks == len(Millers_exp):
        print("All discrepancies check out!")
    indeces = np.array(["A", "C", "M", "N", "R", "R", "N"])
    peaks_int_norm = np.array([1, 1, 1, 1, 1, 0.65, 0.8])
    peaks_int = np.array([np.max(datasets[0][:, 1]), np.max(datasets[1][:, 1]), np.max(datasets[2][:, 1]), np.max(datasets[3][:, 1]), np.max(datasets[4][:, 1]), np.max(datasets[4][:, 1]) * 0.65, np.max(datasets[5][:, 1])])

    return indeces, peaks_int_norm, peaks_int


######################################################
# Plotting for report.
######################################################
def plot_allofit(datasets, indeces, peaks, peaks_int, savename = ""):
    my_dpi = 192
    tnrfont = {'fontname':'Times New Roman'}
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.fancybox"] = False
    zorders = np.linspace(1, 6, 6)
    colors = ["orangered", "forestgreen", "lightseagreen", "darkorchid", "chartreuse", "crimson"]
    labels = ["sample I", "sample II", "sample III", "sample IV", "sample V", "sample VI"]

    _, ax = plt.subplots(num = "plotallofit",
                            figsize = (4000/my_dpi, 2000/my_dpi),
                            dpi = my_dpi)
    for dataset in range(len(datasets)):
        plt.plot(datasets[dataset][:, 0], datasets[dataset][:, 1],
                color = colors[dataset],
                linewidth = 1.5,
                label = labels[dataset],
                zorder = zorders[dataset])
    for peak in range(len(peaks))[:-1]:
        plt.annotate(indeces[peak],
            xy = (peaks[peak] - 0.9, peaks_int[peak] + 0.01), xycoords = "data",
            fontsize = 40)
    plt.text(0.65, 0.9,
            "Bragg-Brentano XRD diffractograms\nof single crystal sapphire thin films",
            **tnrfont,
            fontsize = 52,
            horizontalalignment = 'center',
            verticalalignment = 'center',
            transform = ax.transAxes,
            bbox = dict(facecolor = 'white', alpha = 1))
    label_fontsize = 44
    ax.set_xlabel("$2 \\vartheta$ ($deg$)",
                fontsize = label_fontsize,
                **tnrfont)
    if savename == "":
        ax.set_ylabel("counts",
                fontsize = label_fontsize,
                **tnrfont)
    else:
        ax.set_ylabel("normalized intensity (a.u.)",
                fontsize = label_fontsize,
                **tnrfont)
    x_min, x_max = np.min(datasets[0][:, 0]), np.max(datasets[0][:, 0])
    x_ticks = np.linspace(x_min, x_max, 10)
    x_ticks_names = []
    for tick in x_ticks:
        x_ticks_names.append(str(int(tick)))
    ax.set_xlim(6, 80)
    ax.set_xticks(x_ticks, x_ticks_names)
    if savename == "":
        y_min, y_max = np.min(datasets), np.max(datasets)
        ax.set_ylim(y_min - y_max/100, y_max + y_max/100 + 200)
    else:
        y_min, y_max = 0, 1
        ax.set_ylim(y_min - y_max/100, 1.4)
    y_ticks = np.linspace(y_min, y_max, 6)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis = 'x', labelsize = 34)
    ax.tick_params(axis = 'y', labelsize = 34)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(loc = "best",
                    fontsize = 40)
    plt.setp(legend.texts, family = "Times New Roman")
    plt.tight_layout()
    plt.savefig("plot_allofit" + savename + ".pdf",
                dpi = my_dpi,
                bbox_inches = "tight")
    plt.close()







if __name__ == "__main__":
    # plot_everything(False)
    peaks_flat, peaks_err_flat = compute_peaks()
    indeces, peaks_int_norm, peaks_int = compute_millers(peaks_flat, peaks_err_flat)
    plot_allofit(datasets, indeces, peaks_flat, peaks_int)
    for dataset in range(len(datasets)):
        print(np.max(datasets[dataset][:, 1]))
        datasets[dataset][:, 1] = datasets[dataset][:, 1] / np.max(datasets[dataset][:, 1])
    plot_allofit(datasets, indeces, peaks_flat, peaks_int_norm, savename = "_normalized")