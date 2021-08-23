import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as pltick
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, Ellipse
from scipy.special import expit as sigmoid
from scipy.optimize import fsolve
from math import ceil


def find_tangent(tang_inputs):
    a, b, q, r, fx, fy, linenum = tang_inputs

    def ellipse_func(input):
        theta, k = input
        return [-a * np.sin(theta) + k * a * np.cos(theta) - k * (fx - q),
                 b * np.cos(theta) + k * b * np.sin(theta) - k * (fy - r)]

    guess = np.asarray([np.pi, 1] if linenum in (1, 2) else [0, 1])

    theta0, k0 = fsolve(ellipse_func, guess)
    x0, y0 = a * np.cos(theta0) + q, b * np.sin(theta0) + r

    return x0, y0


def plot_stereo_rig_sq(beta=0.5, gamma=1.0, delta=0.4, epsilon=-1.0):
    fig, ax = plt.subplots()

    # Configure plot
    plt.title('Stereo rig: rectangular object', fontsize=20)
    ax.set_xlabel(r'$x_i$', fontsize=18), ax.set_ylabel(r'$z_i$', fontsize=18)
    plt.gca().set_aspect('equal', adjustable='box')
    loc1 = pltick.MultipleLocator(base=10000.0)
    ax.xaxis.set_major_locator(loc1), ax.yaxis.set_major_locator(loc1)

    # Compute parameters with dimensions
    D, e = f / delta, epsilon * B / 2
    b = beta * D
    c = b / gamma

    # Compute line parameters
    step1, step2 = round(sigmoid(B / 2 + e - c / 2)), round(sigmoid(-B / 2 + e - c / 2))
    step3, step4 = round(sigmoid(-B / 2 - e - c / 2)), round(sigmoid(B / 2 - e - c / 2))

    # Line parameters
    K1, K3 = (D + b * step1 + f) / ((B - c) / 2 + e), -(D + b * step2 + f) / ((B + c) / 2 - e)
    K2, K4, K5 = -a/2 * K1 - f, -(a/2 + B) * K3 - f, (D + b * step3 + f) / ((B + c) / 2 + e)
    K6, K7 = -a/2 * K5 - f, - (D + b * step4 + f) / ((B - c) / 2 - e)
    K8 = -(a/2 + B) * K7 - f

    # Average disparity points
    xAL, xAR = -1/2 * (K2 / K1 + K6 / K5), -1/2 * (K4 / K3 + K8 / K7)

    # Optical center coords
    x_pts, y_pts = [-f, -f, 0, 0, 0, 0], [a/2, a/2+B, 0, a, B, a+B]
    plt.scatter(y_pts, x_pts)

    # Draw rectangle and baseline
    ax.add_patch(Rectangle((a/2 + B/2 + e - c/2, D), c, b)), ax.axline(xy1=(0, 0), xy2=(1, 0), color='k')

    # FOV limits
    ax.axline(xy1=(a / 2, -f), xy2=(0, 0)), ax.axline(xy1=(a / 2, -f), xy2=(a, 0))
    ax.axline(xy1=(a / 2 + B, -f), xy2=(B, 0)), ax.axline(xy1=(a / 2 + B, -f), xy2=(a + B, 0))

    # Side point projection lines
    ax.axline(xy1=(a / 2, -f), xy2=(a/2 + B/2 + e - c/2, D + b * step1), color='k')  # l1
    ax.axline(xy1=(a / 2, -f), xy2=(a/2 + B/2 + e + c/2, D + b * step3), color='k')  # l3
    ax.axline(xy1=(a / 2 + B, -f), xy2=(a / 2 + B / 2 + e - c / 2, D + b * step2), color='k')  # l2
    ax.axline(xy1=(a / 2 + B, -f), xy2=(a / 2 + B / 2 + e + c / 2, D + b * step4), color='k')  # l4

    # Avg disp projection lines
    ax.axline(xy1=(a / 2, -f), xy2=(xAL, 0), color='r'), ax.axline(xy1=(a / 2 + B, -f), xy2=(xAR, 0), color='r')

    return fig


def plot_stereo_rig_el(beta=0.5, gamma=1.0, delta=0.4, epsilon=-1.0):
    fig, ax = plt.subplots()

    # Configure plot
    plt.title('Stereo rig: elliptic object', fontsize=20)
    ax.set_xlabel(r'$x_i$', fontsize=18), ax.set_ylabel(r'$z_i$', fontsize=18)
    plt.gca().set_aspect('equal', adjustable='box')
    loc1 = pltick.MultipleLocator(base=10000.0)
    ax.xaxis.set_major_locator(loc1), ax.yaxis.set_major_locator(loc1)

    # Compute parameters with dimensions
    D, e = f / delta, epsilon * B / 2
    b = beta * D
    c = b / gamma

    # Compute line params
    x0, y0, fx, q, r = [], [], np.asarray([a/2, a/2 + B, a/2, a/2 + B]), a / 2 + B / 2 + e, D + b / 2

    for k in range(4):
        xx, yy = find_tangent((c / 2, b / 2, q, r, fx[k], -f, k + 1))
        x0.append(xx), y0.append(yy)

    # Anchor points
    xAL = a/2 + f/2 * ((x0[0] - a/2) / (y0[0] + f) + (x0[2] - a/2) / (y0[2] + f))
    xAR = a/2 + B + f/2 * ((x0[1] - a/2 - B) / (y0[1] + f) + (x0[3] - a/2 - B) / (y0[3] + f))

    x_pts, y_pts = [-f, -f, 0, 0, 0, 0], [a/2, a/2+B, 0, a, B, a+B]
    plt.scatter(y_pts, x_pts)

    ax.add_patch(Ellipse((q, r), c, b)), ax.axline(xy1=(0, 0), xy2=(1, 0), color='k')  # Draw ellipse and baseline

    # Field of view boundary lines
    ax.axline(xy1=(a / 2, -f), xy2=(0, 0)), ax.axline(xy1=(a / 2, -f), xy2=(a, 0))
    ax.axline(xy1=(a / 2 + B, -f), xy2=(B, 0)), ax.axline(xy1=(a / 2 + B, -f), xy2=(a + B, 0))

    # Object tangents
    ax.axline(xy1=(a / 2, -f), xy2=(x0[0], y0[0]), color='k')  # l1
    ax.axline(xy1=(a / 2 + B, -f), xy2=(x0[1], y0[1]), color='k')  # l2
    ax.axline(xy1=(a / 2, -f), xy2=(x0[2], y0[2]), color='k')  # l3
    ax.axline(xy1=(a / 2 + B, -f), xy2=(x0[3], y0[3]), color='k')  # l4

    # Avg disparity projection lines
    ax.axline(xy1=(a / 2, -f), xy2=(xAL, 0), color='r'), ax.axline(xy1=(a / 2 + B, -f), xy2=(xAR, 0), color='r')

    return fig


def disparity_sq(input=(0.2, 0.5, 0.1, 0.0)):
    # Initialize
    beta, gamma, delta, epsilon = input

    # Compute parameters with dimensions
    D, e = f / delta, epsilon * B / 2
    b = beta * D
    c = b / gamma

    # Compute line parameters
    step1, step2 = round(sigmoid( B / 2 + e - c / 2)), round(sigmoid(-B / 2 + e - c / 2))
    step3, step4 = round(sigmoid(-B / 2 - e - c / 2)), round(sigmoid( B / 2 - e - c / 2))

    # Line parameters
    K1, K3 = (D + b * step1 + f) / ((B - c) / 2 + e), -(D + b * step2 + f) / ((B + c) / 2 - e)
    K2, K4, K5 = -a/2 * K1 - f, -(a/2 + B) * K3 - f, (D + b * step3 + f) / ((B + c) / 2 + e)
    K6, K7 = -a/2 * K5 - f, - (D + b * step4 + f) / ((B - c) / 2 - e)
    K8 = -(a/2 + B) * K7 - f

    # Average disparity points and depth
    xAL, xAR = -1/2 * (K2 / K1 + K6 / K5), -1/2 * (K4 / K3 + K8 / K7)
    yAD = f * (xAR - xAL) / (xAL - xAR + B)

    return [(yAD - D) / b, (yAD - D) / D]  # block ratio, depth error


def disparity_el(input=(0.2, 0.5, 0.1, 0.0)):
    # Initialize
    beta, gamma, delta, epsilon = input

    # Compute parameters with dimensions
    D, e = f / delta, epsilon * B / 2
    b = beta * D
    c = b / gamma

    # Compute line params
    x0, y0, fx, q, r = [], [], np.asarray([a/2, a/2 + B, a/2, a/2 + B]), a / 2 + B / 2 + e, D + b / 2

    for k in range(4):
        xx, yy = find_tangent((c / 2, b / 2, q, r, fx[k], -f, k + 1))
        x0.append(xx), y0.append(yy)

    # Anchor points and average disparity depth
    xAL = a/2 + f/2 * ((x0[0] - a/2) / (y0[0] + f) + (x0[2] - a/2) / (y0[2] + f))
    xAR = a/2 + B + f/2 * ((x0[1] - a/2 - B) / (y0[1] + f) + (x0[3] - a/2 - B) / (y0[3] + f))
    yAD = f * (xAR - xAL) / (xAL - xAR + B)

    return [(yAD - D) / b, (yAD - D) / D]  # block ratio, depth error


if __name__ == '__main__':
    plt.close('all')

    # Configure fonts
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['legend.title_fontsize'] = 10
    fontP = FontProperties()
    fontP.set_size('small')

    # Initialize constants [px]
    a, f = 1280, 620  # image width
    B = a + 400  # baseline length

    # Initialize controls [1]
    beta = 0.5  # b/D where b is depth of the object block (block depth ratio)
    gamma = 0.5  # b/c where c is the object block width (aspect ratio)
    delta = 0.5  # f/D where D is front face distance from the image plane (distance ratio)
    epsilon = 0  # 2e/B where e is object block excentricity from the midplane

    # Lists of shapes, figures and x-axis annotations
    shape_list, fig_list = ['Rectangle', 'Ellipse'], ['bg', 'db', 'eb', 'dg', 'eg', 'de']
    fig_list_num = ['01', '20', '30', '21', '31', '23']
    fig_list_axes = {'b': ['Object depth ratio', r'$\beta_o$'], 'g': ['Aspect ratio', r'$\gamma_o$'],
                     'd': ['Inverse depth ratio', r'$\delta_o$'], 'e': ['Eccentricity ratio', r'$\varepsilon_o$']}

    # Plotting resolution in terms of points per axis and parameters per plot
    x_res, param_res = 100, 4

    inputs_def, value = [beta, gamma, delta, epsilon], []
    x, params = np.linspace(0.001, 1.0, x_res), np.linspace(0.001, 1.0, param_res)
    for shape_n in range(len(shape_list)):
        for row in range(len(fig_list)):
            inputs = inputs_def
            for param in params:
                for x_pos in x:
                    inputs[int(fig_list_num[row][0])], inputs[int(fig_list_num[row][1])] = x_pos, param
                    if shape_n == 0:
                        centre = disparity_sq(inputs)
                    else:
                        centre = disparity_el(inputs)
                    value.append(centre[1])  # 0...block ratio, 1...depth error
    value = np.asarray(value).reshape((len(shape_list), len(fig_list), param_res, x_res))

    nx_sub = 3  # number of plots in one row

    for shape_n in range(len(shape_list)):
        fig, axs = plt.subplots(ceil(len(fig_list) / nx_sub), nx_sub, sharex=True, figsize=(7.07, 4.9))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.45)
        fig_seq_n = 0
        for row in axs:
            for col in row:
                legend_list = []
                for param_n in range(param_res):
                    col.plot(x, 100 * value[shape_n, fig_seq_n, param_n, :],
                             label='{}'.format(round(params[param_n], 2)))
                if fig_seq_n == 0:
                    axs.flat[fig_seq_n].set_ylabel(shape_list[shape_n] +
                                                   r' depth error $\varepsilon_d$ (%) $\longrightarrow$', fontsize=14,
                                                   labelpad=15, loc='bottom')
                    axs.flat[fig_seq_n].yaxis.set_label_coords(-.15, -.9)
                if fig_list_axes[fig_list[fig_seq_n][0]][0] == 'Eccentricity ratio' and \
                   fig_list_axes[fig_list[fig_seq_n][1]][0] == 'Aspect ratio' and \
                   shape_list[shape_n] == 'Ellipse':
                    axs.flat[fig_seq_n].set_ylim(39.9, 40.1)
                    loc2 = pltick.MultipleLocator(base=.1)
                    axs.flat[fig_seq_n].yaxis.set_major_locator(loc2)
                axs.flat[fig_seq_n].set_xlabel('{}: {} (1) {}'.format(fig_list_axes[fig_list[fig_seq_n][0]][0],
                                                                      fig_list_axes[fig_list[fig_seq_n][0]][1],
                                                                      r'$\longrightarrow$'))
                axs.flat[fig_seq_n].ticklabel_format(useOffset=False, style='plain')
                axs.flat[fig_seq_n].legend(title='{}: {}'.format(fig_list_axes[fig_list[fig_seq_n][1]][0],
                                                                 fig_list_axes[fig_list[fig_seq_n][1]][1]),
                                           loc='upper center', framealpha=1.0, bbox_to_anchor=(.5, 1.3), prop=fontP,
                                           ncol=param_res, labelspacing=0, columnspacing=.5, handlelength=1,
                                           handletextpad=0.3)
                fig_seq_n += 1

    # Stereo rig setup for both shapes
    plot_stereo_rig_el(beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)
    plot_stereo_rig_sq(beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)

    plt.show()
