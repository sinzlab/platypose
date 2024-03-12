import numpy as np
import scipy.optimize as opt


def gauss2d_angle(xy, amp, x0, y0, sigma_x, theta, sigma_y):
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    out = amp * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0)
                          + c * ((y - y0) ** 2)))
    return out.ravel()


def fit_single_gaussian(single_hm, init_mean):
    # esti_hm.shape (16, 64, 64)
    # p2d.shape (32,)  x0x1x2...y0y1y2
    solution = []

    w_guess = init_mean[0]
    h_guess = init_mean[1]

    max_confi = single_hm.max().item()
    # initial guess:
    guess = [max_confi, w_guess, h_guess, 1, 0, 1]

    x = np.linspace(0, 64-1, 64)
    y = np.linspace(0, 64-1, 64)
    x, y = np.meshgrid(x, y)

    # first input is function to fit, second is array of x and y values (coordinates) and third is heatmap array
    try:
        pred_params, uncert_cov = opt.curve_fit(gauss2d_angle, (x, y), single_hm.flatten(), p0=guess)
    except:
        print("Runtime error in curve fitting")
        pred_params = guess
        data_fitted = gauss2d_angle((x, y), *pred_params).reshape(64, 64)
        rmse_fit = np.sqrt(np.mean((single_hm.numpy() - data_fitted) ** 2))
        print(rmse_fit)

    mean = pred_params[1:3]
    cov_matrix = np.array([[pred_params[3], pred_params[4]], [pred_params[4], pred_params[5]]])
    params = np.array([pred_params[1], pred_params[2], pred_params[3], pred_params[4], pred_params[5]])

    # returns a single list of fit parameters:
    # A_0, mux_0, muy_0, sigmax_0, theta_0, sigmay_0, fiterr_0, A_1, ...
    return params
