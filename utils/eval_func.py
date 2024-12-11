import numpy as np


def find_sbp_dbp(y):
    """
    Find sbp and dbp in a 1D sequence.
    """
    sbp = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    dbp = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
    return sbp, dbp


def calculate_batch_errors(preds, true):
    """
    Calculate MAE and MSE for sbp and dbp across a batch of sequences.
    """
    batch_size, seq_length = preds.shape
    mae_sbp = []
    mse_sbp = []
    mae_dbp = []
    mse_dbp = []
    all_peak_errors = []
    all_trough_errors = []
    pos_id = []
    neg_id = []

    for i in range(batch_size):
        sbp, dbp = find_sbp_dbp(true[i])
        sbp_errors = np.abs(preds[i, sbp] - true[i, sbp])
        dbp_errors = np.abs(preds[i, dbp] - true[i, dbp])
        all_peak_errors.extend(sbp_errors)
        all_trough_errors.extend(dbp_errors)

        mae_p, mse_p = np.mean(np.abs(preds[i, sbp] - true[i, sbp])), np.mean(
            (preds[i, sbp] - true[i, sbp]) ** 2)
        mae_t, mse_t = np.mean(np.abs(preds[i, dbp] - true[i, dbp])), np.mean(
            (preds[i, dbp] - true[i, dbp]) ** 2)
        mae_sbp.append(mae_p)
        mse_sbp.append(mse_p)
        mae_dbp.append(mae_t)
        mse_dbp.append(mse_t)
        if mae_p < 2 and mae_t < 2:
            pos_id.append(i)
        if mae_p > 8 and mae_t > 8:
            neg_id.append(i)


    sd_peaks = np.std(all_peak_errors) if all_peak_errors else np.nan
    sd_troughs = np.std(all_trough_errors) if all_trough_errors else np.nan

    thresholds = [5, 10, 15]
    peak_percentages = [np.mean(np.array(all_peak_errors) <= thresh) for thresh in thresholds]
    trough_percentages = [np.mean(np.array(all_trough_errors) <= thresh) for thresh in thresholds]

    return np.mean(mae_sbp), np.mean(mse_sbp), np.mean(mae_dbp), np.mean(
        mse_dbp), sd_peaks, sd_troughs, peak_percentages, trough_percentages, pos_id, neg_id


def mse_loss(y_true, y_pred):
    error = y_pred - y_true

    squared_error = error ** 2

    mse = np.mean(squared_error)

    return mse
