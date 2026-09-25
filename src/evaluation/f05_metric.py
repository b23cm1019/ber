from sklearn.metrics import fbeta_score


def evaluate_f05(
        y_true,
        y_pred
):

    return fbeta_score(
        y_true,
        y_pred,
        beta=0.5
    )