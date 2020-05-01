from imblearn.over_sampling import SMOTE


def execute_smote(x_sv_train, y_sv_train, seed):
    sm = SMOTE(random_state=seed)
    x_res, y_res = sm.fit_resample(X=x_sv_train, y=y_sv_train)

    return x_res, y_res
