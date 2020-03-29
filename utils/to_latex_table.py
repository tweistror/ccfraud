def to_latex_table(dataset, results, headers, usv_train, sv_train, sv_train_fraud):
    latex_code = f'\\begin{{table}}\n' \
                 f'\\centering\n' \
                 f'\\caption{{Anomaly detection results (mean $\\pm$ std) for dataset \\textbf{{{dataset}}} with training sizes: ' \
                 f'usv\_train: {usv_train}, sv\_train: {sv_train - sv_train_fraud} benign, {sv_train_fraud} fraud}}\n' \
                 f'\\begin{{tabular}}{{p{{4.5cm}}p{{2.5cm}}p{{2.5cm}}p{{2.5cm}}p{{2.5cm}}}}\\toprule\n' \
                 f'Method & Precision & Recall & F1-score & Accuracy\\\\\\midrule\n'

    for index, entry in enumerate(results):
        if len(entry) == 1:
            if index == 0:
                latex_code = latex_code + f'\\multicolumn{{5}}{{c}}{{{entry[0]}}}\\\\\n'
            else:
                latex_code = latex_code + f'\\midrule\n\\multicolumn{{5}}{{c}}{{{entry[0]}}}\\\\\n'

        else:
            latex_code = latex_code + f'{entry[0]} & {entry[1]} & {entry[2]} & {entry[3]} & {entry[4]}\\\\\n'

    latex_code = latex_code + f'\\bottomrule\n' \
                              f'\\end{{tabular}}\n' \
                              f'\\label{{table:{dataset}{usv_train}{sv_train}{sv_train_fraud}}}\n' \
                              f'\\end{{table}}\n' \

    print(latex_code)
