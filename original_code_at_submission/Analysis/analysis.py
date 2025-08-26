import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_metric(cell, i):
    value = eval(str(cell))[i]
    if value != 'Impossible':
        return float(value)
    else:
        return 0.


def read_sm(cell):
    return read_metric(cell, 1)

def read_cm(cell):
    return read_metric(cell, 2)

def read_em(cell):
    return read_metric(cell, 0)

for fn, title in zip([read_sm, read_cm, read_em], ["SM", "CM", "EM"]):

    df = pd.read_excel("lf_results.xlsx", header = 1)
    headers = list(df.columns)
    lfs = headers[7:]
    df_metric = df[lfs].map(fn)

    best_gpt = df_metric.iloc[47]
    best_lama30 = df_metric.iloc[129]
    best_lama31 = df_metric.iloc[219]
    best_qwen = df_metric.iloc[297]
    models = [best_gpt, best_lama30, best_lama31, best_qwen]
    names = ["ChatGPT-4o mini", "Llama 3.0", "Llama 3.1", "Qwen"]

    # models = [best_gpt, best_qwen]
    # names = ["gpt", "qwen"]

    ordered = sorted(lfs, key=lambda x: sum([best_qwen[x], best_gpt[x], best_lama31[x], best_lama30[x]])/4, reverse=True)
    print(ordered)
    filter = {'Adv_0', 'V_0', 'S_0', 'A_0', 'Anti', 'Oper_1', 'A_1', 'S_1', 'S_2', 'Gener', 'Mult', 'Magn', 'Real_1', 'Mero', 'Hypo', 'Bon', 'Syn', 'Magn*', "Real_1*", "S_1Pred", "Syn_⊂^sex"}
    cats = [c for c in ordered if c in filter]

    width = 0.20
    x = np.arange(len(cats))

    ax = plt.subplot(111)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel('Accuracy ({:s})'.format(title))
    ax.tick_params(axis='x', labelrotation=90)
    # ax.set_title(title)

    for i, (model, name) in enumerate(zip(models, names)):
        ax.bar([t + (i- 1.5) * width for t in x], [model[c] for c in cats], alpha=0.7, width=width, label=name)

    # ax.bar(x, [sum([model[c] for model in models])/len(models) for c in cats])
    # plt.figure(figsize=(15, 10))
    # plt.show()

    # ax = plt.subplot(111)
    # ax.set_xticks(x)
    # ax.set_xticklabels(cats)
    # ax.set_ylabel('accuracy')
    # ax.tick_params(axis='x', labelrotation=90)
    # ax.set_title(title + ' avg')
    y =  [sum([model[c] for model in models])/len(models) for c in cats]
    yerr = [np.std([model[c] for model in models]) for c in cats]
    # ax.bar(x, [sum([model[c] for model in models])/len(models) for c in cats])
    ax.errorbar([t  for t in x], [sum([model[c] for model in models])/len(models) for c in cats], yerr, capsize=3, color='black', fmt='.', linestyle='', label='Mean accuracy')
    ax.grid(axis='y')
    ax.legend(loc='upper right')
    plt.show()


