import json
import pathlib
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# TODO
def calc_stats(out, path_orig_json, path_scored_tsv):
    data_orig = [json.loads(_) for _ in pathlib.Path(path_orig_json).read_text().splitlines()]
    # data_scored = json.loads(pathlib.Path(path_scored_json).read_text())
    scored_df = pd.read_csv(path_scored_tsv, sep='\t')
    for idx in range(len(data_orig)):
        supported, not_supported, irrelevant = 0, 0, 0
        try:
            for annotation in data_orig[idx]['annotations']:
                supported = sum(_['label'] == 'S' for _ in annotation['human-atomic-facts'])
                not_supported = sum(_['label'] == 'NS' for _ in annotation['human-atomic-facts'])
                irrelevant = sum(_['label'] == 'IR' for _ in annotation['human-atomic-facts'])
            data_orig[idx]['supported'] = supported
            data_orig[idx]['not_supported'] = not_supported
            data_orig[idx]['irrelevant'] = irrelevant
            data_orig[idx]['num_facts'] = supported + not_supported + irrelevant  # len(_['human-atomic-facts']) for _ in data_orig[idx]['annotations']
        except:
            continue
    dd = OrderedDict()
    for _, row in scored_df.iterrows():
        t = row["topic"]
        a = row["atom"]
        s = row["is_supported"]
        if dd.get(t) is None:
            dd[t] = []
        dd[t].append({
            "atom": a,
            "llm_verified_equivalent": s
        })
    # data_scored = [{"mined_facts": v} for k, v in dd.items()]
    #
    # data_orig_clean = [_ for _ in data_orig if _['input'][len('Question: Tell me a bio of '):-1] in set(
    #     _['river_name'] for _ in data_scored)]

    data = []

    for river_orig in data_orig:
        tt = river_orig["input"].lstrip("Question: Tell me a bio of ").strip('.').strip()
        river_scored = dd.get(tt)
        if river_scored is None:
            print(f"Not found: {tt}")
            continue

        supported = 0
        try:
            for mined_fact in river_scored:
                supported += mined_fact['llm_verified_equivalent'] if mined_fact[
                                                                          'llm_verified_equivalent'] is not None else False
            data.append((river_orig['supported'], river_orig['num_facts'], supported, len(river_scored)))
        except KeyError:
            continue

    factscore_precisions = [sup_fs / total_fs for sup_fs, total_fs, sup_you, total_you in data if
                            total_you != 0 and total_fs != 0]
    your_precisions = [sup_you / total_you for sup_fs, total_fs, sup_you, total_you in data if
                       total_you != 0 and total_fs != 0]
    differences = np.array(your_precisions) - np.array(factscore_precisions)  # Your - FactScore

    plt.figure(figsize=(10, 8), dpi=100)

    # Main scatter plot
    plt.scatter(
        factscore_precisions,
        your_precisions,
        c=differences,
        cmap='coolwarm',
        s=100,
        edgecolor='black',
        alpha=0.8,
        vmin=-0.2,
        vmax=0.2  # Adjust based on your data range
    )

    # Add reference lines
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.75, label='y=x (Equal Precision)')
    plt.plot(lims, [lims[0] + 0.05, lims[1] + 0.05], 'r:', label='+5% Threshold')  # Customizable offset

    # Labels and title
    plt.xlabel('FactScore Precision', fontsize=12)
    plt.ylabel('Your Checker Precision', fontsize=12)
    plt.title('Precision Comparison: Your Checker vs. FactScore', fontsize=14)
    plt.grid(alpha=0.2)

    # Colorbar for difference
    cbar = plt.colorbar()
    cbar.set_label('Your Precision - FactScore Precision', rotation=270, labelpad=15)

    # Add marginal histograms
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=plt.gca())
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=plt.gca())

    ax_histx.hist(factscore_precisions, bins=20, color='skyblue', alpha=0.7)
    ax_histy.hist(your_precisions, bins=20, orientation='horizontal', color='salmon', alpha=0.7)

    # Hide tick labels for marginal plots
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    plt.legend()
    plt.tight_layout()
    plt.show()


#     plt.savefig(out + '.png')

# print('ChatGPT')
# calc_stats('ChatGPT', './data/ChatGPT.jsonl', './data/ChatGPT_rated.jsonl')
# print('InstructGPT')
# calc_stats('InstructGPT', './data/InstructGPT.jsonl', './data/InstructGPT_scored.jsonl')
# print('Perplexity')
# calc_stats('Perplexity', './data/PerplexityAI.jsonl', './data/PerplexityAI_scored.jsonl')
