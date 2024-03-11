from statistics import mean
from typing import Dict


def get_SP_sense(true_prs: Dict[str, float], est_prs: Dict[str, float]):
    true_prs = {k.lower(): v for k, v in true_prs.items()}

    true_diffs = get_diffs(true_prs)
    est_diffs = get_diffs(est_prs)

    diffs = {}
    for k in true_diffs.keys():
        diffs[k] = abs(abs(true_diffs[k]) - abs(est_diffs[k]))

    senses = {}
    for k in true_diffs.keys():
        senses[k] = 1 if true_diffs[k] * est_diffs[k] > 0 else 0

    return (
        diffs,
        mean(diffs.values()),
        senses,
        mean(senses.values()),
    )


def get_diffs(prs):
    diffs = {}
    for k_1, v_1 in prs.items():
        for k_2, v_2 in prs.items():
            if k_1 == k_2:
                continue
            k = str({k_1, k_2})
            if k in diffs.keys():
                continue
            diffs[k] = v_1 - v_2
    return diffs


true_f = (
    0.07333482843686928,
    {
        "Black": 0.4719626168224299,
        "East Asian": 0.49391727493917276,
        "Indian": 0.4838709677419355,
        "Latino_Hispanic": 0.5117370892018779,
        "Middle Eastern": 0.44314868804664725,
        "Southeast Asian": 0.5164835164835165,
        "White": 0.48312611012433393,
    },
)
est_f = (
    0.11025641025641025,
    {
        "black": 0.4666666666666667,
        "east asian": 0.5256410256410257,
        "indian": 0.49206349206349204,
        "latino_hispanic": 0.5108695652173914,
        "middle eastern": 0.4153846153846154,
        "southeast asian": 0.5038461538461538,
        "white": 0.5013404825737265,
    },
)
sp, sp_mean, sense, sense_mean = get_SP_sense(true_f[1], est_f[1])

print("SP")
print(sp)
print("SP mean")
print(sp_mean)
print("sense")
print(sense)
print("sense mean")
print(sense_mean)
