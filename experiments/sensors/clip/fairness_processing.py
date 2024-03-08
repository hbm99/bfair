from statistics import mean
from typing import Dict


def get_SP(true_prs: Dict[str, float], est_prs: Dict[str, float], macro: bool = True):
    true_diffs = get_diffs(true_prs)
    est_diffs = get_diffs(est_prs)

    diffs = {}
    for k in true_diffs.keys():
        diffs[k] = abs(true_diffs[k] - est_diffs[k])

    return diffs if not macro else mean(diffs.values())


def get_diffs(prs):
    diffs = {}
    for k_1, v_1 in prs.items():
        for k_2, v_2 in prs.items():
            if k_1 == k_2:
                continue
            k = str({k_1, k_2})
            if k in diffs.keys():
                continue
            diffs[k] = abs(v_1 - v_2)
    return diffs


def get_sense(true_prs, est_prs):
    pass


true_f = (
    0.07333482843686928,
    {
        "black": 0.4719626168224299,
        "east asian": 0.49391727493917276,
        "indian": 0.4838709677419355,
        "latino_hispanic": 0.5117370892018779,
        "middle eastern": 0.44314868804664725,
        "southeast asian": 0.5164835164835165,
        "white": 0.48312611012433393,
    },
)
est_f = (
    0.09960468058191019,
    {
        "black": 0.47146401985111663,
        "east asian": 0.5318627450980392,
        "indian": 0.49127906976744184,
        "latino_hispanic": 0.528169014084507,
        "middle eastern": 0.432258064516129,
        "southeast asian": 0.49666666666666665,
        "white": 0.49557522123893805,
    },
)
print(get_SP(true_f[1], est_f[1]))
