import re

TRAINING = "TRAINING"
TESTING = "TESTING"

TOP_ENSEMBLE_PREFIX = r"TOP-ENSEMBLE-\d+ @ "
TOP_ENSEMBLE_SUFFIX = r" #\nScore: (?P<score>.+)\nFScore: (?P<fscore>.+)\n"

K_SCORE = "score"
K_FSCORE = "fscore"

EXAMPLE = """# BASE-MODEL-18 @ TESTING #
Score: 0.8013021313187151
FScore: [0.03199619649796524, 0.8013021313187151]
statistical-parity: [0.03199619649796524]
# BASE-MODEL-19 @ TRAINING #
Score: 0.8071005190258285
FScore: [0.22777445888495398, 0.8071005190258285]
statistical-parity: [0.22777445888495398]
# BASE-MODEL-19 @ TESTING #
Score: 0.8035132977089859
FScore: [0.22221975925422008, 0.8035132977089859]
statistical-parity: [0.22221975925422008]
# TOP-ENSEMBLE-0 @ TRAINING #
Score: 0.8133042596971838
FScore: [0.05356651972310705, 0.8133042596971838]
statistical-parity: [0.05356651972310705]
# TOP-ENSEMBLE-0 @ TESTING #
Score: 0.8134021251765862
FScore: [0.04873568208756411, 0.8134021251765862]
statistical-parity: [0.04873568208756411]
# TOP-ENSEMBLE-1 @ TRAINING #
Score: 0.793188169896502
FScore: [0.035060774795612565, 0.793188169896502]
statistical-parity: [0.035060774795612565]
# TOP-ENSEMBLE-1 @ TESTING #
Score: 0.7952828450340889
FScore: [0.03225806944754439, 0.7952828450340889]
statistical-parity: [0.03225806944754439]
# TOP-ENSEMBLE-2 @ TRAINING #
"""


def get_top_ensembles(text, scenario=TESTING):
    ensemble = re.compile(TOP_ENSEMBLE_PREFIX + scenario + TOP_ENSEMBLE_SUFFIX)
    for match in ensemble.finditer(text):
        groups = match.groupdict()
        yield {key: eval(value) for key, value in groups.items()}