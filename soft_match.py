import pandas as pd
from ShortestPath import ShortestPath


if __name__ == '__main__':
    labeled_rule_csv = pd.read_csv(
        "/Users/xuhaoshuai/GitHub/HumanIE/test/test_data/labeled_rule_demo.csv",
        index_col='id'
    )
    labeled_ruleset = labeled_rule_csv['rule']
    sent_csv = pd.read_csv(
        "/Users/xuhaoshuai/GitHub/HumanIE/test/test_data/to_match_sent_demo.csv",
        header=None
    )
    for r in labeled_ruleset:
        for s in sent_csv[0]:
            test = ShortestPath(r, s)
            print(test.get_score())
