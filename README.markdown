```markdown

_This project is about the popularity of news articles with different features_

## Requirements
1. **Python**: Version 3.7 or higher.
   Run this: python --version
2. **Python Libraries**:
   Run this: pip install pandas numpy scikit-learn pydot
3. **Graphviz**:
   Download from [graphviz.org](https://graphviz.org/download/).
   Add graphviz bin folder to PATH (Windows).
   Run this to verify: dot -version
4. **Dataset**:
   Place `OnlineNewsPopularity.csv` in the project directory.
   - Download from [UCI repository](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) if needed.

## Setup and Execution
1. Install libraries: pip install pandas numpy scikit-learn pydot
2. Install Graphviz and add to PATH
3. Make sure `OnlineNewsPopularity.csv` is in project directory
5. Run: python DecisionTree.py


## Notes
- The tree is trained on `test_df`
- Only the text tree and `decision_tree.png` are output; other outputs (e.g., metrics) are suppressed.
- Pruned nodes have `Gini: 0.0000`
```