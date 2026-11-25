# Decision tree contribution

Extract this archive into the root of the team repository. It contains the tree adapter and CLI, the finished grouped development run, and two LaTeX fragments for the report. The model uses the existing repository's preprocessing, experiment runner and portable tree traversal.

Use `report/decision_tree_methodology.tex` in Methodology and `report/decision_tree_results.tex` in Results. The detailed fold settings, per-class metrics and tuning diagnostics are under `results/decision_tree/`. The error CSV contains raw survey excerpts and belongs with the team's development materials.

## Reproduce from the repository root

The verified run used scikit-learn 1.7.2 and seed 67. A new output directory preserves the supplied run.

```sh
python -m pip install -r requirements-development.txt
python -m experiments.run_decision_tree --output results/decision_tree_reproduction
python -m experiments.summarize_results --results results/decision_tree_reproduction
python report/build_decision_tree_section.py --results results/decision_tree_reproduction
```

The optional tuning figure uses R and jsonlite. Running `Rscript report/plot_decision_tree_tuning.R` plots the saved canonical run. For the course submission, incorporate the relevant development source files into the team's code.zip. The archive provided here is the decision-tree contribution for integration.
