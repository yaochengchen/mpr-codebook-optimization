#!/usr/bin/env bash
# Regenerate every figure, table, in-text number and the analysis checks of the
# paper from the archived results (CPU only, under a minute).
#
#   bash make_figures.sh                                   # results/{main,measured,ablation}
#   MPR_RESULTS=experiments/results_v2 bash make_figures.sh   # after a new campaign
#
# Output: outputs/figs/*.pdf, outputs/tables/*.tex (incl. numbers.tex),
#         outputs/results_summary.md, outputs/analysis_checks.md
set -e
cd "$(dirname "$0")"
mkdir -p outputs
python experiments/make_figures.py --summary outputs/results_summary.md "$@"
python analysis/analyze_results.py > /dev/null && echo "  checks: outputs/analysis_checks.md"
