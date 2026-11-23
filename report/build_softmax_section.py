"""Build shared evaluation and softmax report fragments from measured results.

Run: python report/build_softmax_section.py
Add --pdf with reportlab installed for the two-page model review. These fragments
are source material for the combined report, not a completed three-model report.
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CLASSES = ["Baseball", "Basketball", "Soccer"]


def tex(value):
    return "".join({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}.get(c, c) for c in str(value))


def build_content(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    errors = json.loads((directory / "error_summary.json").read_text(encoding="utf-8"))
    folds = json.loads((directory / "outer_folds.json").read_text(encoding="utf-8"))
    metrics = summary["oof_metrics"]
    methods = (
        "Softmax regression is the linear baseline for the three sport classes. It combines the 73 "
        "numeric, ordinal and fixed binary predictors into class scores, then converts them to probabilities "
        "with softmax. A compact linear model suits the limited respondent count and provides a comparison "
        "with nonlinear tree models. The implementation uses scikit-learn 1.7.2 multinomial logistic "
        "regression with L2 regularization, an intercept, no class reweighting and the L-BFGS optimizer. "
        "The stopping tolerance is 1e-6 and the initial iteration cap is 1,000. A predeclared convergence "
        "warning rule retries the same setting once with a cap of 5,000; unresolved fits cannot win tuning."
    )
    tuning = (
        "Within each outer training partition, grid search compares C = 0.01, 0.1, 1, 10 and 100 "
        "using mean accuracy across its three recorded inner folds. C is inverse regularization strength; "
        "the grid spans strongly to weakly regularized models. Accuracy ties within 1e-12 favor smaller C. "
        "The predeclared expansion rule adds exactly one candidate, 0.001 or 1,000, if the initial winner "
        "is at the corresponding boundary. Outer-fold scores are not used to select C. The five numeric "
        "or ordinal predictors are standardized with training means and population standard deviations; "
        "the 68 binary flags remain 0/1. Preprocessing is fitted separately on every training partition."
    )
    execution = (
        f"This run completed {summary['initial_search_fits']} initial inner-search fits, "
        f"{summary['expansion_search_fits']} expansion fits and {summary['outer_refits']} outer refits "
        f"({summary['fit_attempts']} fit attempts total), with {summary['convergence_retries']} convergence "
        f"retries and {summary['unresolved_inner_failures']} unresolved inner failures. Selected C values "
        "for outer folds 0-4 were " + ", ".join(f"{f['parameters']['C']:g}" for f in folds) +
        (". All initial winners were inside the search bounds, so no expansion was needed. "
         if summary["expansion_search_fits"] == 0 else
         ". Initial boundary winners triggered the predeclared one-step expansion rule. ") +
        "The tuning curves document the regularization trade-off instead of comparing defaults alone."
    )
    performance = (
        f"Across the 870 outer-fold predictions, softmax accuracy is {100 * metrics['accuracy']:.2f}% "
        f"and macro-F1 is {metrics['macro_f1']:.4f}. Mean outer accuracy is "
        f"{100 * summary['mean_outer_accuracy']:.2f}% with sample fold SD "
        f"{100 * summary['sample_std_outer_accuracy']:.2f} percentage points; outer-fold accuracies range "
        f"from {100 * min(summary['outer_fold_accuracies']):.2f}% to {100 * max(summary['outer_fold_accuracies']):.2f}%. "
        f"Mean outer-training accuracy is {100 * summary['mean_training_accuracy']:.2f}%, indicating "
        "a training-validation gap. Aggregate log loss is "
        f"{metrics['log_loss']:.4f}; unlike accuracy, it also reflects the probabilities assigned to the true classes."
    )
    common = errors["common_confusions"][:2]
    confusion_text = "; ".join(f"{p['true_label']} predicted as {p['predicted_label']} ({p['count']} rows)" for p in common)
    weaknesses = (
        f"There are {errors['errors']} errors. The largest directed confusions are {confusion_text}. "
        f"{errors['errors_with_confidence_at_least_0_9']} wrong predictions assign at least 0.9 probability "
        "to the predicted class, showing that confidence alone does not guarantee correctness. "
        "Fixed keyword flags omit unmatched language and a linear classifier cannot represent every "
        "feature interaction; these are structural limitations rather than established causes of individual errors."
    )
    limitation = (
        "These are development cross-validation results for softmax, not untouched-test results or a "
        "final family selection. Preliminary exploration informed keyword definitions before an untouched "
        "local test group could be verified, so earlier feature design may make estimates optimistic. "
        "Final model selection and the single-number hidden-test performance estimate await the tuned "
        "decision-tree and random-forest results."
    )
    export = (
        "For each outer fold, coefficients, intercepts, class/feature order and preprocessing state "
        "were exported to JSON. NumPy inference reproduces all 870 reference predicted labels, with "
        "maximum absolute probability difference "
        f"{max(p['max_absolute_probability_difference'] for p in summary['export_parity']):.2g}. "
        "These five exports are reference models, not a chosen deployment model or an ensemble; "
        "there has been no full-development-data refit."
    )
    return summary, folds, {"methods": methods, "tuning": tuning, "execution": execution,
                            "performance": performance, "weaknesses": weaknesses,
                            "limitation": limitation, "export": export}


def write_fragments(summary, folds, content):
    shared = r"""\subsection{Grouped development evaluation and metrics}
All 290 respondents are development data. We use the recorded five outer
respondent-grouped folds (58 respondents and 174 rows per validation fold),
with three inner grouped folds within each 232-respondent outer training
partition. All three answers from a respondent remain together. Inner
validation accuracy selects hyperparameters; the selected setting is refitted
on the outer training partition and evaluated on its outer validation fold.
Preprocessing is fitted separately within every training partition. The fixed
manual keyword definitions retain the historical exposure limitation described
in Data Exploration. Nested validation separates tuning from outer evaluation;
it does not undo preliminary feature-design exposure.\cite{sklearn_nested_cv}

Accuracy is the primary selection metric because the three classes are balanced.
Macro-F1 gives every sport equal weight and reveals unequal class performance;
per-class precision measures the reliability of predictions for that sport.
We also report recall, F1 and support for each class, and log loss as a
probability-sensitive diagnostic. Precision and recall use zero when their
denominator is zero. Confusion matrices use actual classes as rows and predicted
classes as columns. We report both aggregated outer-fold predictions and
fold-level variability. All families must be tuned under these same rules
before the final family is selected.
"""
    (HERE / "shared_model_methodology.tex").write_text(shared, encoding="utf-8")
    methods = [r"\subsection{Softmax regression}", tex(content["methods"]) + r"\cite{sklearn_logistic}",
               "", tex(content["tuning"]), "", tex(content["execution"])]
    (HERE / "softmax_methodology.tex").write_text("\n".join(methods) + "\n", encoding="utf-8")
    results = [r"\subsection{Softmax regression development results}", tex(content["performance"]), "",
               r"\begin{center}", r"\begin{tabular}{lrrrr}", r"\toprule",
               r"Sport & Precision & Recall & F1 & Support \\", r"\midrule"]
    for label in CLASSES:
        m = summary["oof_metrics"]["per_class"][label]
        results.append(f"{label} & {m['precision']:.4f} & {m['recall']:.4f} & {m['f1']:.4f} & {m['support']} " + r"\\")
    results.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", "", tex(content["weaknesses"]), "",
                    tex(content["limitation"]), "",
                    r"% Optional figure for the combined report; preserve the one-page Results limit.",
                    r"% \includegraphics[width=.55\linewidth]{results/softmax/figures/confusion_matrix.png}"])
    (HERE / "softmax_results.tex").write_text("\n".join(results) + "\n", encoding="utf-8")
    (HERE / "softmax_export_notes.tex").write_text(tex(content["export"]) + "\n", encoding="utf-8")
    references = r"""@misc{sklearn_logistic,
  author = {{scikit-learn developers}},
  title = {{LogisticRegression, scikit-learn 1.7.2 documentation}},
  howpublished = {\url{https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html}},
  note = {Accessed November 23, 2026}
}
@misc{sklearn_nested_cv,
  author = {{scikit-learn developers}},
  title = {{Nested versus non-nested cross-validation}},
  howpublished = {\url{https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html}},
  note = {Accessed November 23, 2026}
}
"""
    (HERE / "model_references.bib").write_text(references, encoding="utf-8")


def render_pdf(directory, summary, folds, content):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    styles = {
        "title": ParagraphStyle("Title", fontName="Times-Bold", fontSize=17, leading=20, spaceAfter=10),
        "heading": ParagraphStyle("Heading", fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=7, spaceAfter=5),
        "body": ParagraphStyle("Body", fontName="Times-Roman", fontSize=12, leading=14.4, spaceAfter=8),
        "small": ParagraphStyle("Small", fontName="Times-Roman", fontSize=10, leading=12, spaceAfter=7),
    }
    p = lambda body, style="body": Paragraph(escape(body), styles[style])
    story = [p("Softmax regression model review", "title"), p("Training and tuning", "heading"),
             p(content["methods"]), p(content["tuning"]), p(content["execution"]),
             Image(str(directory / "figures/inner_validation_accuracy.png"), width=468, height=174),
             p("Mean inner-validation accuracy in each outer split. Stars mark selected C values; outer accuracy does not choose C.", "small"),
             p(content["export"], "small"), PageBreak(), p("Softmax development results", "title"),
             p(content["performance"])]
    rows = [["Sport", "Precision", "Recall", "F1", "Support"]]
    for label in CLASSES:
        m = summary["oof_metrics"]["per_class"][label]
        rows.append([label, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", str(m["support"])])
    table = Table(rows, colWidths=[125, 88, 88, 88, 79])
    table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                               ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"), ("FONTSIZE", (0, 0), (-1, -1), 11),
                               ("LINEBELOW", (0, 0), (-1, 0), .5, colors.black),
                               ("LINEBELOW", (0, -1), (-1, -1), .5, colors.black),
                               ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([table, Spacer(1, 6),
                  Image(str(directory / "figures/confusion_matrix.png"), width=270, height=225),
                  p(content["weaknesses"]), p(content["limitation"], "small")])
    def footer(canvas, doc):
        canvas.setFont("Times-Roman", 9)
        canvas.drawString(72, 756, "CSC311 Group 65374 | Softmax model review")
        canvas.drawRightString(540, 36, str(doc.page))
    output = ROOT / "output/pdf/softmax_model_review.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=letter, leftMargin=72, rightMargin=72,
                      topMargin=72, bottomMargin=72, title="Softmax model review", author="Group 65374").build(
                          story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/softmax")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    summary, folds, content = build_content(args.results)
    write_fragments(summary, folds, content)
    if args.pdf:
        render_pdf(args.results, summary, folds, content)
    print("Built shared methodology and softmax report fragments" + (" and review PDF." if args.pdf else "."))


if __name__ == "__main__":
    main()
