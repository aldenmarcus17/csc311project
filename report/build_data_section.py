"""Build the Data Exploration LaTeX fragment and optional two-page review PDF.

Run from the repository: python report/build_data_section.py
Add --pdf in a Python environment containing reportlab for the standalone review.
The PDF is a section preview; final report layout must be checked after integration.
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def content():
    summary = json.loads((ROOT / "development/eda/summary.json").read_text())
    means = summary["numeric"]
    coverage = summary["keyword_coverage"]
    pages = [[
        ("Scope and distributions",
         "The supplied dataset contains 870 responses from 290 respondents, each answering for Baseball, "
         "Basketball and Soccer. All new descriptive results below use only the recorded exploration "
         "training subset: 232 respondents and 696 responses, with 232 responses per class. The remaining "
         "58 respondents are excluded from this new exploration. These are development partitions, "
         "subject to the historical exposure limitation described below."),
        (None,
         "The inputs comprise one bounded numeric ease score, three ordinal Likert items (strategy, "
         "injury and individual brilliance), three multi-select questions (venue, season and music), "
         "four free-text questions (food, colour, song and feelings), and a free-text payment amount. "
         "The transformation yields 73 predictors: five numeric or ordinal values and 68 binary flags. "
         "Respondent ID and the target label are excluded."),
        (None,
         f"Ease spans 0-10 (mean {means['ease']['mean']:.2f}, sample SD {means['ease']['std']:.2f}). "
         f"The mean Likert scores are {means['strategy']['mean']:.2f}, {means['injury']['mean']:.2f} "
         f"and {means['brilliance']['mean']:.2f}, respectively; agreement is the modal response for all three. "
         "Multi-select indicators can co-occur; summer is the most frequently selected season. "
         "Median answer lengths for food, colour, song and feelings are 16, 5, 17 and 9 characters, "
         "with long right tails. Fixed keyword flags match "
         f"{coverage['food']['percent_all_training_rows']:.1f}% of food, "
         f"{coverage['colour']['percent_all_training_rows']:.1f}% of colour and "
         f"{coverage['feel']['percent_all_training_rows']:.1f}% of feelings rows; song-category flags "
         f"match only {coverage['song']['percent_all_training_rows']:.1f}%. Unmatched language remains a limitation."),
        (None,
         "Baseball has a lower mean ease score (4.06) than Basketball (6.01) or Soccer (6.14). "
         "Indoor venues occur in 96.1% of Basketball answers, versus 24.6% of Baseball and 61.2% of Soccer "
         "answers. The Waka song flag occurs in 57.8% of Soccer answers and none of the other two classes. "
         "These patterns support a linear softmax baseline and trees that capture thresholds and interactions; "
         "a random forest can reduce the variance of an individual tree. They do not establish predictive accuracy."),
        ("Missing values and unusual amounts",
         "There are 15 blank song answers (2.2%), nine blank food answers (1.3%), and seven or eight "
         "blanks (1.0-1.1%) for every other question. Literal N/A and none answers are preserved as content; "
         "this differs from automatic CSV missing-value detection in the proposal. No observed ease score "
         "is outside 0-10, and no nonblank Likert or multi-select response is unrecognized."),
    ], [
        ("Preprocessing shared by the actual candidates",
         "The fixed definitions apply to softmax regression, decision tree and random forest. "
         "Text is normalized for Unicode, whitespace and case. Likert responses map to integers 1-5, "
         "with missing or unrecognized entries set to neutral (3). Venue, season and music use 7, 4 and "
         "10 multi-hot columns; unknown options activate no known option. Food, colour and feelings use "
         "24, 10 and nine fixed regex indicators. Songs use three category flags and a separate named-song flag. "
         "Blank or unmatched text produces zero keyword flags. Historical keyword definitions are preserved "
         "rather than expanding them after inspecting new validation responses."),
        (None,
         "Ease is clipped to 0-10 and missing values use the training median (5 in this exploration). "
         "The payment parser accepts a nonnegative amount, k suffix, explicit refusal (0), simple written "
         "number or unambiguous ascending range (midpoint). It does not average unrelated amounts. "
         f"Here it accepts 597 numeric answers, 30 ranges, 27 refusals and three written numbers. "
         f"The remaining {summary['pay_imputed_rows']} answers ({100 * summary['pay_imputed_rows'] / summary['rows']:.1f}%) "
         "are blank, ambiguous, foreign-currency or unparsed text and use the training median (100). "
         f"Parsed amounts reach 10,000; {summary['outliers']['pay_parsed']['iqr_outlier_count']} exceed the usual "
         "1.5-IQR fences. Valid amounts are retained and transformed with log(1 + amount), reducing skew "
         f"without deleting respondents (transformed mean {means['pay_log']['mean']:.2f}, SD {means['pay_log']['std']:.2f}, "
         "with a spike at zero)."),
        (None,
         "For softmax regression, ease, the three Likert values and log-payment are standardized using "
         "training means and population standard deviations; binary flags remain 0/1. A constant column "
         "uses scale 1. Trees use the unscaled values. Every inner or outer training partition must fit "
         "its own medians and scaling state, then apply that saved state unchanged to validation rows. "
         "The exploration state is not a cross-validation or final deployment state."),
        ("Respondent separation and historical exposure",
         "Preliminary exploration informed the fixed keyword definitions before an untouched local "
         "test group could be verified. All 290 respondents are therefore treated as development data. "
         "The recorded evaluation protocol keeps each respondent's three answers together in five "
         "outer folds, with three inner folds for tuning. Preprocessing is fitted separately within "
         "each training fold and applied unchanged to its validation fold. Earlier feature design "
         "may nevertheless make evaluation estimates optimistic."),
    ]]
    return pages


def tex_escape(value):
    return "".join({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                    "$": r"\$", "{": r"\{", "}": r"\}"}.get(c, c) for c in value)


def render_tex(pages):
    lines = [r"\section{Data Exploration}\label{sec:data-exploration}", ""]
    for page_number, paragraphs in enumerate(pages):
        for heading, body in paragraphs:
            if heading:
                lines.extend([r"\subsection{" + tex_escape(heading) + "}", ""])
            lines.extend([tex_escape(body), ""])
        if page_number == 0:
            lines.extend([
                r"\begin{figure}[ht]", r"\centering",
                r"\includegraphics[width=\linewidth]{development/eda/figures/numeric_by_class.png}",
                r"\caption{Class-conditioned distributions in the 696-row exploration training subset. "
                r"Ease and payment differ by class, but distributions overlap.}",
                r"\label{fig:training-distributions}", r"\end{figure}", ""])
    (HERE / "data_exploration.tex").write_text("\n".join(lines), encoding="utf-8")


def render_pdf(pages):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    directory = ROOT / "output/pdf"
    directory.mkdir(parents=True, exist_ok=True)
    styles = {
        "title": ParagraphStyle("Title", fontName="Times-Bold", fontSize=17, leading=20, spaceAfter=9),
        "heading": ParagraphStyle("Heading", fontName="Times-Bold", fontSize=12, leading=14, spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("Body", fontName="Times-Roman", fontSize=12, leading=14.4, spaceAfter=7, alignment=TA_LEFT),
        "caption": ParagraphStyle("Caption", fontName="Times-Roman", fontSize=9.5, leading=11, spaceAfter=4),
    }
    story = []
    for page_number, paragraphs in enumerate(pages):
        if page_number:
            story.append(PageBreak())
        story.append(Paragraph("Data Exploration" if page_number == 0 else "Data Exploration continued", styles["title"]))
        for heading, body in paragraphs:
            if heading:
                story.append(Paragraph(escape(heading), styles["heading"]))
            story.append(Paragraph(escape(body), styles["body"]))
        if page_number == 0:
            story.append(Spacer(1, 2))
            story.append(Image(str(ROOT / "development/eda/figures/numeric_by_class.png"), width=468, height=132.6))
            story.append(Paragraph("Figure 1. Class-conditioned distributions in the 696-row exploration training subset. "
                                   "Ease and payment differ by class, but distributions overlap.", styles["caption"]))

    def footer(canvas, doc):
        canvas.setFont("Times-Roman", 9)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(72, 756, "CSC311 Group 65374 | Data Exploration section review")
        canvas.drawRightString(540, 36, str(doc.page))

    SimpleDocTemplate(str(directory / "data_exploration_review.pdf"), pagesize=letter,
                      leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72,
                      title="Data Exploration section review", author="Group 65374").build(
                          story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    pages = content()
    render_tex(pages)
    if args.pdf:
        render_pdf(pages)
    print("Built Data Exploration section" + (" and review PDF." if args.pdf else "."))


if __name__ == "__main__":
    main()
