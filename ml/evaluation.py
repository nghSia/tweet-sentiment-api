from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import classification_report, confusion_matrix

from config import config
from ml.model import SentimentModel
from ml.training import RANDOM_STATE, load_dataset, split_dataset
from utils.paths import REPORTS_DIR


METRICS_FILE = REPORTS_DIR / "metrics.json"
CONFUSION_MATRICES_FILE = REPORTS_DIR / "confusion_matrices.json"
PDF_REPORT_FILE = REPORTS_DIR / "evaluation_report.pdf"


def _json_safe(value, ndigits: int = 4):
    if isinstance(value, dict):
        return {key: _json_safe(item, ndigits=ndigits) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item, ndigits=ndigits) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item, ndigits=ndigits) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return round(float(value), ndigits)
    return value


def _task_labels(task_name: str):
    if task_name == "positive":
        return ["not_positive", "positive"], ["human_not_positive", "human_positive"], ["ai_not_positive", "ai_positive"]
    return ["not_negative", "negative"], ["human_not_negative", "human_negative"], ["ai_not_negative", "ai_negative"]


def _compute_task_report(task_name: str, y_true, y_pred):
    target_names, row_labels, column_labels = _task_labels(task_name)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "task": task_name,
        "target_names": target_names,
        "row_labels": row_labels,
        "column_labels": column_labels,
        "matrix": matrix,
        "report": _json_safe(report),
    }


def _metrics_table_rows(task_report):
    rows = []
    for label_name in task_report["target_names"]:
        metrics = task_report["report"].get(label_name, {})
        rows.append(
            [
                label_name,
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1-score", 0),
                metrics.get("support", 0),
            ]
        )
    return rows


def _confusion_table_rows(task_report):
    rows = [["", *task_report["column_labels"]]]
    for row_label, values in zip(task_report["row_labels"], task_report["matrix"]):
        rows.append([row_label, *values])
    return rows


def _create_pdf_report(summary, positive_report, negative_report, pdf_path: Path):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0B1F33"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextSmall",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            spaceAfter=6,
        )
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="Evaluation du modele de sentiment",
    )

    story = [
        Paragraph("Evaluation du modele de sentiment", styles["ReportTitle"]),
        Paragraph(
            f"Validation reproduite avec le split deterministe de training.py (random_state={RANDOM_STATE}).",
            styles["BodyTextSmall"],
        ),
        Paragraph(
            f"Dataset total: {summary['dataset_size']} tweets | Entrainement: {summary['train_size']} | Validation: {summary['validation_size']}",
            styles["BodyTextSmall"],
        ),
    ]

    for task_report in (positive_report, negative_report):
        story.append(Spacer(1, 0.25 * cm))
        story.append(Paragraph(f"Evaluation {task_report['task']}", styles["SectionTitle"]))
        story.append(
            Paragraph(
                "Matrice de confusion: lignes = annotation humaine, colonnes = prediction IA.",
                styles["BodyTextSmall"],
            )
        )

        matrix_table = Table(_confusion_table_rows(task_report), hAlign="LEFT")
        matrix_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE7F2")),
                    ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#EEF3F8")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9AA7B4")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(matrix_table)
        story.append(Spacer(1, 0.2 * cm))

        metrics_table = Table(
            [["classe", "precision", "rappel", "f1", "support"]] + _metrics_table_rows(task_report),
            hAlign="LEFT",
        )
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE7F2")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9AA7B4")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(metrics_table)

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Analyse des biais", styles["SectionTitle"]))
    bias_paragraphs = [
        "Aucun tweet neutre n'est present dans le dataset, ce qui limite la capacite du modele a distinguer les zones ambigues. Une phrase factuelle peut recevoir un score legerement positif, par exemple autour de +0.11 au lieu de 0.",
        "Le corpus est entierement anglais. Une phrase en francais sort du vocabulaire d'apprentissage et le modele tend a produire un score proche de 0, ce qui traduit davantage une absence de signal qu'une vraie comprehension.",
        "Le jeu de validation ne contient que 99 tweets au total, donc la sous-partie de validation reste tres petite. Les metriques varient facilement selon le split, ce qui rend l'estimation instable.",
    ]
    for paragraph in bias_paragraphs:
        story.append(Paragraph(paragraph, styles["BodyTextSmall"]))

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Recommandations", styles["SectionTitle"]))
    recommendations = [
        "Ajouter des exemples neutres pour mieux calibrer la zone de decision autour de 0.",
        "Integrer des tweets francais et, plus largement, des exemples multilingues pour reduire le biais de langue.",
        "Grossir le dataset pour rendre la validation plus stable et les mesures plus fiables.",
    ]
    for recommendation in recommendations:
        story.append(Paragraph(recommendation, styles["BodyTextSmall"]))

    doc.build(story)


def evaluate_and_save(model_path=None, reports_dir: Path | None = None):
    reports_dir = reports_dir or REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_path or config.MODEL_PATH
    texts, y_positive, y_negative = load_dataset()

    (
        texts_train,
        texts_val,
        y_pos_train,
        y_pos_val,
        y_neg_train,
        y_neg_val,
    ) = split_dataset(texts, y_positive, y_negative)

    model = SentimentModel.load(model_path)
    pred_pos, pred_neg = model.predict_labels(texts_val)

    positive_report = _compute_task_report("positive", y_pos_val, pred_pos)
    negative_report = _compute_task_report("negative", y_neg_val, pred_neg)

    summary = {
        "dataset_size": len(texts),
        "train_size": len(texts_train),
        "validation_size": len(texts_val),
        "random_state": RANDOM_STATE,
    }

    metrics_payload = {
        "summary": summary,
        "positive": positive_report,
        "negative": negative_report,
    }
    confusion_payload = {
        "summary": summary,
        "positive": {
            "row_labels": positive_report["row_labels"],
            "column_labels": positive_report["column_labels"],
            "matrix": positive_report["matrix"],
        },
        "negative": {
            "row_labels": negative_report["row_labels"],
            "column_labels": negative_report["column_labels"],
            "matrix": negative_report["matrix"],
        },
    }

    METRICS_FILE.write_text(
        json.dumps(_json_safe(metrics_payload), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    CONFUSION_MATRICES_FILE.write_text(
        json.dumps(_json_safe(confusion_payload), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    _create_pdf_report(summary, positive_report, negative_report, PDF_REPORT_FILE)

    return {
        "metrics_path": METRICS_FILE,
        "confusion_matrices_path": CONFUSION_MATRICES_FILE,
        "pdf_path": PDF_REPORT_FILE,
        "summary": summary,
    }


def main():
    results = evaluate_and_save()
    print("Evaluation terminee.")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()