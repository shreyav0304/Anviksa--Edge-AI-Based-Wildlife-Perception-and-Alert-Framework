"""Create the user-video provenance report from source evidence only."""

from __future__ import annotations

import csv
from pathlib import Path

from audit_global_videos import USER_EVIDENCE


ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / "metadata/global_video_manifest.csv"
REPORT = ROOT / "reports/user_video_provenance_report.csv"
SUMMARY = ROOT / "reports/user_video_provenance_summary.md"
DATASET_URL = "https://doi.org/10.26180/29624891.v2"
ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12582407/"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
FIELDS = (
    "filename", "source_recovered", "original_source_url", "original_title",
    "species_common_name", "species_scientific_name",
    "species_verification_status", "species_verification_source",
    "venom_status", "venom_status_source", "assigned_class",
    "license_status", "license_evidence", "eligible_for_dataset",
    "confidence_of_provenance", "manual_review_required", "notes",
)


def main() -> None:
    with GLOBAL.open(newline="", encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
        manifest_fields = list(manifest[0])
    user_rows = [row for row in manifest if row["source_type"] == "user_collected"]
    report_rows = []
    for row in user_rows:
        evidence = USER_EVIDENCE.get(row["filename"])
        if evidence:
            row.update({
                "current_class": "Venomous_Snake",
                "verification_status": "VERIFIED",
                "species_common_name": evidence["common"],
                "species_scientific_name": evidence["scientific"],
                "species_verification_source": DATASET_URL,
                "venom_status_verification_source": ARTICLE_URL,
                "license_verification_status": "REVIEW",
                "eligible_for_dataset": "REVIEW",
                "exclusion_reason": "LICENSE_NOT_VERIFIED",
                "notes": (
                    "Exact local filename and byte size match the Monash/Figshare dataset. "
                    "Dataset license is CC BY-NC-ND 4.0; ML-training reuse requires eligibility review."
                ),
            })
            result = {
                "filename": row["filename"], "source_recovered": "YES",
                "original_source_url": DATASET_URL,
                "original_title": 'High speed videos of snake strikes for "Kinematics of strikes in venomous snakes"',
                "species_common_name": evidence["common"],
                "species_scientific_name": evidence["scientific"],
                "species_verification_status": "VERIFIED",
                "species_verification_source": DATASET_URL,
                "venom_status": "VENOMOUS", "venom_status_source": ARTICLE_URL,
                "assigned_class": "Venomous_Snake", "license_status": "REVIEW",
                "license_evidence": f"CC BY-NC-ND 4.0 ({LICENSE_URL}); training reuse not assumed.",
                "eligible_for_dataset": "REVIEW", "confidence_of_provenance": "HIGH",
                "manual_review_required": "YES",
                "notes": "Exact filename and byte-size match; license eligibility requires a human/legal decision.",
            }
        else:
            row.update({
                "current_class": "Unverified", "verification_status": "UNVERIFIED",
                "species_common_name": "", "species_scientific_name": "",
                "species_verification_source": "", "venom_status_verification_source": "",
                "license_verification_status": "UNVERIFIED", "eligible_for_dataset": "NO",
                "exclusion_reason": "LABEL_NOT_VERIFIED",
            })
            if row["filename"].startswith("TE1-"):
                note = "Camera identifier and timestamp only; exact identifier searches found no credible snake source."
            else:
                note = "Timestamp-style filename only; no source reference found in project records or public exact-name search."
            result = {
                "filename": row["filename"], "source_recovered": "NO",
                "original_source_url": "", "original_title": "",
                "species_common_name": "", "species_scientific_name": "",
                "species_verification_status": "UNVERIFIED",
                "species_verification_source": "", "venom_status": "UNVERIFIED",
                "venom_status_source": "", "assigned_class": "Unverified",
                "license_status": "UNKNOWN", "license_evidence": "",
                "eligible_for_dataset": "NO", "confidence_of_provenance": "LOW",
                "manual_review_required": "YES", "notes": note,
            }
        report_rows.append(result)

    with GLOBAL.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest)
    with REPORT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    recovered = sum(row["source_recovered"] == "YES" for row in report_rows)
    unresolved = sum(row["assigned_class"] == "Unverified" for row in report_rows)
    summary = [
        "# User Video Provenance Summary", "",
        "## Method", "",
        "All 43 user-video filenames, project records, OpenCV metadata, filesystem metadata, and distinctive public identifiers were investigated. `ffprobe` was unavailable in the environment. No descriptive source tags were recoverable through available metadata. No visual or AI identification was used.", "",
        "The 10 STV files exactly match filenames and byte sizes in the Monash/Figshare dataset linked by the peer-reviewed Journal of Experimental Biology article. Seven are newly recovered in this phase; three were previously verified and were rechecked.", "",
        "The dataset is licensed CC BY-NC-ND 4.0. Because ML-training eligibility under the NoDerivatives term is not assumed, all 10 remain REVIEW and require a human/legal eligibility decision.", "",
        "## Counts", "",
        "- Previously unresolved videos investigated: 40",
        "- Newly recovered sources: 7",
        "- Newly species-verified: 7",
        "- Newly verified venomous: 7",
        "- Newly verified non-venomous: 0",
        f"- Still unverified: {unresolved}",
        "- New dataset-eligible: 0",
        f"- License review required: {recovered}",
        f"- Not eligible: {unresolved}",
        "- Manual review required: 43", "",
        "## Updated Total Dataset Status", "",
        "- Total eligible venomous videos: 4",
        "- Total eligible non-venomous videos: 4",
        "- Total eligible videos: 8",
        f"- Total unverified: {unresolved}",
        f"- Total review required: {recovered}", "",
        "## Integrity", "",
        "- clean_dataset modified: NO",
        "- existing test dataset modified: NO",
        "- model modified: NO",
        "- TFLite modified: NO",
        "- training performed: NO",
        "- training frame extraction performed: NO",
        "- representative contact-sheet previews created: NO", "",
        "No files were moved. The existing source group for the two TE1-047 partial-overlap candidates is preserved.",
        "", "## Final State", "",
        "- USER VIDEO PROVENANCE AUDIT: COMPLETE",
        "- USER LABEL RECOVERY: PARTIAL",
        "- READY FOR DATASET ELIGIBILITY REVIEW: YES",
        "- READY FOR VIDEO SPLITTING: NO",
        "- READY FOR TRAINING FRAME EXTRACTION: NO",
    ]
    SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
