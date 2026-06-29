"""
Compile doc_type_overrides.json from the audit exceptions list.
Maps file_id -> new doc_type for all manually reviewed classification decisions.

Requires data/metadata.json to exist (produced by get_docs.py).

Usage:
  python pipeline/build_overrides.py
"""
import json
from pathlib import Path

PROJECT_ROOT   = Path(__file__).parent.parent
METADATA_PATH  = PROJECT_ROOT / "data" / "metadata.json"
OVERRIDES_PATH = PROJECT_ROOT / "config" / "doc_type_overrides.json"

# (partial filename fragment, new_type)
EXCEPTIONS = [
    # intake_survey audit
    ("SWS Year 5 Tracker_Completion", "program_logistics"),
    ("SWS_Master Participant Database", "program_logistics"),
    ("24-25 Solving with Students - Coaching Call Log", "engagement_notes"),
    ("24-25 NEW SWS Application Acceptance", "program_logistics"),
    ("23-24 SWS Y3 Alumni Information", "program_logistics"),
    ("EMAILS of 25-26 Solving with Students", "program_logistics"),
    ("23-24 Solving with Students - Year 3 Tracker", "program_logistics"),
    # site_visit audit
    ("Polk Site Visit Notes", "engagement_notes"),
    ("Osceola Site Visit Notes_June 2025", "engagement_notes"),
    ("Osceola Site Visit Notes (Internal)", "engagement_notes"),
    ("Lee Site Visit Notes", "engagement_notes"),
    ("MMC_Cross-District Learning Walk Synthesis", "progress_summary"),
    ("MMC_Hillsborough Learning Walk", "progress_summary"),
    ("MMC_Lake Learning Walk", "progress_summary"),
    ("MMC_Breavard Learning Walk", "progress_summary"),
    ("MMC_ Volusia Learning Walk", "progress_summary"),
    ("Site Visits 1 and 2 Survey", "teacher_practice_data"),
    ("Site Visit 5 Survey_February", "teacher_practice_data"),
    ("Reflection on Year 1_Additions to SV 5", "teacher_practice_data"),
    ("Site Visit 5 Meeting Notes", "engagement_notes"),
    ("Site Visit 4 Survey_January", "teacher_practice_data"),
    ("Site Visit 4 Meeting Notes", "engagement_notes"),
    ("Site Visit 3 Survey_November", "teacher_practice_data"),
    ("Site Visit 3 Meeting Notes", "engagement_notes"),
    ("Hillsborough Kickoff Survey_September 2025", "teacher_practice_data"),
    ("Focus [K-3 Math] Cadre Summary_March 2026", "progress_summary"),
    ("Focus [K-3 Math] Cadre_November 2025 Summary", "progress_summary"),
    ("Focus K-3 Math Cadre Site Visit 3 Survey(1)", "teacher_practice_data"),
    ("Site Visit 1 Meeting Notes", "engagement_notes"),
    ("Focus [K-3 Math] Cadre Summary_September 2025", "progress_summary"),
    # convening audit
    ("TW_Convening Feedback Survey Responses_June 2026", "feedback_survey_data"),
    ("TW_Convening 3 Day 2 Feedback", "feedback_survey_data"),
    ("TW_Convening 3 Day 1 Feedback", "feedback_survey_data"),
    ("TW_Convening 1 Feedback Survey", "feedback_survey_data"),
    ("TW_Convening 2 Feedback", "feedback_survey_data"),
    ("TW_Convening Feedback Survey Responses_April 2025", "feedback_survey_data"),
    ("Focus K-3 Math Cadre_Convening 4 Day 2", "feedback_survey_data"),
    ("Focus K-3 Math Cadre_Convening 4 Day 1", "feedback_survey_data"),
    ("Focus K-3 Math Cadre_Convening 3 Day 2", "feedback_survey_data"),
    ("Focus K-3 Math Cadre_Convening 3 Day 1", "feedback_survey_data"),
    ("Focus K-3 Math Cadre_Convening 2 Survey", "feedback_survey_data"),
    ("Focus K-3 Math Cadre_Convening 1 Slido", "feedback_survey_data"),
    ("Convening Team Huddle Tool_December", "engagement_notes"),
    ("Convening Team Huddle Tool_February", "engagement_notes"),
    ("Focus [K-3 Math] Cadre Summaries_October 2025", "progress_summary"),
    ("Convening 1 Survey_Doug Clements", "feedback_survey_data"),
    ("Convening 1 Survey_K12 Lift", "feedback_survey_data"),
    ("Convening 1 Survey_Overall", "feedback_survey_data"),
    # survey_reflection audit
    ("SWS June 2026 Day 1 Exit Ticket", "feedback_survey_data"),
    ("SWS June 2026 Day 2 Exit Ticket", "feedback_survey_data"),
    ("SWS June 2025 Day 2 Exit Ticket", "feedback_survey_data"),
    ("SWS June 2025 Day 1 Exit Ticket", "feedback_survey_data"),
    ("22-23 Cycle E Reflection", "teacher_practice_data"),
    ("22-23 Cycle D Reflection", "teacher_practice_data"),
    ("Survey 2 Reflection Form (Responses)", "teacher_practice_data"),
    ("Survey 1 Reflection Form (Responses)", "teacher_practice_data"),
    ("Survey 1, 2, 3, and 4 Reflection Form", "teacher_practice_data"),
    ("22-23 Cycle C Reflection", "teacher_practice_data"),
    ("22-23 Cycle B Reflection", "teacher_practice_data"),
    ("22-23 Cycle A Reflection", "teacher_practice_data"),
    ("23-24 Survey 2 Reflections", "teacher_practice_data"),
    ("23-24 Survey 3 Data and Reflections", "teacher_practice_data"),
    ("23-24 Survey 1 Reflections", "teacher_practice_data"),
    # evaluation_report audit
    ("SWS Impact Brief", "impact_data_report"),
    ("Solving with Students 2024-25 Data Slides", "impact_data_report"),
    ("SWS PERTS Graphs From WestEd", "other_data_file"),
    ("SWS PERTS Grade-Level Visualizations from WestEd", "other_data_file"),
    ("SWS Year 3 Report", "impact_data_report"),
    ("SWS Impact Report_2024-25", "impact_data_report"),
    ("SWS Year 2 Report", "impact_data_report"),
    ("SWS Evaluation Report_2022", "impact_data_report"),
    ("Solving with Students Results Slides", "impact_data_report"),
    ("23-24 WestEd_PERTS (Teacher-Level Visualizations)", "other_data_file"),
    # previously null - now evaluation_report
    ("Legends of Learning Usability Study", "evaluation_report"),
    ("Legends of Learning Feasibility Study Report", "evaluation_report"),
    ("Feasibility Study Executive Summary", "evaluation_report"),
    ("Registry of Efficacy and Effectiveness", "evaluation_report"),
    # field_influence audit
    ("Practice-to-Policy Lab One Pager", "program_overview"),
    # planning audit
    ("SPARK District Overview Doc", "program_overview"),
    ("SPARK Program Concept", "program_overview"),
    # readiness_data audit
    ("SPARK_ District Readiness and Needs Assessment_6.16", "intake_survey"),
    ("Needs_Readiness Assessment Breakout Room Notes", "engagement_notes"),
    # focus_group audit
    ("SWS District Leader Focus Group Summary", "qualitative_theming"),
    # null doc assignments
    ("EIR Mid-Phase Q2 Performance Report", "grants_and_funder_reporting"),
    ("TW_Outcomes Survey Context_2025", "progress_summary"),
    ("TW_Cross-District Data Slide Deck_2025", "other_data_file"),
    ("Network Survey_TW Questions Only_Data Report", "other_data_file"),
    ("TW Update to Gates_May 2025", "grants_and_funder_reporting"),
    ("TW Ongoing Funder Check-In Notes", "grants_and_funder_reporting"),
    ("TW_Outcomes Survey Responses_2025", "feedback_survey_data"),
    ("EIR/Game-Based Learning_Running Biweekly Call Notes", "engagement_notes"),
    ("EIR Mid-Phase Annual Performance Report 1_2024", "impact_data_report"),
    ("EIR/Game-Based Learning Results Tracker", "program_logistics"),
    ("Weekly SPARK Meeting Notes", "engagement_notes"),
    ("SWS_Historical Account of Cadre", "progress_summary"),
    ("SWS_Monthly Funder Check-In Call Notes", "grants_and_funder_reporting"),
    ("Focus K-3 Math Cadre_Monthly Progress Summaries", "progress_summary"),
    # district_data audit
    ("Focus K-3_Teacher and Leader Perceptions", "progress_summary"),
    ("Focus K-3_Teacher and Leader Perspectives on Instructional", "progress_summary"),
    ("Focus K-3 Math_Implementation Pattern Summary", "progress_summary"),
    ("Focus K-3_Wave 2 Focus Groups Report", "qualitative_theming"),
    ("Focus K-3_Wave 1 Focus Groups Report", "qualitative_theming"),
    ("Focus K-3_Cadre-Wide Student and Family Empathy", "qualitative_theming"),
    ("Focus K-3_Cross-Site K-3 Math Educator Listening Sessions", "qualitative_theming"),
]


def main(
    metadata_path: Path | None = None,
    overrides_path: Path | None = None,
) -> None:
    metadata_path  = metadata_path  or METADATA_PATH
    overrides_path = overrides_path or OVERRIDES_PATH

    with open(metadata_path, encoding="utf-8") as f:
        docs = json.load(f)

    by_name = {d["file_name"]: d["file_id"] for d in docs}

    overrides: dict = {}
    unmatched: list[str] = []

    for fragment, new_type in EXCEPTIONS:
        matches = [
            (name, fid)
            for name, fid in by_name.items()
            if fragment.lower() in name.lower()
        ]
        if not matches:
            unmatched.append(fragment)
        else:
            for name, fid in matches:
                overrides[fid] = {"file_name": name, "doc_type": new_type}

    print(f"  Matched: {len(overrides)} overrides")
    if unmatched:
        print(f"  Unmatched ({len(unmatched)}):")
        for u in unmatched:
            print(f"    - {u}")

    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overrides_path, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)
    print(f"  Written: {overrides_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
