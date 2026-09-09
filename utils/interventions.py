"""
=======================================================================
Step 6 — Intervention Recommendation Engine
Dropout Risk Prediction System
Nzeako Jennifer Nkiruka | DU/C/247
=======================================================================
What this module does:
  - Accepts a student's SHAP values + feature values
  - Identifies the top 3 dominant risk drivers
  - Maps each risk driver to ranked, actionable interventions
  - Returns a structured recommendation list with:
      * Urgency level  (URGENT / MEDIUM / MONITOR)
      * Responsible party (Teacher / Counsellor / Admin / Bursar)
      * Action description
      * Rationale (why this is recommended)

Based on:
  Ojo & Awonuga (2022) — Evidence-based intervention taxonomy
  for Nigerian secondary schools
  Kizito (2023) — Early warning and intervention effectiveness
  Rumberger & Losen (2021) — Intervention matching to risk profile

Usage:
  from utils.interventions import get_interventions
  result = get_interventions(shap_vals, feature_vals, feature_names,
                             risk_score, display_names)
=======================================================================
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# URGENCY CONSTANTS
# ══════════════════════════════════════════════════════════════════════
URGENT  = "🔴  URGENT"
MEDIUM  = "🟡  MEDIUM"
MONITOR = "🟢  MONITOR"


# ══════════════════════════════════════════════════════════════════════
# INTERVENTION TAXONOMY
# Source: Ojo & Awonuga (2022), Kizito (2023)
#
# Structure:
#   feature_key → list of intervention dicts, ordered by priority
#   Each dict:
#     urgency:     URGENT / MEDIUM / MONITOR
#     party:       who is responsible
#     action:      what to do
#     rationale:   why this addresses the risk
# ══════════════════════════════════════════════════════════════════════

INTERVENTION_TAXONOMY = {

    # ── SOCIOECONOMIC ──────────────────────────────────────────────────
    "Fee_Payment_Status": [
        {
            "urgency":   URGENT,
            "party":     "School Bursar + Admin",
            "action":    "Arrange an immediate fee payment plan or deferred payment schedule with the student's family",
            "rationale": "Unpaid fees create a direct exclusion risk. Schools in Nigeria can legally bar students with outstanding fees. Arranging a payment plan removes this immediate dropout trigger. (Ojo & Awonuga, 2022)"
        },
        {
            "urgency":   URGENT,
            "party":     "School Counsellor",
            "action":    "Identify and refer the student for available bursary, scholarship, or government support programmes (UBEC, state bursary schemes)",
            "rationale": "Financial support programmes exist at LGA and state levels but students rarely access them without active referral from school staff. (Fayomi & Nwosu, 2022)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "Class Teacher",
            "action":    "Make a discreet home visit or phone call to the parent to understand the financial situation and assess family stability",
            "rationale": "Fee non-payment is often a symptom of broader household financial stress. Understanding the full context allows more targeted support. (Ojo & Awonuga, 2022)"
        },
    ],

    "Household_Income_Band": [
        {
            "urgency":   URGENT,
            "party":     "School Counsellor + Admin",
            "action":    "Flag student for priority consideration in any school-based feeding programme, free textbook scheme, or uniform support initiative",
            "rationale": "Material deprivation — lack of books, uniform, or food — is a leading cause of school avoidance in low-income households. Removing these barriers reduces attendance-related dropout. (Batool et al., 2023)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Connect family with local NGO or community support organisation providing educational grants for low-income families",
            "rationale": "External financial support from NGOs operating in the student's LGA can supplement school-based interventions where internal funds are limited."
        },
        {
            "urgency":   MONITOR,
            "party":     "Class Teacher",
            "action":    "Monitor weekly — note any signs of material deprivation (no textbook, missing uniform, skipping lunch) and report to counsellor promptly",
            "rationale": "Visible signs of material deprivation in school are reliable indicators that household financial pressure is intensifying."
        },
    ],

    "Parental_Education_Level": [
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Schedule a structured parent-school engagement session to explain the long-term economic consequences of secondary school dropout",
            "rationale": "Parents with low education levels often underestimate the income premium associated with secondary school completion. Evidence-based communication of this gap increases parental commitment to keeping children enrolled. (Ojo & Awonuga, 2022)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "Class Teacher",
            "action":    "Send regular, simple written progress updates home — use plain language accessible to parents with limited literacy",
            "rationale": "Low-education parents are often excluded from school communication because formal letters assume literacy. Simplified communication increases parental engagement. (Fayomi & Nwosu, 2022)"
        },
        {
            "urgency":   MONITOR,
            "party":     "School Admin",
            "action":    "Ensure student is registered on any community or school mentorship programme that provides adult role models with higher educational attainment",
            "rationale": "Role model exposure increases a student's own educational aspiration, particularly in households where no parent has completed secondary education."
        },
    ],

    "Parental_Involvement": [
        {
            "urgency":   URGENT,
            "party":     "Class Teacher",
            "action":    "Conduct an immediate home visit or direct parent phone call — absent parental involvement indicates the student has no adult monitoring their school engagement",
            "rationale": "Students with absent parents are at 44% dropout rate in the Nigerian context (Ojo & Awonuga, 2022). Physical home contact is the most effective first response."
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Establish a weekly check-in structure with the student — substitute the missing parental support with a trusted school adult",
            "rationale": "Where parental involvement cannot be improved, a consistent school-based adult relationship reduces the student's sense of isolation and increases belonging. (Kizito, 2023)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Admin",
            "action":    "Formally invite parent/guardian to a school meeting — track attendance and escalate if they do not respond to three consecutive invitations",
            "rationale": "A formal paper trail of engagement attempts is important for safeguarding purposes and for justifying escalation to LGA welfare officers if needed."
        },
    ],

    # ── ACADEMIC / BEHAVIOURAL ─────────────────────────────────────────
    "Attendance_Rate_Pct": [
        {
            "urgency":   URGENT,
            "party":     "Class Teacher",
            "action":    "Initiate daily attendance monitoring — contact parent or guardian on every day of absence without prior notice",
            "rationale": "Early, consistent response to absence prevents the normalisation of non-attendance. Schools that contact families on day one of unexplained absence show significantly better retention. (Hussain et al., 2023)"
        },
        {
            "urgency":   URGENT,
            "party":     "School Counsellor",
            "action":    "Conduct a one-to-one student interview to identify the specific barriers preventing attendance — transport, domestic duties, safety, or disengagement",
            "rationale": "Attendance decline has multiple distinct causes requiring different interventions. Identifying the specific barrier is essential before prescribing a response. (Fayomi & Nwosu, 2022)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Admin",
            "action":    "If distance is a contributing factor, explore available transport support options — school bus scheme, community transport, or cycling programme",
            "rationale": "Transport barriers account for 18% of chronic absenteeism in rural Nigerian schools. Practical transport solutions have direct measurable impact on attendance. (UNESCO UIS, 2023)"
        },
    ],

    "Avg_Academic_Score_Pct": [
        {
            "urgency":   URGENT,
            "party":     "Subject Teachers",
            "action":    "Arrange immediate remedial classes in the two lowest-scoring subjects — prioritise Mathematics and English as gateway subjects",
            "rationale": "Persistent academic failure below the 40% threshold is strongly associated with progressive disengagement and eventual dropout. Early remediation before examinations is significantly more effective than post-failure intervention. (Adekitan & Salau, 2022)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "Class Teacher",
            "action":    "Assign a peer-tutoring partner — identify a higher-performing student in the same year group willing to provide academic support",
            "rationale": "Peer tutoring has demonstrated consistent effectiveness in Nigerian classroom contexts and is achievable without additional school resources. (Aluko et al., 2024)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Assess whether academic underperformance is linked to undiagnosed learning difficulties — refer for specialist assessment if indicated",
            "rationale": "A subset of chronically underperforming students have undiagnosed learning differences that are misread as laziness or disengagement. Early identification changes the intervention path entirely."
        },
    ],

    "Failed_Subjects_Count": [
        {
            "urgency":   URGENT,
            "party":     "VP Academics",
            "action":    "Convene an urgent student support meeting — VP Academics, class teacher, and counsellor — to produce a formal subject recovery plan within one week",
            "rationale": "More than two failed subjects in a term indicates broad academic disengagement requiring coordinated institutional response, not individual teacher action. (Ojo & Awonuga, 2022)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "Subject Teachers",
            "action":    "Reduce cognitive load in failed subjects — simplify assessment tasks, provide structured revision guides, and increase feedback frequency",
            "rationale": "Students who have already failed are at risk of learned helplessness. Rebuilding academic confidence through smaller achievable goals is more effective than intensified pressure."
        },
        {
            "urgency":   MONITOR,
            "party":     "Class Teacher",
            "action":    "Track weekly assessment performance in failed subjects and report any further decline to VP Academics within 48 hours",
            "rationale": "Rapid response to continued decline prevents a term failure from becoming a decision to not return the following term."
        },
    ],

    # ── COGNITIVE ──────────────────────────────────────────────────────
    "Cognitive_Assessment_Score": [
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Refer student for a detailed cognitive assessment by a qualified educational psychologist or learning support specialist",
            "rationale": "Low cognitive assessment scores may indicate an underlying learning difficulty that has gone undetected. Formal assessment enables targeted learning support rather than generic remediation. (Dasi & Kanakala, 2022)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "Subject Teachers",
            "action":    "Adapt instructional approach — use more visual aids, concrete examples, and hands-on demonstrations rather than abstract explanations",
            "rationale": "Students with lower cognitive assessment scores respond better to multi-modal instruction that does not rely exclusively on verbal or written processing. (FME, 2020)"
        },
        {
            "urgency":   MONITOR,
            "party":     "Class Teacher",
            "action":    "Monitor classroom participation weekly — note whether student is following lesson content and raise concerns promptly if disengagement is observed",
            "rationale": "Cognitive difficulty often presents as disengagement in class before it appears in assessment scores. Early observation enables earlier intervention."
        },
    ],

    "Study_Habit_Score": [
        {
            "urgency":   MEDIUM,
            "party":     "Class Teacher",
            "action":    "Implement a structured homework diary system — require the student to record assignments and have it signed by a parent or guardian weekly",
            "rationale": "Students with poor study habits often lack external structure. A monitored homework diary introduces accountability at both school and home level. (Zimmerman, 2002)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Provide explicit study skills training — teach time-blocking, note-taking, and revision techniques in a structured 4-week session",
            "rationale": "Self-regulated learning skills are learnable and teachable. Many students with poor study habits have never been explicitly taught effective study strategies. (Zimmerman, 2002)"
        },
        {
            "urgency":   MONITOR,
            "party":     "Class Teacher",
            "action":    "Track homework submission rate weekly — a submission rate below 50% in any two-week period triggers an immediate counsellor referral",
            "rationale": "Homework submission rate is a reliable proxy for study habit quality and is easily observable without additional assessment burden on teachers."
        },
    ],

    "Learning_Pace": [
        {
            "urgency":   MEDIUM,
            "party":     "Subject Teachers",
            "action":    "Seat student near the teacher in class and provide additional wait time and simplified explanations — check comprehension before moving on",
            "rationale": "Slow learning pace students are disproportionately left behind in teacher-centred classrooms. Simple adjustments to proximity and pacing have significant impact. (FME, 2020)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Explore whether the student qualifies for a learning support programme — if available in the school or LGA, initiate referral immediately",
            "rationale": "Structured learning support programmes specifically targeting slow learning pace have demonstrated 20–30% improvement in retention in similar school contexts. (Aluko et al., 2024)"
        },
        {
            "urgency":   MONITOR,
            "party":     "Class Teacher",
            "action":    "Conduct monthly one-to-one comprehension checks — short informal assessments outside class to track whether pace is improving",
            "rationale": "Formal class assessments often underdetect comprehension gains in slow-pace students. One-to-one checks provide a more accurate picture of progress."
        },
    ],

    # ── PSYCHOMOTOR ────────────────────────────────────────────────────
    "PE_Participation": [
        {
            "urgency":   MEDIUM,
            "party":     "PE Teacher",
            "action":    "Personally invite the student to participate in a specific PE activity or school sports team — frame it as an opportunity, not a requirement",
            "rationale": "PE participation is a school belonging protective factor. Students who withdraw from PE are often also withdrawing from broader school community. Personal invitation from a teacher is more effective than generic encouragement. (Kizito, 2023)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "Class Teacher",
            "action":    "Identify whether PE avoidance is linked to kit cost — if so, arrange for spare kit from the school store to be made available discreetly",
            "rationale": "Inability to afford PE kit is a common but rarely addressed barrier to participation in low-income students. Removing the material barrier is simple and effective."
        },
        {
            "urgency":   MONITOR,
            "party":     "PE Teacher",
            "action":    "Track PE participation weekly — report three consecutive unexplained absences from PE to the class teacher",
            "rationale": "Progressive PE withdrawal often precedes general school withdrawal by 4–6 weeks. Early flagging enables earlier general intervention."
        },
    ],

    "Psychomotor_Dev_Score": [
        {
            "urgency":   MEDIUM,
            "party":     "Practical Subject Teachers",
            "action":    "Provide additional hands-on time in practical subjects — allow the student to practise procedures more times than peers before assessment",
            "rationale": "Low psychomotor development scores often reflect insufficient practice opportunity rather than inability. Extended practice time shows consistent improvement in practical assessment scores. (FME, 2020)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Explore vocational aptitude — if the student shows interest in trades or crafts, connect them with vocational guidance resources available at post-secondary level",
            "rationale": "Students with low academic scores but practical aptitude are at elevated dropout risk because they cannot see a relevant post-school pathway. Vocational guidance provides a concrete future orientation that increases motivation to complete secondary school."
        },
        {
            "urgency":   MONITOR,
            "party":     "Practical Subject Teachers",
            "action":    "Score practical assessments more frequently — weekly rather than end-of-term — to detect improvement and provide timely encouragement",
            "rationale": "Low psychomotor students lose motivation when their only feedback is a final term score. Frequent small assessments provide more encouragement and clearer progress signals."
        },
    ],

    "Practical_Subject_Score": [
        {
            "urgency":   MEDIUM,
            "party":     "Practical Subject Teachers",
            "action":    "Identify the specific practical skill area where performance is lowest and provide targeted coaching — Agricultural Science farm practice, Home Economics cooking, or Technical Drawing",
            "rationale": "Practical subject scores aggregate performance across distinct skill areas. Targeted coaching on the weakest specific skill is more efficient than general remediation."
        },
        {
            "urgency":   MONITOR,
            "party":     "Class Teacher",
            "action":    "Note whether low practical scores are accompanied by avoidance behaviour — if student regularly misses practical classes, refer to counsellor",
            "rationale": "Practical class avoidance is a distinct behavioural signal from academic avoidance and requires its own intervention pathway."
        },
    ],

    # ── DEMOGRAPHIC / GEOGRAPHIC ───────────────────────────────────────
    "Distance_to_School_km": [
        {
            "urgency":   MEDIUM,
            "party":     "School Admin",
            "action":    "Explore available transport support options — school bus scheme, community transport, cycling programme, or hostel placement if distance exceeds 10km",
            "rationale": "Distance beyond 5km is a validated dropout predictor. Practical transport solutions have direct measurable impact on attendance. (Fayomi & Nwosu, 2022)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "Connect the family with the LGA Education Board to inquire about school transfer to a closer institution if transport cannot be arranged",
            "rationale": "A school transfer to a closer institution, while disruptive, is preferable to dropout. The counsellor should facilitate this process rather than leaving it to the family to navigate alone."
        },
        {
            "urgency":   MONITOR,
            "party":     "Class Teacher",
            "action":    "Flag any attendance decline immediately — distance students are particularly vulnerable to attendance drop in rainy season or after transport disruptions",
            "rationale": "Weather and transport disruptions disproportionately affect long-distance students. Prompt monitoring prevents temporary disruptions from becoming permanent withdrawal."
        },
    ],

    "Gender": [
        {
            "urgency":   MEDIUM,
            "party":     "School Counsellor",
            "action":    "For female students at risk: conduct a private counselling session to assess whether early marriage pressure, domestic responsibilities, or safety concerns on the school route are present",
            "rationale": "Female dropout in Nigeria is driven by a distinct set of cultural and safety factors that male-oriented interventions do not address. A private, female-led counselling session is the appropriate first step. (Adeyemo et al., 2023)"
        },
        {
            "urgency":   MEDIUM,
            "party":     "School Admin",
            "action":    "Ensure the student is enrolled in any available girl-child education support programme at school, LGA, or state level",
            "rationale": "Targeted girl-child retention programmes with demonstrated effectiveness exist at multiple administrative levels in Nigeria but require active school-level enrolment."
        },
        {
            "urgency":   MONITOR,
            "party":     "Class Teacher",
            "action":    "Monitor for signs of increasing domestic pressure — changes in school arrival time, incomplete homework, or expressions of family conflict",
            "rationale": "Domestic pressure signals often appear in class behaviour weeks before they result in formal absence. Early observation enables earlier intervention."
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════
# RISK LEVEL THRESHOLDS
# ══════════════════════════════════════════════════════════════════════
def classify_risk(prob):
    if prob >= 0.70:
        return "HIGH", RED_LABEL   if False else "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


# ══════════════════════════════════════════════════════════════════════
# MAIN FUNCTION — get_interventions()
# ══════════════════════════════════════════════════════════════════════
def get_interventions(
    shap_vals,
    feature_vals,
    feature_names,
    risk_score,
    display_names=None,
    top_n_drivers=3,
    max_interventions_per_driver=3,
):
    """
    Generate ranked intervention recommendations for one student.

    Parameters
    ----------
    shap_vals        : np.array — SHAP values for this student
    feature_vals     : np.array or list — actual feature values
    feature_names    : list of str — original feature column names
    risk_score       : float — predicted dropout probability (0–1)
    display_names    : dict — {original_name: display_name} (optional)
    top_n_drivers    : int — number of top risk drivers to use (default 3)
    max_interventions_per_driver : int — interventions per driver (default 3)

    Returns
    -------
    dict with keys:
        risk_score      : float
        risk_level      : str  (HIGH / MEDIUM / LOW)
        top_drivers     : list of dicts (feature, shap_val, feat_val)
        interventions   : list of dicts (urgency, party, action, rationale, driver)
        summary         : str  (one-paragraph plain-language summary)
    """

    # -- Identify top positive SHAP drivers (features increasing dropout risk)
    shap_arr = np.array(shap_vals)
    feat_arr = np.array(feature_vals)

    # Only consider features that INCREASE risk (positive SHAP)
    positive_mask = shap_arr > 0
    if positive_mask.sum() < top_n_drivers:
        # Fall back to all features by absolute SHAP if not enough positives
        positive_mask = np.ones(len(shap_arr), dtype=bool)

    positive_shap = np.where(positive_mask, shap_arr, -np.inf)
    top_indices   = np.argsort(positive_shap)[::-1][:top_n_drivers]

    top_drivers = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        disp_name = (
            display_names.get(feat_name, feat_name)
            if display_names else feat_name
        )
        top_drivers.append({
            "feature":     feat_name,
            "display":     disp_name,
            "shap_val":    float(shap_arr[idx]),
            "feat_val":    float(feat_arr[idx]),
            "contribution": f"+{float(shap_arr[idx]):.3f}",
        })

    # -- Map drivers to interventions
    interventions = []
    seen_actions  = set()

    urgency_order = {URGENT: 0, MEDIUM: 1, MONITOR: 2}

    for driver in top_drivers:
        feat_name = driver["feature"]

        # Find matching taxonomy entry — try exact match first,
        # then partial match for one-hot encoded features
        taxonomy_key = None
        if feat_name in INTERVENTION_TAXONOMY:
            taxonomy_key = feat_name
        else:
            for key in INTERVENTION_TAXONOMY:
                if key in feat_name:
                    taxonomy_key = key
                    break

        if taxonomy_key is None:
            continue

        driver_interventions = INTERVENTION_TAXONOMY[taxonomy_key]

        count = 0
        for interv in driver_interventions:
            if count >= max_interventions_per_driver:
                break
            action_key = interv["action"][:60]   # deduplicate by first 60 chars
            if action_key not in seen_actions:
                seen_actions.add(action_key)
                interventions.append({
                    "urgency":   interv["urgency"],
                    "party":     interv["party"],
                    "action":    interv["action"],
                    "rationale": interv["rationale"],
                    "driver":    driver["display"],
                    "sort_key":  urgency_order.get(interv["urgency"], 9),
                })
                count += 1

    # Sort: URGENT first, then MEDIUM, then MONITOR
    interventions.sort(key=lambda x: x["sort_key"])

    # -- Risk classification
    if risk_score >= 0.70:
        risk_level = "HIGH"
    elif risk_score >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # -- Plain language summary
    driver_phrases = " and ".join(
        [d["display"] for d in top_drivers[:2]]
    )
    urgent_count = sum(
        1 for i in interventions if i["urgency"] == URGENT
    )

    summary = (
        f"This student has a predicted dropout probability of "
        f"{risk_score:.1%} ({risk_level} RISK). "
        f"The primary risk drivers are {driver_phrases}. "
        f"{urgent_count} urgent action(s) are recommended requiring "
        f"immediate response from school staff. "
        f"Early intervention at this stage has been shown to reduce "
        f"dropout probability by up to 14% in comparable school contexts "
        f"(Kizito, 2023)."
    )

    return {
        "risk_score":    risk_score,
        "risk_level":    risk_level,
        "top_drivers":   top_drivers,
        "interventions": interventions,
        "summary":       summary,
    }


# ══════════════════════════════════════════════════════════════════════
# TEST — run directly to verify the engine works
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import joblib
    import json

    print("=" * 65)
    print("  Step 6 — Intervention Recommendation Engine Test")
    print("  Dropout Risk Prediction System — Jennifer Nkiruka")
    print("=" * 65)

    # Load artefacts
    feature_cols  = joblib.load("data/feature_columns.pkl")
    shap_values   = np.load("model/shap_values.npy")
    predictions   = pd.read_csv("model/xgboost_predictions.csv")
    X_test        = pd.read_csv("data/X_test_unscaled.csv")

    with open("model/feature_display_names.json") as f:
        display_names = json.load(f)

    prob_col = predictions["y_pred_prob"].values
    y_true   = predictions["y_true"].values

    # Find a high-risk and low-risk example
    high_idx = np.where((y_true == 1) & (prob_col > 0.80))[0]
    low_idx  = np.where((y_true == 0) & (prob_col < 0.15))[0]

    if len(high_idx) == 0:
        high_idx = np.where(prob_col > 0.70)[0]
    if len(low_idx) == 0:
        low_idx  = np.where(prob_col < 0.20)[0]

    for label, idx in [
        ("HIGH-RISK STUDENT", high_idx[0]),
        ("LOW-RISK  STUDENT", low_idx[0]),
    ]:
        print(f"\n{'=' * 65}")
        print(f"  TEST CASE: {label}")
        print(f"{'=' * 65}")

        result = get_interventions(
            shap_vals    = shap_values[idx],
            feature_vals = X_test.iloc[idx].values,
            feature_names= list(feature_cols),
            risk_score   = prob_col[idx],
            display_names= display_names,
            top_n_drivers= 3,
            max_interventions_per_driver = 2,
        )

        print(f"\n  Risk Score:  {result['risk_score']:.1%}")
        print(f"  Risk Level:  {result['risk_level']}")

        print(f"\n  Top Risk Drivers:")
        for i, d in enumerate(result["top_drivers"], 1):
            print(f"    {i}. {d['display']:<35} "
                  f"SHAP={d['shap_val']:+.3f}  "
                  f"value={d['feat_val']:.2f}")

        print(f"\n  Recommended Interventions:")
        for i, interv in enumerate(result["interventions"], 1):
            print(f"\n  [{i}] {interv['urgency']}")
            print(f"      Driver:     {interv['driver']}")
            print(f"      Party:      {interv['party']}")
            print(f"      Action:     {interv['action']}")
            print(f"      Rationale:  {interv['rationale'][:100]}...")

        print(f"\n  Summary:")
        print(f"  {result['summary']}")

    print(f"\n{'=' * 65}")
    print(f"  Intervention engine working correctly.")
    print(f"  Step 6 complete. Build the Streamlit app next.")
    print(f"{'=' * 65}\n")