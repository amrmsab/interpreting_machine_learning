# Chapter 2: Governing the Black Box: Stakeholders, Regulation, and Explainable AI

**Omar Emad**

## 1. Introduction: Is the Algorithm's Word Enough?

Building on the technical foundations of machine learning discussed in Chapter 1, we must now confront the socio-technical reality of deploying these models in the real world.

Imagine this scenario: You apply for a vital bank loan to buy a house, but your application is rejected. When you ask the bank why, the representative simply says: _"The algorithm categorized you as high risk. The machine decided."_

Is that acceptable? Who deserves an explanation in that room, and what kind of explanation do they deserve? More importantly, is the bank legally required to provide one?

Artificial Intelligence (AI) is now part of the core infrastructure for high-stakes decision-making in employment, criminal justice, credit, healthcare, education, and public administration. As algorithms increasingly affect human lives, opening the "black box" of AI is no longer only a debugging exercise for software engineers. It becomes a political, ethical, and legal challenge about accountability, contestability, and power.

This chapter explores why Explainable AI (XAI) is an essential governance tool for distributing accountability across the AI ecosystem. It examines the different needs of AI stakeholders, critically discusses regulatory frameworks such as the [GDPR](#41-the-gdpr-and-the-right-to-explanation-debate) and the [EU AI Act](#42-the-eu-ai-act-2024), explains the theoretical limits of transparency, and introduces practical engineering approaches for building more defensible AI systems.

**Chapter Flow:** [Section 2](#2-what-is-an-explanation-explanatory-pragmatism-and-measurement) defines what makes an explanation useful and measurable. [Section 3](#3-the-stakeholder-ecosystem-conflicting-needs) maps the conflicting transparency needs of AI stakeholders. [Section 4](#4-the-regulatory-landscape-from-data-privacy-to-product-safety) examines the regulatory shift from data protection to product safety. [Section 5](#5-theoretical-deep-dive-intuition-vs-normative-defensibility) explains why technical transparency alone cannot guarantee fairness. [Section 6](#6-ecosystem-governance-the-scor-framework) introduces the SCOR framework as an emerging ecosystem governance model. [Section 7](#7-practical-implementation-bridging-the-last-mile) provides a practical Natural Language Explanation (NLE) implementation pattern, and [Section 8](#8-reflective-exercises-and-discussion) offers discussion prompts.

---

## 2. What Is an Explanation? Explanatory Pragmatism and Measurement

Before regulating explanations, we must define what makes an explanation _good_ and how it can be assessed.

### 2.1 Explanatory Pragmatism

In XAI, explanations should not be treated as raw mathematical outputs. A list of model weights, feature importances, or probability scores may be technically accurate, but it may still fail to help a human understand or contest a decision. Social-science research on explanation shows that explanations are selective, contrastive, and audience-dependent: people usually ask why one outcome happened instead of another, and they expect an answer that is relevant to their context ([Miller, 2019](#ref-miller)).

This chapter therefore adopts an explanatory pragmatist view. Under this view, an explanation is a communicative act in which an explainer provides information to help a specific audience achieve understanding, make a decision, or seek redress. A pragmatic explanation should be:

- **Factually correct:** It must accurately reflect the model, data, and decision process.
- **Useful and context-specific:** It must provide information that is meaningful in the recipient's situation.
- **User-specific:** It must match the recipient's technical knowledge and practical needs.
- **Actionable where possible:** It should help the recipient understand what can be challenged, corrected, or improved.
- **Pluralistic:** It should acknowledge that different stakeholders may need different explanations for the same system.

This matters because a bank auditor, a machine-learning engineer, a loan officer, and a rejected applicant do not need the same explanation. The engineer may need feature attributions and error distributions. The applicant may need understandable reasons and a path to human review.

### 2.2 Measuring Transparency: IEEE P7001 and IEEE 7001-2021

For software engineers, abstract ethical principles must be translated into testable requirements. Winfield et al. (2021) discuss **IEEE P7001** as a proposed transparency standard for autonomous systems, presenting transparency and explainability as measurable properties rather than vague ideals ([Winfield et al., 2021](#ref-winfield)). The later formal standard, **IEEE 7001-2021**, describes measurable and testable levels of transparency for autonomous systems ([IEEE Standards Association, 2021](#ref-ieee7001)).

A simplified teaching-oriented version of the end-user transparency levels is:

- **Level 1:** A comprehensive user manual is provided.
- **Level 2:** The manual is presented through an interactive visualization or simulation.
- **Level 3:** The system includes a _"why did you just do that?"_ function, explaining previous actions.
- **Level 4:** The system includes a _"what would you do if...?"_ function, allowing users to test hypothetical scenarios.
- **Level 5:** A future or maximum level of transparency, depending on the stakeholder category and system context.

The important point is not simply the numbering. The standardization approach shows that transparency can be specified, tested, audited, and improved as part of engineering practice.

---

## 3. The Stakeholder Ecosystem: Conflicting Needs

If an explanation is a communicative act tailored to an audience, we must map who the audience is. XAI research emphasizes that explainability must satisfy different stakeholder desiderata: different stakeholders have different interests, expectations, and needs regarding artificial systems ([Langer et al., 2021](#ref-langer)).

A practical stakeholder map includes five groups:

1. **Developers, such as ML engineers and data scientists:** They build the system and need reliability, performance, debugging support, error analysis, and model monitoring. Their explanations may include feature-attribution graphs, counterfactual examples, calibration metrics, and failure-case analysis.
2. **Deployers, such as banks, hospitals, universities, and public agencies:** They implement the system in an organizational process. They need legal compliance, operational usability, documentation, audit trails, and evidence that staff can use the system appropriately.
3. **End-users, such as doctors, bank clerks, recruiters, or case workers:** They interact with the system during daily work. They need interpretability, trust calibration, and clear guidance on when to rely on the system and when to override it.
4. **Affected persons and communities, such as patients, loan applicants, job applicants, or students:** They are affected by the final decision. They need fairness, correctness, meaningful reasons, contestability, and redress.
5. **Regulators, auditors, and courts:** They evaluate whether the system complies with legal, safety, and accountability requirements. They need documentation, traceability, risk assessments, incident reports, and evidence of human oversight.

**The central tension:** One mathematical explanation cannot satisfy all these groups. A SHAP plot may help an engineer debug a model, but it may not help an affected applicant understand how to challenge a rejection. XAI therefore acts as a bridge between technical model behavior, legal obligations, organizational accountability, and human empowerment.

---

## 4. The Regulatory Landscape: From Data Privacy to Product Safety

Governments are responding to algorithmic opacity by moving from data-protection rules toward broader AI governance and product-safety frameworks.

### 4.1 The GDPR and the "Right to Explanation" Debate

The EU General Data Protection Regulation (GDPR) was adopted in 2016 and became applicable in May 2018 ([European Parliament and Council, 2016](#ref-gdpr)). Article 22 gives data subjects the right not to be subject to decisions based solely on automated processing, including profiling, when those decisions produce legal effects or similarly significant effects.

Goodman and Flaxman (2017) argued that the GDPR could effectively create a form of **"right to explanation"** for algorithmic decisions ([Goodman & Flaxman, 2017](#ref-goodman)). However, this interpretation remains contested.

- **The "solely automated" limitation:** Article 22 focuses on decisions based solely on automated processing. This creates difficult questions when a human reviewer is added to the process. A meaningful human review may remove the decision from Article 22's strict scope, but a superficial rubber-stamp review should not be treated as genuine human oversight.
- **The explanation vs. redress problem:** Affected persons often do not only want a technical explanation of a model. They usually want a way to correct inaccurate data, challenge the decision, receive human review, and repair the harm caused by the decision.
- **The technical-legal gap:** GDPR-style transparency rights do not automatically tell engineers which XAI method to implement, how detailed the explanation must be, or how the explanation should be adapted for different users.

The GDPR therefore matters because it introduced strong data rights and automated decision-making safeguards, but it did not fully solve the practical governance problem of explainable AI.

### 4.2 The EU AI Act (2024)

The EU AI Act, Regulation (EU) 2024/1689, represents a broader shift from data protection toward AI product-safety regulation ([European Parliament and Council, 2024](#ref-european)). It classifies AI systems according to risk categories, including unacceptable risk, high risk, limited risk, and minimal risk.

For high-risk systems, such as some systems used in credit scoring, education, employment, law enforcement, migration, and access to essential services, the Act introduces several governance obligations:

- **Transparency and instructions for use, Article 13:** High-risk AI systems must be designed and documented so that deployers can understand their output and use the system appropriately.
- **Human oversight, Article 14:** High-risk AI systems must allow effective human oversight, including measures to reduce automation bias and enable intervention where necessary.
- **Right to explanation, Article 86:** Affected persons have the right to obtain clear and meaningful explanations where a deployer makes a decision based on the output of certain high-risk AI systems listed in Annex III, and the decision produces legal or similarly significant adverse effects on health, safety, or fundamental rights. This right is subject to the Act's scope and limitations, and it becomes relevant within the Act's phased application timeline.

The AI Act also creates a governance regime for **General-Purpose AI (GPAI)**. Under Article 51, GPAI models are presumed to have high-impact capabilities when the cumulative compute used for training is greater than **10²⁵ floating-point operations (FLOPs)**. Such models may face additional systemic-risk evaluation and mitigation duties.

The significance of the AI Act is that explainability is no longer only a research preference or a design ideal. In high-risk contexts, it becomes part of a broader compliance architecture that includes documentation, risk management, human oversight, incident reporting, and post-market monitoring.

### 4.3 Global Trends: Is Governance Working?

The **Artificial Intelligence Index Report 2025** reported a significant short-term improvement in foundation-model transparency: the Foundation Model Transparency Index average score increased from **37/100 in 2023** to **58/100 in 2024**, partly because developers disclosed previously nonpublic information about labor, data, and compute ([Stanford Institute for Human-Centered Artificial Intelligence, 2025](#ref-stanford)).

However, this should not be framed as a stable long-term trend. Later FMTI work reported a lower average score of **40.69/100 in 2025**, indicating that transparency progress remains uneven and fragile ([Wan et al., 2025](#ref-wan)). A careful conclusion is therefore that AI governance is improving in some areas, but transparency remains inconsistent across companies, business models, and disclosure categories.

---

## 5. Theoretical Deep Dive: Intuition vs. Normative Defensibility

Will technical transparency actually make AI fair? Selbst and Barocas (2018) argue that when we call an algorithm a "black box," we often conflate two related but distinct problems: **inscrutability** and **nonintuitiveness** ([Selbst & Barocas, 2018](#ref-selbst)).

1. **Inscrutability:** The model is too complex for humans to inspect directly. For example, a deep neural network may contain millions or billions of parameters, making its internal logic difficult to trace.
2. **Nonintuitiveness:** The model may identify statistical relationships that are real or predictive but do not align with human intuition. Even if the mathematical rule is revealed, a human may still find it difficult to evaluate normatively.

This distinction matters because many legal and technical approaches assume that transparency will solve fairness problems. But explaining the final model may not be enough. If a model learns a nonintuitive pattern from biased, incomplete, or historically unequal data, the explanation may reveal the pattern without justifying it.

**The ethical dilemma:** Humans often use intuition to evaluate whether a decision is fair. Machine learning weakens this shortcut because some predictive relationships are not intuitive, and some intuitive relationships may still be unfair. Therefore, to show that an AI decision is **normatively defensible**, organizations must explain more than the final prediction. They must also document the process that produced the model, including data collection, feature selection, training choices, validation procedures, fairness analysis, human oversight, and appeal mechanisms.

---

## 6. Ecosystem Governance: The SCOR Framework

Legal compliance by a single organization is often insufficient in a complex AI ecosystem involving developers, vendors, deployers, regulators, affected communities, and third-party auditors. To address this broader governance challenge, Torkestani and Mansouri (2025) propose the **SCOR Framework** as an emerging model for responsible AI innovation in digital ecosystems ([Torkestani & Mansouri, 2025](#ref-torkestani)).

SCOR should be treated as a recent proposed framework rather than a settled industry standard. Its value is that it shifts attention from one-time technical explanation toward continuous ecosystem governance:

- **S, Shared Ethical Charter:** Participants agree on binding ethical commitments, such as fairness, accountability, safety, and transparency, before deployment.
- **C, Co-Design Mechanisms:** End-users and affected communities are included early in the design process to reduce corporate capture and improve contextual fit.
- **O, Oversight and Learning:** Independent audits, logs, monitoring, feedback channels, and incident-review procedures are used after deployment.
- **R, Regulatory Alignment:** Governance adapts to changing legal requirements through tools such as regulatory sandboxes, compliance reviews, and risk-based documentation.

In relation to the GDPR and EU AI Act, SCOR can be understood as a practical governance layer. The law defines many of the obligations; ecosystem governance helps organizations operationalize those obligations across the actors who design, deploy, monitor, and contest AI systems.

---

## 7. Practical Implementation: Bridging the "Last Mile"

How can engineers deliver explanations to non-technical users in a way that satisfies explanatory pragmatism and supports legal governance?

Recent work in the _Cambridge Journal of Artificial Intelligence_ argues for **Natural Language Explanations (NLEs)** delivered through dialogue systems as a way to address the "last mile" of explainability ([Nicolis & Kingsman, 2024](#ref-nicolis)). Rather than presenting a rejected applicant with a raw [SHAP](https://github.com/shap/shap) plot, engineers can convert feature-contribution evidence into a short, plain-language explanation.

However, a critical design rule must be followed: the language model must not become the source of truth. The model explanation should come from the verified decision evidence, such as feature values, SHAP contributions, policy rules, and audit logs. The language model should only rewrite the verified facts into clearer language, and its output should be validated or replaced by a deterministic template if it is incomplete or misleading.

### Code Snippet: Generating Governance-Aware NLEs in Python

The following example demonstrates a safer pattern. It uses a synthetic credit-like dataset for teaching purposes, trains an XGBoost classifier, computes SHAP contributions, builds a controlled explanation from verified facts, and then optionally asks an instruction-tuned model to rewrite the explanation. If the language-model output fails basic validation, the system falls back to a deterministic template.

This is an educational prototype, not a legally compliant lending system. A real financial system would require validated data, lawful feature selection, domain-specific compliance review, model-risk management, audit logging, human oversight, and an appeal process.

```python
import warnings
import re
import numpy as np
import pandas as pd
import shap
import torch
import xgboost

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")

# =========================================================
# 1. Create a synthetic credit-like dataset for demonstration
# =========================================================

rng = np.random.default_rng(42)
n = 1500

X = pd.DataFrame({
    "annual_income": rng.normal(55000, 18000, n).clip(15000, 150000),
    "debt_to_income_ratio": rng.uniform(0.05, 0.75, n),
    "credit_history_years": rng.uniform(0, 25, n),
    "missed_payments_12m": rng.poisson(1.0, n).clip(0, 8),
    "savings_balance": rng.normal(7000, 6000, n).clip(0, 50000),
    "loan_amount": rng.normal(30000, 12000, n).clip(5000, 100000),
})

# Synthetic approval rule with noise. Class 1 means approved.
approval_score = (
    0.000035 * X["annual_income"]
    - 3.4 * X["debt_to_income_ratio"]
    + 0.10 * X["credit_history_years"]
    - 0.55 * X["missed_payments_12m"]
    + 0.00008 * X["savings_balance"]
    - 0.000018 * X["loan_amount"]
    + rng.normal(0, 0.35, n)
)

y = (approval_score > 0.8).astype(int)

xgb_model = xgboost.XGBClassifier(
    random_state=42,
    n_estimators=120,
    max_depth=3,
    learning_rate=0.06,
    eval_metric="logloss"
)

xgb_model.fit(X, y)

# A hypothetical applicant. These values are intentionally chosen
# to produce a likely rejection in this educational example.
applicant = pd.DataFrame([{
    "annual_income": 38000,
    "debt_to_income_ratio": 0.56,
    "credit_history_years": 1.2,
    "missed_payments_12m": 3,
    "savings_balance": 900,
    "loan_amount": 35000,
}])

approval_probability = xgb_model.predict_proba(applicant)[0][1]
decision = "approved" if approval_probability >= 0.5 else "denied"

# =========================================================
# 2. Generate SHAP contributions
# =========================================================

explainer = shap.Explainer(xgb_model, X)
shap_values = explainer(applicant)

feature_impacts = []

for feature, contribution in zip(applicant.columns, shap_values.values[0]):
    feature_impacts.append({
        "feature": feature,
        "value": applicant.iloc[0][feature],
        "contribution": float(contribution)
    })

# Since class 1 means approval, negative SHAP values lowered approval.
negative_reasons = sorted(feature_impacts, key=lambda item: item["contribution"])
top_reasons = negative_reasons[:3]

# =========================================================
# 3. Convert SHAP facts into controlled natural-language facts
# =========================================================

def readable_reason(reason):
    feature = reason["feature"]
    value = reason["value"]

    if feature == "annual_income":
        return f"the stated annual income (${value:,.0f})"
    if feature == "debt_to_income_ratio":
        return f"the debt-to-income ratio ({value:.0%})"
    if feature == "credit_history_years":
        return f"the length of credit history ({value:.1f} years)"
    if feature == "missed_payments_12m":
        return f"recent missed payments ({int(value)} in the last 12 months)"
    if feature == "savings_balance":
        return f"the available savings balance (${value:,.0f})"
    if feature == "loan_amount":
        return f"the requested loan amount (${value:,.0f})"

    return feature.replace("_", " ")

reason_texts = [readable_reason(reason) for reason in top_reasons]

controlled_facts = (
    f"Decision: {decision}.\n"
    f"Estimated approval probability: {approval_probability:.1%}.\n"
    f"Main factors lowering the approval score: {', '.join(reason_texts)}.\n"
    "Required user rights message: The applicant may request human review or provide corrected information."
)

def deterministic_user_message(decision, probability, reasons):
    if decision == "approved":
        return (
            "Thank you for your application. Based on the information assessed by the system, "
            f"your application was approved with an estimated approval probability of {probability:.1%}. "
            "You may still request clarification if any information appears incorrect."
        )

    return (
        "Thank you for your application. Based on the information assessed by the system, "
        f"we cannot approve it at this stage. The estimated approval probability was {probability:.1%}. "
        f"The main factors that lowered the score were {', '.join(reasons)}. "
        "You may request a human review or provide corrected information if you believe the data is incomplete or inaccurate."
    )

fallback_message = deterministic_user_message(decision, approval_probability, reason_texts)

# =========================================================
# 4. Optional LLM rewriting, with validation and fallback
# =========================================================

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = f"""
Rewrite the verified facts below into a short, polite explanation for a non-technical loan applicant.

Rules:
1. Do not add new reasons.
2. Do not remove the right to request human review.
3. Do not mention SHAP or technical model internals.
4. Keep the explanation clear and factual.
5. If the facts say the application was denied, explain the main factors.

Verified facts:
{controlled_facts}

Final explanation:
"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=140,
        num_beams=4,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

candidate_message = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def is_usable_message(message):
    message_lower = message.lower()

    if len(message.split()) < 30:
        return False
    if decision == "denied" and "human review" not in message_lower:
        return False
    if decision == "denied" and not any(reason.split("(")[0].strip().lower() in message_lower for reason in reason_texts):
        return False
    if re.search(r"approved.*denied|denied.*approved", message_lower):
        return False

    return True

final_message = candidate_message if is_usable_message(candidate_message) else fallback_message

print("Controlled facts:\n", controlled_facts)
print("\nFinal user-facing explanation:\n", final_message)
```

**Actual Output:**

> **Controlled facts:**
>
> Decision: denied.
>
> Estimated approval probability: 0.4%.
>
> Main factors lowering the approval score: the length of credit history (1.2 years), recent missed payments (3 in the last 12 months), the debt-to-income ratio (56%).
>
> Required user rights message: The applicant may request human review or provide corrected information.
>
> **Final user-facing explanation:**
>
> Thank you for your application. Based on the information assessed by the system, we cannot approve it at this stage. The estimated approval probability was 0.4%. The main factors that lowered the score were the length of credit history (1.2 years), recent missed payments (3 in the last 12 months), the debt-to-income ratio (56%). You may request a human review or provide corrected information if you believe the data is incomplete or inaccurate.

**Implementation lesson:** The safest XAI architecture is not "model plus LLM equals explanation." A safer architecture is: verified model evidence, controlled explanation facts, optional language-model rewriting, validation, fallback template, audit logging, and human review.

Recommended Python libraries for XAI implementation include [`shap`](https://github.com/shap/shap), [`lime`](https://github.com/marcotcr/lime), [`interpret-community`](https://github.com/interpretml/interpret-community), and [`DALEX`](https://github.com/ModelOriented/DALEX).

---

## 8. Reflective Exercises and Discussion

To solidify your understanding of these governance mechanisms, consider the following prompts before moving on to Chapter 3.

> **Discussion Prompt 1: The Trade Secrets Tension**
>
> The EU AI Act requires technical documentation and transparency obligations for high-risk AI systems. How should companies balance transparency duties with the protection of proprietary intellectual property and trade secrets?

> **Critical Thinking Exercise 2: Measuring Transparency**
>
> Choose an AI system you use regularly, such as a chatbot, recommendation algorithm, translation tool, or fraud-detection system. Evaluate it using the simplified IEEE transparency levels discussed in Section 2.2. Which level does it currently achieve, and what engineering steps would be required to reach Level 4?

> **Critical Thinking Exercise 3: Explanation vs. Redress**
>
> Imagine a student is rejected by an automated scholarship-screening system. What would be more useful: a technical explanation of the model, a list of the data used, a counterfactual explanation, a human appeal process, or a combination of these? Justify your answer.

---

## 9. Conclusion

The laws reviewed in this chapter, from the GDPR to the EU AI Act, attempt to build legal guardrails around algorithmic decision-making. However, compliance is only the beginning.

Explanatory pragmatism shows that an explanation must be useful to a specific audience. Stakeholder analysis shows that developers, deployers, end-users, affected persons, and regulators need different forms of transparency. The GDPR and EU AI Act show that explainability is becoming part of formal governance. Selbst and Barocas show that transparency alone cannot solve the deeper problem of normative justification. SCOR and similar ecosystem-governance models show that responsible AI requires continuous oversight across multiple actors.

Explainable AI is therefore not merely a technical feature that opens black boxes for engineers. It is a governance mechanism that can support accountability, contestability, and fairer distribution of power. When implemented responsibly, XAI can help bridge the gap between complex mathematical systems and human decision-making, but only if explanations are grounded in verified evidence, connected to human oversight, and paired with meaningful redress.

In Chapter 3, we will examine how these regulated systems are deployed in specific industry verticals and how organizations can maintain continuous compliance in production.

---

## 10. AI Transparency Statement

In accordance with the **CSEN1152** (_Seminar of XAI: Concepts, Applications and Future Directions_) syllabus guidelines at the **German University in Cairo (GUC)**, Spring 2026, I acknowledge that an AI language model was used during the preparation of this chapter. Specifically, the LLM was used for draft refinement, structural organization, pedagogical scaffolding, code-example improvement, and citation-quality review. All AI-assisted outputs were critically evaluated, independently checked against the cited academic and legal sources, and synthesized to reflect my own original understanding and analysis of the topic.

---

## References

<a id="ref-wan"></a>Wan, A., Klyman, K., Kapoor, S., Maslej, N., Longpre, S., Xiong, B., Liang, P., & Bommasani, R. (2025). _The 2025 Foundation Model Transparency Index_. arXiv:2512.10169. https://arxiv.org/abs/2512.10169

<a id="ref-gdpr"></a>European Parliament and Council of the European Union. (2016). Regulation (EU) 2016/679 of 27 April 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC, General Data Protection Regulation. _Official Journal of the European Union_, L 119, 1-88.

<a id="ref-european"></a>European Parliament and Council of the European Union. (2024). Regulation (EU) 2024/1689 of 13 June 2024 laying down harmonised rules on artificial intelligence, Artificial Intelligence Act. _Official Journal of the European Union_, L 2024/1689.

<a id="ref-goodman"></a>Goodman, B., & Flaxman, S. (2017). European Union regulations on algorithmic decision-making and a "right to explanation." _AI Magazine, 38_(3), 50-57. https://doi.org/10.1609/aimag.v38i3.2741

<a id="ref-ieee7001"></a>IEEE Standards Association. (2021). _IEEE Standard for Transparency of Autonomous Systems_ (IEEE Std 7001-2021). IEEE. https://doi.org/10.1109/IEEESTD.2022.9726144

<a id="ref-langer"></a>Langer, M., Oster, D., Speith, T., Hermanns, H., Kästner, L., Schmidt, E., Sesing, A., & Baum, K. (2021). What do we want from explainable artificial intelligence (XAI)? A stakeholder perspective on XAI and a conceptual model guiding interdisciplinary XAI research. _Artificial Intelligence, 296_, 103473. https://doi.org/10.1016/j.artint.2021.103473

<a id="ref-miller"></a>Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. _Artificial Intelligence, 267_, 1-38. https://doi.org/10.1016/j.artint.2018.07.007

<a id="ref-nicolis"></a>Nicolis, A., & Kingsman, N. (2024). AI explainability in the EU AI Act: A case for an NLE approach towards pragmatic explanations. _Cambridge Journal of Artificial Intelligence, 1_(1), 3-16.

<a id="ref-selbst"></a>Selbst, A. D., & Barocas, S. (2018). The intuitive appeal of explainable machines. _Fordham Law Review, 87_(3), 1085-1139.

<a id="ref-stanford"></a>Stanford Institute for Human-Centered Artificial Intelligence. (2025). _Artificial Intelligence Index Report 2025_. Stanford University.

<a id="ref-torkestani"></a>Torkestani, M. S., & Mansouri, T. (2025). SCOR: A framework for responsible AI innovation in digital ecosystems. _Proceedings of the British Academy of Management Conference 2025, University of Kent, UK_. arXiv:2509.10653. https://doi.org/10.48550/arXiv.2509.10653

<a id="ref-winfield"></a>Winfield, A. F. T., Booth, S., Dennis, L. A., Egawa, T., Hastie, H., Jacobs, N., Muttram, R. I., Olszewska, J. I., Rajabiyazdi, F., Theodorou, A., Underwood, M. A., Wortham, R. H., & Watson, E. (2021). IEEE P7001: A proposed standard on transparency. _Frontiers in Robotics and AI, 8_, 665729. https://doi.org/10.3389/frobt.2021.665729

---

To cite this chapter, please use the following BibTeX:

```bibtex
@misc{emad_2026_XAI,
  author       = {Omar Emad},
  title        = {Interpreting Machine Learning: A Gentle Introduction, Chapter 2},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/interpreting_machine_learning}}
}
```
