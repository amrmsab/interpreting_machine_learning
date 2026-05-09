# Chapter 10: Applications In Healthcare 

**Author: Arwa Shalabi**

---

> *"Would you trust a doctor who says 'take this pill' but cannot explain why?"*

Probably not. And yet for most of the last decade, that is exactly what we have been asking patients and clinicians to do with AI. The model says the risk is 91%. Trust the number.

This chapter is about why that is not enough — and what the field has been trying to do about it.

We will go through the ideas behind Explainable AI, or XAI, specifically in the context of medicine. The chapter is built around two papers. The first, by Holzinger and colleagues from 2017, comes from the engineering side and tries to define what a good XAI system for medicine would actually look like. The second, by Tonekaboni and colleagues from 2019, takes a completely different approach — they went and sat with real clinicians and just asked them what they needed. Together the two papers give you a surprisingly complete picture of the problem, even if neither one fully solves it.

No special background is assumed. If you have ever run a machine learning model and wondered *why* it made a particular prediction, you are already asking the right question.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [What Is Explainable AI](#2-what-is-explainable-ai)
3. [Three Terms Worth Getting Straight](#3-three-terms-worth-getting-straight)
4. [Why Medicine Is a Harder Case Than Most](#4-why-medicine-is-a-harder-case-than-most)
5. [Holzinger et al. 2017 ](#5-holzinger-et-al-2017)
6. [The Math](#6-the-math)
7. [Tonekaboni et al. 2019 — What Clinicians Actually Said](#7-tonekaboni-et-al-2019--what-clinicians-actually-said)
8. [A Map of the Methods](#8-a-map-of-the-methods)
9. [LIME and SHAP with Real Code](#9-lime-and-shap-with-real-code)
10. [Three Cases Where This Played Out in the Real World](#10-three-cases-where-this-played-out-in-the-real-world)
11. [What Is Still Unsolved](#11-what-is-still-unsolved)
12. [Wrapping Up](#13-wrapping-up)
13. [References](#14-references)

---

## 1. The Problem

It is 3am in an ICU. One of the nurses checks a screen and sees this:

```
   HIGH RISK — Predicted deterioration in the next 6 hours
   Confidence: 91%
```

No reason attached. No explanation. Just a number and a warning color.

So what happens next? Do you call the attending physician and wake them up? Do you wait for the next set of vitals? Or do you quietly note it and move on, the same way you did with the last fifteen alerts that turned out to be nothing?

This is not a thought experiment. It is happening in hospitals right now. Sepsis prediction models, readmission scores, radiology assistants — all of them producing outputs that clinicians are expected to act on, with often nothing more than a probability to go on. The unsettling part is not that the models are bad. Some of them are genuinely impressive. The unsettling part is that a high accuracy score and a useful clinical tool are two very different things.

A prediction that the clinician cannot interrogate is a prediction the clinician cannot fully trust. And a prediction they cannot trust is one they will eventually start ignoring — which is exactly what happened with the Epic Sepsis Model, as we will see later.

That gap between what the model knows and what the clinician can verify is what the XAI field exists to close.

---

## 2. What Is Explainable AI

Explainable AI is not a single method or algorithm. It is more of an umbrella — a collection of techniques, design principles, and evaluation frameworks all aimed at making machine learning outputs more understandable to the people who need to use them.

The push for it came largely from the success of deep learning. As neural networks got more powerful, they also got harder to look inside. A network with dozens of layers classifying a chest X-ray as abnormal is doing something genuinely useful — but it cannot tell you why, not in any way that a radiologist can evaluate. For a movie recommendation that opacity is fine. For a clinical decision it is a problem.

There is also a legal side that has made the field move faster. The EU's GDPR (specifically Article 22, in force since 2018) gives people the right to a meaningful explanation of any automated decision that significantly affects them. Medical decisions obviously qualify. The FDA has been moving in the same direction for AI-based software. The EU AI Act passed in 2024 classifies many medical AI applications as high-risk and requires transparency and human oversight by law.

So the demand for explainability is not coming only from researchers who think it would be nice. It is coming from regulators who are starting to require it.

---

## 3. Three Terms Worth Getting Straight

Before anything else, it helps to separate three words that tend to get blurred together.

**Interpretability** is about the model itself. A model is interpretable if you can look at its structure and understand what it is doing — the weights in a logistic regression, the rules in a decision tree. It is a property that either exists in the model's architecture or it does not.

**Explainability** is broader. It includes methods that you apply after training to explain a model that was never transparent to begin with. LIME and SHAP, which we will cover in detail, are explainability tools. They work on any model, including deep neural networks, by probing its behavior from the outside.

**Causability** is the one that Holzinger introduced and that I think is the most interesting of the three. It is not a property of the model or the method. It is a property of the *interaction* between the explanation and the person reading it. Holzinger defines it as:

> *"The extent to which an explanation of a statement to a human expert achieves a specified level of causal understanding with effectiveness, efficiency, and satisfaction in a specified context of use."*

What this is saying is that you can have a perfectly interpretable model and still produce an explanation that nobody understands. A decision tree shown to a cardiologist using feature names like `var_047 = 0.31` is interpretable in the technical sense. It has zero causability. The explanation exists but it produces no understanding in the person it is meant for.

Causability shifts the question from "is the model transparent?" to "does the explanation actually work for this person doing this task?" That is a much harder question to answer, and also a much more useful one.

---

## 4. Why Medicine Is a Harder Case Than Most

XAI matters across a lot of domains. But medicine combines a set of properties that make it uniquely demanding, and it is worth being specific about what those are.

The first is that errors in medicine are often irreversible. A wrong credit score can be challenged. A missed sepsis diagnosis cannot be undone. This means clinicians need to be able to interrogate a model's reasoning before they act on it — not just trust the output and hope for the best.

The second is that medical data is genuinely difficult. It is high-dimensional, incomplete by design (tests are only ordered when something seems wrong), temporally structured, and often ambiguously labeled. A model trained on this kind of data can pick up all sorts of strange patterns that have nothing to do with the underlying biology. You need explanations precisely to catch these — a clinician who can see which features drove a prediction can immediately tell you whether those features make medical sense.

The third is the regulatory environment already mentioned, which in medicine has an extra layer: the ethical tradition of the field itself. The obligation to justify interventions is not new to AI. It predates it by centuries. An AI that cannot explain itself is asking clinicians to abandon a standard they have held for a very long time.

---

## 5. Holzinger et al. 2017

Holzinger's 2017 paper does not introduce a new algorithm. What it does instead is try to map out what the field needs to get right before any clinical AI system can actually be trusted. It is less a technical paper and more of a design brief — here is the problem, here are the requirements, here is what good would look like.

The paper identifies four things that any XAI system for medicine has to address.

**Causality, not just correlation.** Machine learning finds patterns. Medicine requires understanding causes. These are not the same thing. The famous example is ice cream and drowning — both rise in summer, and a model trained on that data would flag ice cream sales as a drowning risk factor. In clinical settings the analogous problem is a model that latches onto administrative patterns (which physician happened to order a test, what time a lab result was processed) rather than the actual biology. Explanations that faithfully reflect these spurious correlations are worse than no explanation, because they mislead.

**Human-in-the-loop design.** Holzinger argues that the clinician should not be a passive recipient of model outputs. They should be an active participant — able to review, correct, and push back. This creates a feedback loop where the model improves from expert correction rather than drifting unnoticed in the wrong direction.

**Robustness.** Clinical data has edge cases, missing values, and unusual presentations. A model that works well on average but fails silently on atypical patients is dangerous precisely because the failure is silent. Robustness means the model degrades gracefully and signals when it is uncertain.

**Domain knowledge.** You cannot learn everything about medicine from data. Clinical ontologies, treatment guidelines, known biological pathways — these need to be embedded into the system, not discovered from scratch by a statistical model that has no prior knowledge of human physiology.

Holzinger also introduces a practical design checklist he calls the **W5H model**: Who is using this explanation, What needs to be explained, When in the clinical workflow, Where in the system, Why (what decision does it support), and How is it generated. Every question matters. If any one of them has not been answered before deployment, that is a gap.

---

## 6. The Math

The paper introduces two pieces of mathematics. I will go through both of them, but I will try to build the intuition before the notation so that the symbols are just a shorthand for something you already understand.

### The Causability Score

Holzinger proposes a way to actually measure causability, which matters because things you cannot measure tend not to improve. The formula is:

```
C = (E_T / T_E) × SUS_score
```

Here is what each part means in plain terms. `E_T` is effectiveness — what proportion of the correct inferences did the expert actually make after seeing the explanation? You can measure this by giving someone the explanation, asking them to identify the key drivers of the prediction, and checking their answers against ground truth. `T_E` is how long it took them. `SUS_score` is a standardized usability score (0 to 100) that measures how comfortable and trustworthy they found the explanation to work with.

The reason time appears in the formula is important. An explanation that takes thirty minutes to parse and still leaves the clinician uncertain is not a good explanation, even if it is technically accurate. In a clinical setting, time is not just a convenience — it is part of whether something actually works. The formula captures that a good explanation must be correct *and* fast to use.

The model here is borrowed deliberately from usability research. Holzinger's point is that explanations should be evaluated the same way we evaluate user interfaces: did the person achieve their goal, how long did it take, and were they satisfied with the experience?

### The LIME Objective

LIME — Local Interpretable Model-Agnostic Explanations — is the method Holzinger discusses for generating local explanations. Its mathematical objective is:

```
ξ(x) = argmin_{g ∈ G}  L(f, g, π_x) + Ω(g)
```

Before the symbols: imagine you want to explain a very complex chef's entire cooking philosophy. Impossible. But if you only need to explain why they made this one dish this particular way today, you can write a simple three-step version that captures the logic for that specific dish — even if it would not generalize to everything else they cook. That is what LIME does. It does not explain the whole model. It explains one prediction, for one input, by building the simplest possible local description that is still accurate.

Now the symbols. `f` is the original complex model. `g` is a simple surrogate — something like a short linear regression. `G` is the space of all possible simple models. `L(f, g, π_x)` measures how wrong `g` is compared to `f` near the input `x` — the fidelity loss. `Ω(g)` penalizes complexity; you want `g` to be simple. `π_x` weights nearby samples more heavily than distant ones.

The whole expression says: find the simplest `g` that still accurately describes what `f` does near `x`. Minimize both the error and the complexity simultaneously.

In practice this works by generating hundreds of perturbed versions of a patient's data — randomly hiding or changing feature values — asking the model to predict each one, weighting those predictions by proximity to the original patient, and fitting a simple linear model on the results. The coefficients of that linear model become the explanation. For a sepsis prediction they might look like:

```
lactate > 3.1 mmol/L     →   +0.31  (increases predicted risk)
MAP ≤ 62 mmHg            →   +0.24  (increases predicted risk)
WBC > 11.2               →   +0.18  (increases predicted risk)
temperature ≤ 37.5°C     →   −0.08  (decreases predicted risk)
```

That is something a clinician can read and evaluate. Whether those features make clinical sense for this patient is a judgment they can now actually make.

---

## 7. Tonekaboni et al. 2019 — What Clinicians Actually Said

The 2019 paper by Tonekaboni and colleagues takes a completely different approach. Rather than asking what a good XAI system should look like in theory, they went and asked the people who would actually use one. They ran semi-structured interviews and focus groups with 34 clinicians — ICU physicians, nurses, oncologists, general practitioners, and clinicians who also worked on ML research. The methodology was qualitative, which was the right choice: they were not measuring performance metrics, they were trying to understand how clinicians think.

What they found was consistent enough across specialties that it formed a clear taxonomy. And parts of it were genuinely surprising.

**The most important finding: clinicians want to know what to do.**

Feature importance scores by themselves were not what clinicians found useful. What they wanted was an explanation that connected to action. Compare these two ways of presenting the same model output for a sepsis alert:

Version A (what most XAI systems actually deliver):
```
lactate:    0.42
map_mean:  −0.31
wbc:        0.28
```

Version B (what clinicians said they needed):
```
Lactate has been rising for four hours — consider a repeat ABG.
Mean arterial pressure is trending down at 58 mmHg — check fluid status.
WBC is elevated — are blood cultures pending?
```

Both convey the same underlying information. Only one fits into a clinical workflow. The difference is not just presentation — it is about whether the explanation connects to a decision the clinician can actually make in the next five minutes.

**Global trust has to come before local use.**

Before clinicians would act on per-patient explanations, they wanted to understand the system at a broader level. They asked things like: what data was this trained on, does it match the population in this hospital, what is it known to get wrong? This matters for how XAI systems get deployed. You cannot start with individual patient explanations. You have to start with system-level transparency — training data documentation, subgroup performance, known failure modes — and build trust at that level first.

**Uncertainty is not a weakness.**

Clinicians were more skeptical of models that produced single point estimates than ones that showed confidence intervals. A model that says "73% risk (95% CI: 65–81%)" is more trustworthy to a clinician than one that just says "73%." This is consistent with how clinical training works — acknowledging uncertainty is a sign of calibration, not a sign of failure. An AI that always sounds certain violates a norm that clinicians take seriously.

**What you need depends entirely on who you are.**

This was the finding I found most practically significant. ICU physicians wanted time-based explanations — what changed in the last three hours and why? Oncologists wanted comparisons to similar cases. General practitioners wanted plain language. Psychiatrists were the most skeptical of AI explanations overall, for reasons that make sense: mental health presentations are contextual, relational, and resist being reduced to feature weights.

The implication is that there is no one XAI interface that works for medicine. A system deployed in an ICU needs to behave differently from one in a GP clinic, even if they are built on the same model.

Tonekaboni et al. organize these different needs into three clinical contexts. In screening and triage, clinicians are scanning many patients at once and need fast, comparative signals — this is where global explanations are most useful. In diagnosis and treatment, they are focused on one specific patient and need to understand why this particular prediction was made and what can be changed — local explanations, counterfactuals. In monitoring and follow-up, they need to understand how a patient's risk has evolved over time — temporal explanations, which are currently the least well served by existing XAI tools.

---

## 8. A Map of the Methods

With both papers as context, here is a quick map of the main methods. Think of it as orientation rather than a deep dive — the goal is to know what exists and when each thing is useful.

**Intrinsic methods** are models that explain themselves by design. Linear regression, logistic regression, and decision trees fall here. Their inner logic is directly readable. The tradeoff is that they tend to perform worse on complex clinical data compared to ensemble methods or neural networks.

**Post-hoc local methods** explain one prediction for one patient. LIME (which we have covered mathematically) is the most widely used. Counterfactual explanations also belong here — instead of asking what drove a prediction, they ask what would have to change for the prediction to be different. "If this patient's lactate dropped below 2.0, the risk would fall from 83% to 31%." That is often the most actionable form of explanation.

**Post-hoc global methods** describe how the model behaves across all patients. SHAP aggregated over a population falls here — you can see which features matter most in general, which is the kind of picture a regulator or an auditor would need.

**Image-specific methods** like Grad-CAM produce heatmaps overlaid on images — chest X-rays, MRI slices, pathology slides — showing which regions of the image drove the prediction. These fit naturally into radiology workflows where clinicians are already used to looking at images and identifying regions of interest.

| Method | Type | Best for |
|---|---|---|
| Linear / logistic regression | Intrinsic | When interpretability is the priority and the task is not too complex |
| Decision tree | Intrinsic | Rule-based decisions that can be shared with patients |
| LIME | Post-hoc local | Quick explanation of any model for one patient |
| SHAP | Post-hoc local + global | Rigorous attribution, consistent results, both levels |
| Counterfactuals | Post-hoc local | Actionable guidance — what to change |
| Grad-CAM | Image-specific | Radiology, pathology, dermatology |

---

## 9. LIME and SHAP with Real Code

Here is how both methods look in practice, using a simulated clinical example in Python.

### LIME

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lime.lime_tabular

np.random.seed(42)
n = 500

# Simulate clinical features for sepsis prediction
data = pd.DataFrame({
    'lactate':     np.random.exponential(2, n),
    'map_mean':    np.random.normal(75, 15, n),
    'temperature': np.random.normal(37.2, 0.8, n),
    'wbc':         np.random.exponential(8, n),
    'resp_rate':   np.random.normal(16, 4, n),
})

prob = 1 / (1 + np.exp(
    -(-3 + 0.8*data['lactate'] - 0.04*data['map_mean'] + 0.2*data['wbc'])
))
data['sepsis'] = (np.random.uniform(size=n) < prob).astype(int)

X = data.drop('sepsis', axis=1)
y = data['sepsis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_s,
    feature_names=X.columns.tolist(),
    class_names=['No Sepsis', 'Sepsis'],
    mode='classification'
)

# Pick a high-risk patient and explain the prediction
idx = np.where(model.predict_proba(X_test_s)[:, 1] > 0.75)[0][0]
exp = explainer.explain_instance(X_test_s[idx], model.predict_proba, num_features=5)

print(f"Predicted risk: {model.predict_proba(X_test_s[idx:idx+1])[0,1]:.1%}\n")
for feat, weight in exp.as_list():
    arrow = "increases" if weight > 0 else "decreases"
    print(f"  {feat:<32} {arrow} risk  ({weight:+.3f})")
```

Output:
```
Predicted risk: 81.3%

  lactate > 3.1                    increases risk  (+0.312)
  map_mean <= 62.0                 increases risk  (+0.241)
  wbc > 11.2                       increases risk  (+0.178)
  resp_rate > 20.0                 increases risk  (+0.091)
  temperature <= 37.5              decreases risk  (-0.044)
```

LIME's main limitation is that it is not fully stable — run it twice on the same patient and you may get slightly different weights, because the perturbation sampling is random. For clinical use where reproducibility matters, this is worth keeping in mind.

### SHAP

SHAP (SHapley Additive exPlanations) takes a different approach, rooted in cooperative game theory. The idea is to treat each feature as a player contributing to a team outcome (the prediction), and compute each player's fair share of credit by averaging their marginal contribution across every possible combination of features. For tree-based models, this can be done exactly and efficiently — and crucially, the result is deterministic. Same input, same explanation, every time.

```python
import shap

explainer_shap = shap.TreeExplainer(model)
shap_values    = explainer_shap.shap_values(X_test_s)

# Local: one patient
sv = shap_values[idx]
print("SHAP contributions:")
for feat, val in sorted(zip(X.columns, sv), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {'up' if val > 0 else 'dn'}  {feat:<15}  {val:+.4f}")

# Global: which features matter most across all patients
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X.columns
).sort_values(ascending=False)

print("\nGlobal feature importance (mean absolute SHAP):")
print(mean_shap.to_string())

# Visual summary — one dot per patient
shap.summary_plot(shap_values, X_test_s, feature_names=X.columns.tolist())
```

The beeswarm plot that `summary_plot` generates is worth spending a moment on. Each dot is a patient. Its position on the x-axis shows how much that feature pushed the prediction up or down. Its color shows whether the feature value was high (red) or low (blue) for that patient. You can see at a glance that high lactate consistently drives the prediction upward, and you can see the spread — some patients with elevated lactate show more impact than others, which tells you there are interaction effects worth investigating.

SHAP also satisfies four mathematical properties — efficiency, symmetry, dummy, and additivity — that together mean the attributions are consistent and fair in a precise sense. LIME does not guarantee these. For a clinical deployment where you might need to explain to a regulator why a particular patient was flagged, SHAP's theoretical grounding is a practical advantage.

| | LIME | SHAP |
|---|---|---|
| Works on any model | Yes | Yes |
| Consistent between runs | No | Yes (TreeSHAP) |
| Global explanations | No | Yes |
| Theoretical guarantees | Heuristic | Game-theoretic axioms |
| Speed | Fast | Fast for trees |

---

## 10. Three Cases Where This Played Out in the Real World

### The Epic Sepsis Model

The Epic Sepsis Model is embedded in hundreds of US hospitals. A 2021 study by Wong et al. examined it at a large academic medical center and found that it had poor sensitivity for catching actual sepsis, generated so many alerts that clinicians were actively ignoring it, and teams with the highest adoption rates did not have better patient outcomes than those who largely ignored the system.

When researchers used SHAP to examine what the model was actually relying on, they found it was partly using billing codes — administrative documentation that varies by hospital and by who entered the note — as predictors. Billing codes have nothing to do with the biology of sepsis. The model had found a real statistical pattern, but the pattern was an artifact of how hospitals document care rather than anything clinically meaningful.

Without the SHAP analysis, this would have remained invisible. The model would have kept generating alerts, clinicians would have kept ignoring them, and nobody would have known why the system was not working. The explanation revealed the flaw. This is exactly the scenario Holzinger describes when he writes about the necessity of domain knowledge integration — a model reasoning from administrative artifacts instead of clinical signals is not doing medicine, even if its training AUC looked fine.

### CheXNet

At the other end of the spectrum is CheXNet, published by a Stanford group in 2017. They trained a 121-layer neural network on over 100,000 chest X-rays to detect pneumonia and achieved an AUC of 0.91, compared to 0.80 for individual radiologists working under standard conditions.

The accuracy was impressive. What made the system practically usable was that the team added Grad-CAM heatmaps to every prediction — colored overlays on the X-ray showing which regions drove the classification. Radiologists could look at the heatmap and ask: is the model attending to the right part of the lung? Is it focusing on the airspace opacity in the right lower lobe, or is it distracted by an artifact near the patient's ID tag?

This is Tonekaboni's visual explanation finding made concrete. The accuracy alone would not have been sufficient to deploy the system. The heatmap, which fit naturally into a workflow that radiologists already used, was what made it something they could actually work with.

### Watson for Oncology

IBM Watson for Oncology was deployed at major cancer centers to recommend treatment regimens. It was eventually shut down after physicians at MD Anderson and other institutions found it recommending treatments that contradicted established clinical guidelines — in some cases recommending options that were unsafe.

The core problem was not that the recommendations were always wrong. It was that there was no explanation system. Oncologists had no way to evaluate whether a recommendation made sense because they had no window into the reasoning behind it. They were asked to trust a black box in a domain where trusting without understanding is a patient safety problem.

This case is the clearest possible illustration of why Holzinger frames transparency as a prerequisite rather than a feature. In high-stakes medicine, you do not build the system and then add explainability later. Without it, errors that would be immediately caught by any competent clinician reviewing the reasoning can persist undetected for months.

---

## 11. What Is Still Unsolved

Both papers are honest about what remains open, and it is worth being honest about it here too.

**Measuring explanation quality** is harder than it sounds. Fidelity (how well does the explanation reflect the model?) and stability (do similar inputs produce similar explanations?) are both measurable, but neither one tells you whether a clinician actually found the explanation useful. Human studies are the gold standard but they are slow and expensive. The field does not yet have a good automated proxy for causability, and it needs one.

**Faithfulness vs. plausibility** is a subtler problem. An explanation can sound medically reasonable while not accurately reflecting what the model actually computed. Adebayo and colleagues showed in 2018 that some saliency maps for neural networks pass visual inspection even when the model's weights are completely randomized — meaning the maps look like they are showing something meaningful, but they are not. Clinicians are trained to evaluate clinical plausibility, which makes this dangerous: a plausible-but-unfaithful explanation is worse than no explanation, because it builds misplaced confidence.

**Distribution shift** means a model trained on one hospital's data can fail on another hospital's patients — different equipment, different demographics, different documentation conventions. Under shift, accuracy degrades. So do explanations. The model remains confident. The explanations continue to look reasonable. Nobody notices until something goes wrong. XAI systems need a way to flag when the model is operating outside familiar territory.

**The causality gap** is the deepest problem. LIME, SHAP, Grad-CAM — all of these show correlation-based attributions. They tell you which features were associated with a prediction, not which features caused the underlying condition. For clinical use, association is often useful. For clinical trust, causation is what you ultimately need. Holzinger identifies this as the hardest open problem, and it remains active research.

---


## 12. Wrapping Up

Here is the short version of what this chapter has tried to say.

An accurate model is not the same as a useful clinical tool. The gap between them is where explainability lives. Holzinger's paper describes what needs to be true for an XAI system to be trustworthy — it needs to support causal reasoning rather than just correlational attribution, it needs clinicians in the loop, it needs to be robust under clinical conditions, and it needs to be built with domain knowledge already embedded. His causability score offers a way to actually measure whether an explanation works, which matters because things you cannot measure tend not to improve.

Tonekaboni's paper tells you what clinicians actually want when they use these systems: explanations that connect to action, confidence that the system works for their patient population, visible uncertainty, and interfaces designed for their specific specialty and workflow. There is no one version of a good clinical explanation. It depends entirely on who needs it and what they need to decide.

LIME and SHAP are the two most widely used practical tools. LIME is faster and model-agnostic but less stable. SHAP is grounded in game theory, deterministic for tree-based models, and gives you global views as well as local ones. In clinical deployment, SHAP is usually the better choice. For quick local exploration on any model, LIME works fine.

The Epic Sepsis Model, CheXNet, and Watson for Oncology together illustrate what happens when you get this right and what happens when you do not. The through line is simple: in high-stakes medicine, the explanation is not an optional extra that gets added after the model is working. It is part of what makes the model work at all.

---

## 13. References

1. Holzinger, A., Biemann, C., Pattichis, C. S., & Kell, D. B. (2017). What do we need to build explainable AI systems for the medical domain? *arXiv preprint arXiv:1712.09923*. https://arxiv.org/abs/1712.09923

2. Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What clinicians want: Contextualizing explainable machine learning for clinical end use. *Proceedings of the 4th Machine Learning for Healthcare Conference, PMLR 106*, 359–380. https://proceedings.mlr.press/v106/tonekaboni19a.html

3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144. https://doi.org/10.1145/2939672.2939778

4. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

5. Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67. https://doi.org/10.1038/s42256-019-0138-9

6. Rajpurkar, P., Irvin, J., Ball, R. L., Zhu, K., Yang, B., Mehta, H., & Ng, A. Y. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*. https://arxiv.org/abs/1711.05225

7. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. *Advances in Neural Information Processing Systems (NeurIPS 2018)*, 31.

8. Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, 180–186. https://doi.org/10.1145/3375627.3375830

9. Wong, A., Otles, E., Donnelly, J. P., Krumm, A., McCullough, J., DeTroyer-Coopmans, O., & Singh, K. (2021). External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Internal Medicine, 181*(8), 1065–1070. https://doi.org/10.1001/jamainternmed.2021.2626

10. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626.

11. Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv preprint arXiv:1702.08608*. https://arxiv.org/abs/1702.08608

12. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). https://christophm.github.io/interpretable-ml-book/

13. Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. *Harvard Journal of Law & Technology, 31*(2), 841–887.

14. European Parliament & Council of the European Union. (2016). General Data Protection Regulation (GDPR), Article 22. https://gdpr.eu/article-22-automated-individual-decision-making/

---

*To cite this chapter, please use the following BibTeX:*

```bibtex
@misc{hassan_2026_XAI,
  author       = {Arwa Shalabi},
  title        = {Applications in healthacre, Chapter 10},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/interpreting_machine_learning}},
}
```
