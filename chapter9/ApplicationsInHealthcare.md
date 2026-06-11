# XAI In Medicine 

*Based on Holzinger et al. (2017) and Tonekaboni et al. (2019)*

---

## 1. Introduction

The rapid proliferation of machine learning (ML) and artificial intelligence (AI) in healthcare has brought with it a fundamental tension: the most powerful predictive models are often the least interpretable. As these systems are increasingly considered for deployment in high-stakes clinical environments, a critical question emerges — not merely whether AI can perform well, but whether its decisions can be understood, trusted, and acted upon by those responsible for patient care.

This literature review examines two seminal works that approach this challenge from complementary angles. Holzinger et al. (2017) provide a foundational technical and philosophical framework for building explainable AI (XAI) systems in medicine, exploring what explainability means and why it matters from a systems-design perspective. Tonekaboni et al. (2019) ground the discussion in empirical reality, drawing on the expressed preferences of clinicians to define what explainability must look like to be useful in practice. Together, these works map the landscape of XAI in medicine — from the theoretical to the applied, and from the laboratory to the bedside.

---

## 2. Explainable AI as a Design Imperative: Holzinger et al. (2017)

### 2.1 Motivation and Problem Framing

Holzinger et al. (2017) open with a provocation: the most accurate models in machine learning — deep neural networks, ensemble methods, and other black-box architectures — are precisely those that resist human understanding. In a domain such as medicine, where misclassification carries serious consequences and clinical decision-making must be accountable and legally defensible, this opacity is not merely inconvenient; it is a barrier to adoption.

The authors argue that the medical domain places particularly stringent demands on AI systems. Unlike other application areas, medicine requires models that can be interrogated, audited, and explained to patients, clinicians, and regulatory bodies. This is not simply a matter of user preference — in many jurisdictions, algorithmic decisions affecting individuals are subject to legal requirements for transparency. The European Union's General Data Protection Regulation (GDPR), for instance, enshrines a limited 'right to explanation' that has direct implications for AI systems used in clinical contexts.

### 2.2 Defining Explainability and Interpretability

A significant contribution of Holzinger et al. (2017) is their careful disambiguation of key concepts. The authors distinguish between *interpretability* — the degree to which a human can understand the cause of a model's decision — and *explainability* — the extent to which the internal mechanics of a machine learning model can be represented in human-understandable terms. While these are often used interchangeably in the literature, the distinction is meaningful in medical contexts where accountability and auditability matter.

The authors further introduce the notion of *comprehensibility* as the ability of a model to represent its learned knowledge in a form that is understandable to humans — not merely accurate. This tripartite framework (interpretability, explainability, comprehensibility) provides a richer vocabulary for discussing what it means for an AI system to be transparent, and it anticipates later debates about whether post-hoc explanations are sufficient or whether inherently interpretable models are required.

### 2.3 Technical Approaches to XAI

Holzinger et al. survey a range of methods aimed at making ML models more explainable. These include rule extraction from neural networks, decision tree approximations of complex models, attention mechanisms in deep learning. The authors note that no single approach is universally suitable — the appropriate technique depends on the model type, the task, the domain, and the intended audience for the explanation.

Particularly relevant to the medical domain is the authors' discussion of *interactive machine learning* (iML), in which human experts are involved in the learning process itself rather than merely receiving outputs. This participatory model of AI development is argued to produce systems that are not only more accurate (benefiting from expert domain knowledge) but also more interpretable, because human cognitive structures are embedded in the model's reasoning process from the outset. This stands in contrast to post-hoc explanation methods, which attempt to impose interpretability on an already-trained black-box model.

### 2.4 Domain-Specific Challenges in Medicine

The authors devote considerable attention to features of the medical domain that complicate the design of XAI systems. Medical data is notoriously heterogeneous, combining structured records (lab values, vital signs), unstructured text (clinical notes), imaging data, and genomic information. It is also frequently incomplete, biased, and subject to significant inter-rater variability. Building models that are both accurate and explainable under these conditions is a substantially harder problem than it might appear in benchmark evaluations.

Furthermore, Holzinger et al. highlight the epistemological challenge: in medicine, a correct prediction is not always sufficient. Clinicians must understand *why* a prediction was made in order to integrate it with their broader clinical reasoning. An AI system that correctly identifies a malignancy but cannot indicate which features led to that conclusion may paradoxically be less useful than a less accurate but more transparent system, because the former cannot be reconciled with the clinician's own knowledge and experience.

---

## 3. Clinician-Centred Explainability: Tonekaboni et al. (2019)

### 3.1 Shifting the Frame: From Technical to Human

Where Holzinger et al. (2017) approach XAI primarily as a technical and philosophical challenge, Tonekaboni et al. (2019) take an empirical, human-centred approach. The authors conducted qualitative research with clinicians — physicians, nurses, and other healthcare professionals — to understand what they actually need from explainable AI systems. This shift in perspective is methodologically significant: rather than asking what AI *can* explain, the authors ask what clinicians *need* explained.

The results reveal a substantial gap between the kinds of explanations that XAI researchers tend to produce and the kinds of explanations that clinicians find useful. Feature importance scores, attention weights, and partial dependence plots — staples of the XAI toolkit — are often poorly aligned with clinical reasoning. Clinicians, the authors find, think in terms of patient trajectories, differential diagnoses, and clinical actions, not in terms of model features or statistical weights.

### 3.2 Contextualizing Explanation for Clinical Use

A central finding of Tonekaboni et al. is that the utility of an explanation is inseparable from the clinical context in which it is used. Different clinical roles, different points in the care pathway, and different types of decisions all generate different explanatory needs. An intensivist evaluating a sepsis alert in real time requires a fundamentally different kind of explanation than a radiologist reviewing AI-assisted image analysis at the end of a diagnostic workflow.

The authors propose a taxonomy of clinical use contexts that should inform XAI design. They distinguish between decision-support systems (where AI augments clinical judgment), monitoring systems (where AI flags deviations from expected patterns), and workflow automation systems (where AI operates with significant autonomy). Each category generates different demands on explainability: decision support requires that explanations be *actionable*; monitoring requires that they be *timely*; automation requires that they be *auditable*. This taxonomy provides a useful framework for aligning technical explanation methods with clinical requirements.

### 3.3 Trust, Appropriation, and Accountability

Tonekaboni et al. also engage with the affective and institutional dimensions of clinical AI adoption. Clinicians do not simply evaluate AI outputs; they form relationships with AI systems over time, developing calibrated trust based on accumulated experience. Explanations play a crucial role in this trust-formation process. When an AI system can articulate why it made a particular prediction, clinicians can assess whether its reasoning is consistent with their own knowledge, identify cases where it may be unreliable, and develop appropriate patterns of reliance and scepticism.

The authors note, however, that explanations can also distort this process. Over-reliance — accepting AI recommendations uncritically because they come with a convincing-sounding explanation — is as problematic as under-reliance. This concern echoes broader debates in human factors research about automation bias, and it underscores the importance of designing explanations that facilitate *calibration* rather than merely compliance.

Accountability is a related concern. In high-stakes clinical environments, responsibility for decisions must be clearly assigned. When an AI system contributes to a clinical decision that leads to a poor outcome, questions of legal and professional accountability become urgent. Explainability is part of the answer, but only if the explanations produced are meaningful enough to be evaluated by a competent professional — not just sufficiently plausible to satisfy a lay audience.

### 3.4 Design Recommendations

Based on their qualitative findings, Tonekaboni et al. offer a set of design recommendations for clinical XAI. These include the importance of aligning explanation granularity with decision complexity, providing contrastive explanations that clarify not just why a prediction was made but why alternative predictions were not, and integrating explanations into clinical workflows in ways that do not impose undue cognitive burden. The authors also advocate for iterative, participatory design processes that involve clinicians from the earliest stages of system development — an approach that resonates strongly with the interactive machine learning paradigm proposed by Holzinger et al.

---

## 4. Synthesis and Comparative Analysis

### 4.1 Points of Convergence

Despite their different methodological approaches, Holzinger et al. (2017) and Tonekaboni et al. (2019) converge on several important points. Both emphasise that explainability in medical AI is not a secondary concern or an add-on feature, but a fundamental design requirement that must be addressed from the outset. Both argue that the medical domain places unique demands on AI systems that make general-purpose XAI methods insufficient. And both point toward participatory, human-centred design processes as a necessary condition for developing AI systems that clinicians can actually use.

A second area of convergence concerns the *relational* nature of explanation. Both papers implicitly — and in Tonekaboni et al.'s case, explicitly — reject the notion that an explanation is a fixed property of a model. Explanations are relational: their value depends on who receives them, in what context, and for what purpose. This has significant implications for how XAI is evaluated. Benchmark evaluations of explanation quality that do not account for the intended audience and use context are, at best, incomplete.

### 4.2 Points of Tension

The two papers also exhibit productive tensions. Holzinger et al.'s framework is primarily concerned with building explainability *into* models — developing inherently interpretable systems rather than layering post-hoc explanations onto black boxes. Tonekaboni et al., by contrast, are agnostic about model architecture; their interest is in the explanatory interface between model and clinician, whatever form the underlying model takes.

This difference in emphasis reflects a genuine debate in the XAI literature. Advocates of inherent interpretability argue that post-hoc explanations are inherently unreliable — they approximate the behaviour of a model rather than revealing its true reasoning. Critics of this view argue that the most accurate models are rarely inherently interpretable, and that sacrificing accuracy for interpretability is an unacceptable trade-off in high-stakes domains. The two papers do not resolve this debate, but together they illuminate its stakes in the medical context.

### 4.3 Implications for Future Research

The dialogue between these two works suggests several productive directions for future research. First, the field would benefit from empirical studies that connect the technical properties of XAI methods with clinician-reported utility. Most XAI evaluations use proxy measures of explanation quality (fidelity, stability, comprehensiveness) rather than direct measures of clinical usefulness. Bridging this gap requires interdisciplinary collaboration between ML researchers, clinical informaticists, and human factors experts.

Second, the development of clinical XAI would benefit from richer theoretical frameworks that specify the relationship between explanation type, decision context, and desired outcome. Both papers gesture toward such frameworks, but neither fully develops one. The taxonomy proposed by Tonekaboni et al. is a useful starting point, but it requires further elaboration and empirical validation.

Third, as AI systems become more autonomous and are used in increasingly high-stakes clinical settings, the governance dimensions of XAI — accountability, transparency, auditability — will become correspondingly more important. This points toward the need for regulatory frameworks that specify explainability requirements for medical AI, informed by both the technical considerations raised by Holzinger et al. and the clinical considerations raised by Tonekaboni et al.

---

## 5. Conclusion

The papers reviewed here represent two essential perspectives on a shared problem: how to make AI systems in medicine trustworthy, usable, and accountable. Holzinger et al. (2017) establish the technical and philosophical foundations of XAI in the medical domain, articulating why explainability is necessary and surveying the methods available to achieve it. Tonekaboni et al. (2019) ground this framework in clinical reality, demonstrating that the success of XAI depends not only on technical sophistication but on attentiveness to the needs, workflows, and epistemic practices of clinical end-users.

What emerges from reading these works together is a picture of XAI in medicine as an inherently sociotechnical challenge — one that cannot be addressed by technical innovation alone, but requires ongoing dialogue between AI developers, clinicians, and the patients whose care these systems affect. The goal is not simply to produce models that explain themselves, but to produce explanations that are meaningful, actionable, and trustworthy in the contexts that matter most.

---

## References

Holzinger, A., Biemann, C., Pattichis, C. S., & Kell, D. B. (2017). What do we need to build explainable AI systems for the medical domain? *arXiv preprint arXiv:1712.09923*.

Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What clinicians want: Contextualizing explainable machine learning for clinical end use. *Proceedings of the Machine Learning for Healthcare Conference (MLHC)*.
