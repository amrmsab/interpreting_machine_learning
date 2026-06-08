# Chapter 1 – Motivation for Explainable AI

**Author:** [Diana Elzeftawy](https://www.linkedin.com/in/diana-rehan)

---

Imagine you apply for a loan. A few seconds later, a system tells you: *rejected*. No reason. No explanation. Just a number, and a door closed in your face.

Now imagine a doctor who gets an AI recommendation to treat a patient for pneumonia, but has no idea what the model looked at to reach that conclusion. Should the doctor trust it? Should she question it? What if the model was wrong for all the right-seeming reasons?

These situations are not hypothetical. They happen today. And they are exactly why **Explainable AI (XAI)** exists.

Modern machine learning systems can classify images, recommend products, detect fraud, and support medical decisions with impressive accuracy. But high accuracy alone does not mean a system is safe, fair, or trustworthy. In many real-world settings, the problem is not *that* a model produces answers. It is that it produces them in ways no one can properly inspect, question, or challenge.

This chapter explains the core ideas behind that problem in a simple way. It introduces the black box problem, why explainability matters in practice, who needs explanations, what makes an explanation *good*, and how the topic connects to trust, fairness, and accountability.

---

## 1. The Black Box Problem

The phrase *black box* usually refers to a system whose inputs and outputs are visible, but whose internal reasoning is opaque. You can see what went in, and you can see what came out, but the middle part is a mystery.

In machine learning, this often describes models like deep neural networks or large ensembles. These systems can involve millions of parameters and interactions that no human could trace step by step. But the black box problem is not *just* about complexity. A model becomes a black box when people cannot meaningfully determine **why** it produced a decision, **what evidence mattered most**, or **whether its reasoning was sensible**.

### Clever Hans: The horse that wasn't solving math

In the early 1900s, a horse named Clever Hans became famous in Germany for apparently solving arithmetic problems. People would ask him a question, and he would tap his hoof the correct number of times. Crowds were amazed. Scientists were baffled.

Then someone had a careful look. Hans was not doing arithmetic at all. He was picking up on tiny, unconscious cues from the people around him, such as slight shifts in posture, breathing, or facial expressions, that told him when to stop tapping. Remove the human audience, or blindfold the trainer, and Hans got everything wrong.

Machine learning models can behave in exactly the same way. A classifier might achieve excellent performance by relying on patterns that *happen* to work in one dataset but break down completely in the real world.

### The wolf-and-husky problem

A famous example in XAI teaching goes like this: a classifier is trained to distinguish wolves from huskies. It performs well. But when you look at *why* it is making those decisions, using explanation techniques we will explore in later chapters, you discover it is mostly looking at the **background** of the image. Wolves tend to appear in snowy scenes; huskies often do not. The model learned a background shortcut instead of learning what the animal actually looks like (Ribeiro et al., 2016; Molnar, 2024).

That is a harmless example, but the implications are not harmless at all.

### A real-world failure: hospital shortcuts in medical imaging

Zech et al. (2018) studied deep learning models trained to detect pneumonia from chest X-rays collected at different hospitals. On the surface, results looked promising. But hospitals are not identical. They use different scanners, different image processing pipelines, and different patient populations with different pneumonia rates.

What the models quietly learned was, in part, to **identify the hospital** that produced the image and use that as a clue for predicting pneumonia. If Hospital A had more pneumonia patients than Hospital B, then knowing "this image came from Hospital A" was already useful to the model, even though it has nothing to do with what pneumonia *looks like* in an X-ray.

When those models were moved to a different hospital system, performance dropped sharply. The shortcut stopped working. Without explanation tools, no one would have known.

The black box problem, then, is not just that a model is mathematically complicated. It is that **you cannot tell whether the model is making decisions for the right reasons**, and that distinction can have real consequences for real people.

---

## 2. Interpretability vs. Explainability — What's the Difference?

You will often see these two words used interchangeably, but they carry slightly different meanings in the research literature.

**Interpretability** is about whether a human can understand the cause of a decision, often by looking at the model itself. A simple decision tree, for example, is interpretable: you can literally follow the branches from input to output.

**Explainability** is about the ability to provide an understandable account of a decision, especially for models that are not inherently transparent. It often involves a separate technique applied *after* the model is trained.

Doshi-Velez and Kim (2017) define interpretability as "the ability to explain or present something in understandable terms to a human." Importantly, what counts as *understandable* depends heavily on who is asking and why.

Lipton (2018) adds a useful warning: "interpretability" is not a single property. Sometimes people want transparency of the model itself: can I read the weights and make sense of them? Other times they want a post-hoc explanation: can someone explain to me *after the fact* why this specific decision was made? Treating these as the same thing leads to confusion.

### The key idea: incompleteness

Doshi-Velez and Kim (2017) make an important argument that still shapes how people think about explainability: the need for interpretation arises from **incompleteness in the problem formalization**. Put simply, the problem we write down is often narrower than the one we actually care about.

When we train a machine learning model, we give it an objective, usually something measurable like minimizing a loss function or maximizing accuracy on a test set. The assumption is that optimizing that objective will produce a system that does what we actually want. But that assumption often breaks down. The mathematical objective we write down is almost never a complete description of everything we care about. It is a proxy, a simplified stand-in for a far richer set of human values, safety requirements, fairness constraints, and contextual expectations.

Doshi-Velez and Kim identify several situations where this incompleteness appears. One is scientific understanding. If the goal is knowledge rather than pure prediction, then a model that gives accurate answers but cannot explain *why* it works is less useful to a researcher. Another is multi-objective trade-offs. Important goals such as privacy and predictive accuracy may compete with each other and may not fit neatly into one objective function. In those cases, interpretability helps humans understand the trade-off. The same issue appears when the problem formalization cannot fully specify what "fair," "safe," or "reliable" means. Those ideas depend on context, judgment, and values that are hard to capture exactly in mathematics. As a result, a model can score well on paper while still violating the real intent in ways that remain invisible without structured inspection (Doshi-Velez and Kim, 2017).

This is a different kind of problem from ordinary *uncertainty*. Uncertainty can often be measured and communicated. For example, a classifier might output a probability score that reflects its confidence. Incompleteness is harder to handle. It is the gap between what we can formally specify and what we actually want, and that gap cannot always be fixed by collecting more data or tuning more parameters. That is where XAI becomes useful. Explanations are one of the tools that help humans notice and manage the effects of that gap (Doshi-Velez and Kim, 2017).

A concrete example helps. A model trained to predict recidivism risk in a criminal justice setting might achieve strong predictive accuracy while using a proxy variable that correlates with race. The formal objective, predict accurately, is being met. But the real goals, fairness, legal defensibility, and contestability, are not captured in that objective. Only by inspecting what the model is actually using as evidence can a human detect that something has gone wrong. That inspection is interpretability in action.

---

## 3. Why Explainability Matters

There is no single reason to want explanations. Different domains demand them for different purposes. Still, several motivations appear repeatedly across the literature.

### Safety and reliability

In high-stakes systems, incorrect reasoning can be dangerous in ways that average accuracy does not reveal. A model that works most of the time but fails for hidden reasons may be unsafe in deployment. Think of autonomous vehicles, medical diagnostics, or power grid management. Explanations help users check whether the system is relying on robust evidence or on brittle shortcuts that collapse when the environment changes slightly.

### Debugging and model improvement

Interpretability is often a debugging tool first. By examining which features, examples, or internal patterns matter most, practitioners can discover data leakage, misleading correlations, poor feature engineering, label errors, or biased proxy variables.

A striking example comes from Caruana et al. (2015), who built an interpretable model to predict pneumonia risk. The model revealed something alarming: having a history of **asthma** appeared to *lower* the predicted risk of dying from pneumonia. That sounds medically backwards because asthma is a serious complication.

The explanation? Asthma patients were being treated more aggressively and admitted earlier, which changed their outcomes in the data. A black-box model would have quietly learned to send high-risk asthma patients home. Without interpretation, that dangerous pattern would have been invisible (Caruana et al., 2015; Molnar, 2024).

### Scientific discovery

In some applications, prediction is just the beginning. Researchers use machine learning to study biology, agriculture, or social behavior because they want to *learn something about the world*, not only predict the next label. A highly accurate but opaque model is limited in that context: it tells a scientist *what* is likely to happen without showing *why*. Interpretability can turn a model into a tool for generating scientific insight rather than just a black box that produces numbers (Molnar, 2024).

### Human trust and decision support

Many machine learning systems are used to *support* human decision-makers, not replace them. A doctor, loan officer, or policy analyst may be expected to act on the model's output, but acting responsibly requires some basis for judgment. Users often want to know why the model made a recommendation, whether similar cases were treated consistently, and how confident the system is when inputs are unusual or unfamiliar.

### Recourse for affected people

People who are directly affected by automated decisions often need explanations for deeply practical reasons. A rejected loan applicant, a flagged job candidate, or a denied insurance customer may want to know what contributed to the decision and what could change in the future. Explanation, in this context, is not merely informative. It is tied to **recourse, contestability, and procedural fairness**.

---

## 4. What Makes a Good Explanation?

If explanations are important, the next question is: what counts as a *good* one? Research in psychology and XAI suggests that useful explanations are not exhaustive descriptions of everything the model computed. People do not want a full causal audit. They want explanations that help them answer a relevant question.

**Contrastiveness** is one of the most important properties. People often ask not "Why did this happen?" but "Why did this happen *instead of something else*?" A rejected loan applicant wants to know why they were denied *instead of approved*. A doctor wants to know why the model predicts pneumonia *rather than heart failure*. Contrastive explanations are especially valuable because they support action.

**Selectivity** matters too. Humans can process only so many reasons at once. A useful explanation highlights the most important factors instead of dumping an unstructured list of all possible contributors. Selectivity does not mean dishonesty. It means being cognitively manageable.

Explanations are also **social**. They are given to someone, in a context, for a purpose. The same model decision may need to be explained differently to a regulator, a patient, a data scientist, and a business executive. An explanation that is mathematically precise but inaccessible to its audience fails as an explanation, no matter how technically correct it is.

Miller (2019) adds that people are more interested in unusual or decision-changing factors than in background conditions that are always present. Saying a house is expensive because it has walls is not helpful. Saying it is expensive because it has a rare second balcony *is*. Good explanations direct attention to what is meaningful in context.

A final caution: explanation quality cannot be judged only by how persuasive it sounds. A smooth explanation may be psychologically satisfying while misrepresenting the real model. This is one of the major risks of post-hoc XAI methods: they can produce plausible stories that humans like even when those stories are not faithful to the true decision process. A good explanation must balance **human usefulness** with **fidelity to the underlying model**.

---

## 5. Who Needs Explanations?

One reason explainability is difficult is that different stakeholders need different kinds of understanding. A single explanation style rarely satisfies everyone.

**Creators**, meaning data scientists and engineers, often need explanations for debugging, validation, and model improvement. They want to know whether the system has learned robust structure or is exploiting shortcuts.

**Operators** interact directly with the system in practice. They may need concise, case-specific explanations that help them decide whether to rely on the output in a particular moment.

**Executors** make decisions based on the AI's output, such as a hiring manager or a clinician. They need enough understanding to justify an action and to know when human judgment should override the model.

**Decision subjects** are people directly affected by automated decisions. They may care less about model architecture than about whether the outcome was fair, whether it can be challenged, and what changes could alter future outcomes.

**Auditors and regulators** need explanations at the system level. Their concern is often whether a model is compliant, traceable, and consistent with legal or institutional standards.

**Data subjects** form another important group: people whose data helped train the system, even if they are not the immediate targets of a given decision. They may have concerns about how their data were used and whether the resulting model reproduces harmful social patterns.

The stakeholder perspective shows why explainability is not a single interface problem. It is a problem of matching forms of understanding to different roles, rights, and responsibilities.

---

## 6. Trust, Accountability, and Contestability

Interpretability is often linked to trust, but that relationship is more complicated than it first appears. It is tempting to assume that a more interpretable model is automatically more trustworthy. Lipton (2018) warns against that. Sometimes an explanation reveals that the model is using a biased proxy, a shortcut unrelated to the actual task, or a pattern too weak to rely on. In those cases, interpretation should *reduce* trust, not increase it.

Trust therefore needs to be understood more carefully. Broadly, a person may want confidence in three things: that the model performs well outside the lab, that it is not systematically unfair, and that its mechanism is understandable enough to justify reliance. These are related but distinct concerns, and a model can do well on one and poorly on the others (Lipton, 2018).

This is why accountability matters. The FAT/ML principles for accountable algorithms argue that there is always a human ultimately responsible for algorithmic systems and their consequences. "The algorithm did it" is not an acceptable excuse. Responsibility, explainability, auditability, and fairness all become part of a broader governance problem: if an automated system causes harm, who can inspect it, question it, and change it? (FAT/ML, 2016).

From the perspective of affected individuals, accountability is closely tied to *contestability*. A system is not meaningfully accountable if people cannot challenge its outcomes. That challenge requires more than access to source code. In many cases, affected individuals need an understandable explanation, a channel for recourse, and access to human review with real authority. Otherwise, explanation becomes symbolic rather than practical.

### The legal picture: GDPR and the EU AI Act

Accountability is also a legal question. Goodman and Flaxman's (2017) analysis of the GDPR popularized the debate over a "right to explanation." The more precise point is that EU data protection law gives individuals rights against certain forms of solely automated decision-making, including rights to obtain human intervention, express a point of view, and contest the decision under Article 22 (Goodman and Flaxman, 2017; GDPR, 2016).

The EU AI Act (2024) goes further and organizes the regulatory framework around a four-level risk pyramid. That pyramid gives a practical way to see where explainability matters most in real deployments.

![The EU AI Act risk pyramid, showing four tiers from bottom to top: Minimal Risk, Limited Risk, High Risk, and Unacceptable Risk.](images/ai_risk_pyramid.png)
*Figure: The EU AI Act risk pyramid. Source: [European Commission](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)*

**Minimal or no risk** sits at the base of the pyramid and covers the vast majority of AI systems currently used in the EU. The Act imposes no additional requirements on these systems. Spam filters and AI-enabled video games fall into this category. They may occasionally make errors, but those errors carry no serious consequences for people's rights or safety (EU AI Act, 2024).

**Limited risk**, which the official EC page calls the "transparency risk" tier, covers AI systems that interact directly with people in ways where the use of AI may not be obvious. The main requirement here is disclosure. Users must be told when they are interacting with a machine. Chatbots are the clearest example. A customer service bot or a medical triage assistant that could be mistaken for a human falls into this category, so providers must make the artificial nature of the system clear. Providers of generative AI also have obligations to label AI-generated content, especially deepfakes (EU AI Act, 2024).

**High risk** covers AI use cases that can pose serious risks to health, safety, or fundamental rights. The Act gives a detailed list. It includes AI in critical infrastructure such as transport and power, AI that affects access to education such as automated exam scoring, AI safety components in products such as robot-assisted surgery tools, AI used in hiring and worker management such as CV-sorting software, AI for credit scoring, biometric identification systems, AI used in law enforcement and immigration processing, and AI used in the administration of justice. For all of these, the Act imposes strict obligations. These include mandatory risk assessments, high data quality requirements, activity logging for traceability, detailed documentation, human oversight measures, and high standards of robustness and cybersecurity. Rules for most high-risk systems come into effect in August 2026 (EU AI Act, 2024).

**Unacceptable risk** sits at the top of the pyramid and means a complete prohibition. The Act bans eight specific practices that are seen as clear threats to people's safety, livelihoods, or rights. These include AI-based manipulation and deception that causes harm, AI systems that exploit psychological or social vulnerabilities, social scoring systems based on behavior or personal characteristics, systems that predict individual criminal offenses, mass scraping of the internet or CCTV footage to build facial recognition databases, emotion recognition in workplaces and educational institutions, biometric categorisation to infer protected characteristics, and real-time remote biometric identification by law enforcement in public spaces. These prohibitions became effective in February 2025 (EU AI Act, 2024).

The important idea for XAI is that transparency, documentation, traceability, and human oversight are central governance requirements across the high-risk tier. Explainability is not an optional extra in those contexts. It is part of what compliance means.

---

## 7. Case Studies: When Black Boxes Fail

The idea becomes clearer when you look at real cases. The next three examples show different ways black box systems can fail in practice.

### 7.1 Shortcut learning: wolves, huskies, and snow

Returning to the wolf-versus-husky example with more analytical depth: Ribeiro et al. (2016) used this case precisely because it shows how a model can appear to succeed while solving an easier, less meaningful problem than the one humans intended. Geirhos et al. (2020) gave this pattern a formal name, **shortcut learning**, and argued that deep models systematically tend to rely on decision rules that are easier to learn, even when those rules do not generalize to new environments. A model that has learned "snowy background means wolf" will fail the moment wolves appear in forests or huskies appear in the snow. Interpretability is valuable here because it can reveal whether a model has learned the intended signal or only a convenient proxy.

### 7.2 Hospital generalization failure in medical imaging

Returning to the Zech et al. (2018) study with deeper focus: what makes this case so important for XAI is not merely that performance dropped across hospitals. It is that the model's failure was *silent*. Inside the original dataset, the model that partly used hospital identity as a shortcut actually performed better on measured metrics than one that ignored it because hospital identity genuinely correlated with pneumonia rates in that data. The shortcut was rewarded by the objective. Only by inspecting what the model was actually using, through interpretability tools, could anyone detect that the performance was partly built on a proxy that would collapse in deployment.

### 7.3 COMPAS and contestability

ProPublica's (2016) investigation of the COMPAS risk assessment tool, used in U.S. criminal justice settings to predict recidivism, became central to debates about algorithmic fairness. The investigation raised serious questions about racial disparities in the tool's error rates, specifically that Black defendants were more likely to be incorrectly flagged as high risk, while white defendants were more likely to be incorrectly flagged as low risk. Beyond the statistical dispute, the deeper XAI lesson is about contestability: COMPAS is a proprietary system, meaning defendants and their lawyers could not inspect the model's reasoning. Whatever one concludes about the statistical fairness debate, an opaque system in a high-stakes legal setting creates a structural problem. People affected by the decision cannot meaningfully challenge the basis on which it was made (ProPublica, 2016).

These three cases support the same conclusion from different angles: explainability is needed not because humans are curious, but because models can fail in ways that remain invisible without structured inspection.

---

## 8. Evaluating Interpretability

If interpretability matters so much, how should it be evaluated? Doshi-Velez and Kim (2017) offer a useful framework for thinking about that question. Their main point is simple: people often make claims about interpretability without the right kind of evidence, and the evaluation should match the claim being made.

![Taxonomy of evaluation approaches for interpretability, showing three levels: Application-grounded (Real Humans, Real Tasks), Human-grounded (Real Humans, Simple Tasks), and Functionally-grounded (No Real Humans, Proxy Tasks). Higher levels are more specific and costly.](images/eval_taxonomy.png)
*Figure: Taxonomy of evaluation approaches for interpretability (Doshi-Velez and Kim, 2017)*

**Application-grounded evaluation** is the most realistic and demanding approach. It puts real users in their actual task setting and checks whether explanations really help. Suppose a system is designed to help radiologists detect lung cancer from CT scans. An application-grounded evaluation would recruit actual radiologists, have them use the model and its explanations in conditions close to real clinical work, and then measure the outcome: do they catch more errors? Do they override the model when it is wrong? Do they make safer decisions overall? This is the gold standard because it tests explanation where it matters most. The drawback is cost and complexity. Studies like this require domain experts, careful ethical design, and time (Doshi-Velez and Kim, 2017).

**Human-grounded evaluation** still involves real people, but it simplifies the task so that participants do not need deep domain expertise. Instead of testing inside a hospital, researchers might ask people to use an explanation to predict what the model will do on a new input, compare two explanations and choose which one is clearer, or identify what change in the input would reverse the prediction. For example, a study might show two people the same loan rejection with two different explanation formats and ask which one would be easier to act on. This approach is more practical than application-grounded evaluation and is useful for testing whether a person can actually understand and use an explanation, even if it does not fully reproduce real deployment conditions (Doshi-Velez and Kim, 2017).

**Functionally-grounded evaluation** uses no human subjects at all. Instead, it evaluates proxy properties believed to be related to interpretability, for example, whether a model uses fewer rules, has a shallower decision tree, relies on fewer features, or exhibits monotonic relationships between inputs and outputs. The reasoning is that simpler structures are generally easier to understand. A research paper might report that a sparse rule list with 10 rules is more interpretable than a random forest with 500 trees, using rule count as a proxy. This approach is fast and convenient, especially in early-stage method development, but it is also the most limited. A mathematically compact model is not automatically easy for a specific person to understand in a specific context (Doshi-Velez and Kim, 2017).

The key lesson from this taxonomy is that **evaluation should match the claim**. If a paper asserts that an explanation method helps doctors make safer decisions, a purely mathematical proxy is not sufficient evidence. That is a human outcome claim and requires human-grounded or application-grounded testing. If a method claims only to produce a structurally simpler model, proxy-based evaluation may be perfectly appropriate. The broader scientific discipline XAI needs is to stop relying on vague assertions that a method "feels intuitive" and to instead be specific: interpretable to whom, for what task, and measured how?

---

## 9. Historical Context: From Intrinsic Interpretability to XAI

The demand for explainability may feel contemporary, but its roots go back decades.

In classical statistics, interpretability was often built into the modeling approach itself. Linear regression coefficients, for example, were valued not only for prediction but because each one could be tied to a meaningful variable and an interpretable effect. The goal was to *represent* a relationship in understandable terms, not just produce a number.

In the expert systems era of the 1970s and 1980s, explanation was central by design. Systems like MYCIN used rule-based reasoning in medicine and were built so that their recommendation paths could be shown step by step (Buchanan and Shortliffe, 1984). The model did not need a separate explanation module. Its structure *was* the explanation.

Here is a simplified illustration of what MYCIN-style rule-based reasoning looked like:

```python
# A simplified MYCIN-style rule for illustrative purposes.
# MYCIN used rules like these to recommend antibiotic treatments.
# Each rule maps observable conditions to a conclusion 
# along with a "certainty factor," MYCIN's way of handling uncertainty.

def diagnose(organism_gram_stain, organism_morphology, patient_has_fever):
    """
    Real MYCIN had ~600 rules. This shows just one, for clarity.
    Note: this is NOT a real medical system.
    """
    if organism_gram_stain == "gram-negative" and organism_morphology == "rod":
        # Rule: gram-negative rods in a feverish patient -> consider E. coli
        if patient_has_fever:
            return {"diagnosis": "E. coli", "certainty": 0.7}
    return {"diagnosis": "unknown", "certainty": 0.0}

result = diagnose("gram-negative", "rod", patient_has_fever=True)
print(f"Diagnosis: {result['diagnosis']} (certainty: {result['certainty']})")
# Output: Diagnosis: E. coli (certainty: 0.7)
```

Notice what makes this interpretable: every rule is explicit and readable. A doctor can trace the exact path from symptom to conclusion. You lose that with modern neural networks, but you gain extraordinary predictive power.

The rise of modern machine learning changed that balance. As predictive performance improved through complex statistical methods, ensemble models, and deep neural networks, the field shifted away from intrinsically transparent reasoning and toward accuracy-oriented optimization. This brought enormous practical success, but it also created systems whose internal logic was difficult to interpret.

XAI can be understood as the modern response to that shift. In some cases, the response is to prefer simpler, intrinsically interpretable models where possible. In other cases, the response is post-hoc explanation: methods such as LIME, SHAP, saliency maps, feature importance scores, concept-based explanations, or example-based explanations that try to make a complex model more inspectable after training. Lipton's critique is important here. The historical story is not simply "linear models are interpretable and neural networks are not." A huge linear model with opaque engineered features may also be difficult to understand (Lipton, 2018; Molnar, 2024). The real question is always what kind of understanding is needed, and whether the chosen method provides it.

---

## 10. From Motivation to Methods

This chapter has focused on *why* explainability is needed, not on the full toolbox of techniques used to achieve it. That distinction matters. Before choosing a technique, one must first understand the purpose of explanation. Are we trying to debug a model, justify a decision, support recourse, discover scientific structure, or satisfy regulatory requirements? Different goals require different forms of explanation, and later chapters will address those methods in detail.

---

## 11. Conclusion

Explainable AI begins from a simple observation: good predictions are not enough.

In many real-world tasks, the formal objective captures only part of what humans care about. Safety, fairness, scientific understanding, contestability, and institutional accountability often remain outside the metric. That gap, which Doshi-Velez and Kim call the incompleteness of the problem formalization, is what makes explanation necessary.

The motivation for XAI is therefore broader than a vague desire for "transparency." It is a response to incomplete objectives, hidden shortcuts, and the need for models that people can inspect, question, and use responsibly. Black box systems are not a problem simply because they are complex. They become a problem when that complexity makes it hard to tell whether the system is reasoning well, failing in a dangerous way, or making decisions without enough justification.

As machine learning moves into more socially important settings, explainability becomes part of making these systems answer to human goals. It helps improve models, supports the people using them, and gives affected individuals a better chance to understand or challenge important decisions. That is why XAI is not a side issue in modern AI. It is one of the field's main responses to the limits of prediction alone.

---

## Further Reading

If you want to go deeper on any of the ideas in this chapter, the following resources are well worth your time.

- **Christoph Molnar's *Interpretable Machine Learning* (free online):** https://christophm.github.io/interpretable-ml-book/ — The clearest and most comprehensive introduction to the full field.
- **"Why Should I Trust You?" (Ribeiro et al., 2016):** https://arxiv.org/abs/1602.04938 — The original LIME paper, which introduced the wolf-husky example and sparked much of the post-hoc explanation literature.
- **"Towards a Rigorous Science of Interpretable Machine Learning" (Doshi-Velez & Kim, 2017):** https://arxiv.org/abs/1702.08608 — Foundational for thinking carefully about what interpretability means and how to evaluate it properly.
- **"The Mythos of Model Interpretability" (Lipton, 2018):** https://arxiv.org/abs/1606.03490 — An essential critical perspective on the slippery ways "interpretability" gets used in practice.
- **ProPublica's COMPAS investigation (2016):** https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing — The investigation that brought algorithmic fairness into mainstream conversation.
- **Shortcut learning in deep neural networks (Geirhos et al., 2020):** https://www.nature.com/articles/s42256-020-00257-z — A thorough synthesis of why models learn the wrong things and what can be done about it.
- **EU AI Act (official text and explainer):** https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai — The source of the risk pyramid and all related regulatory obligations.

---

## References

1. Biran, O., and Cotton, C. (2017). *Explanation and justification in machine learning: A survey*. Proceedings of the IJCAI 2017 Workshop on Explainable AI.
2. Buchanan, B. G., and Shortliffe, E. H. (1984). *Rule-Based Expert Systems: The MYCIN Experiments of the Stanford Heuristic Programming Project*. Addison-Wesley. Chapter 11: https://people.dbmi.columbia.edu/~ehs7001/Buchanan-Shortliffe-1984/Chapter-11.pdf
3. Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., and Elhadad, N. (2015). *Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission*. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1721–1730. https://doi.org/10.1145/2783258.2788613
4. Doshi-Velez, F., and Kim, B. (2017). *Towards a rigorous science of interpretable machine learning*. arXiv. https://arxiv.org/abs/1702.08608
5. Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. (2020). *Shortcut learning in deep neural networks*. Nature Machine Intelligence, 2, 665–673.
6. Goodman, B., and Flaxman, S. (2017). *European Union regulations on algorithmic decision-making and a "right to explanation"*. AI Magazine, 38(3), 50–57. Preprint: https://arxiv.org/abs/1606.08813
7. Kim, B., Khanna, R., and Koyejo, O. O. (2016). *Examples are not enough, learn to criticize! Criticism for interpretability*. NeurIPS.
8. Lipton, Z. C. (2018). *The mythos of model interpretability*. Communications of the ACM, 61(10), 36–43. Preprint: https://arxiv.org/abs/1606.03490
9. Miller, T. (2019). *Explanation in artificial intelligence: Insights from the social sciences*. Artificial Intelligence, 267, 1–38.
10. Molnar, C. (2024). *Interpretable Machine Learning* (3rd ed.). https://christophm.github.io/interpretable-ml-book/
11. Partnership on AI / FAT/ML. (2016). *Principles for accountable algorithms and a social impact statement for algorithms*. https://www.fatml.org/resources/principles-for-accountable-algorithms
12. ProPublica. (2016). *Machine bias: There is software used across the country to predict future criminals. And it is biased against Blacks.* https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
13. Regulation (EU) 2016/679 of the European Parliament and of the Council (General Data Protection Regulation). Official text: https://eur-lex.europa.eu/eli/reg/2016/679/oj
14. Regulation (EU) 2024/1689 of the European Parliament and of the Council (Artificial Intelligence Act). Official text: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
15. Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135–1144. Preprint: https://arxiv.org/abs/1602.04938
16. Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., and Oermann, E. K. (2018). *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study*. PLOS Medicine, 15(11), e1002683. https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683

---

To cite this, please use the following bibtex:

```bibtex
@misc{elzeftawy_2026_XAI,
  author       = {Diana Elzeftawy},
  title        = {Interpreting Machine Learning: A Gentle Introduction, Chapter 1},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/interpreting_machine_learning}},
}
```
