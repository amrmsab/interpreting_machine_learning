# Explanations Through Examples

### *Prototypes, Influence Functions, and Concept Vectors — How Machines Can Reason More Like Us*

**Mahmoud Abdelgawad**

---

## 1. A radiologist, a lawyer, and a stuck model

Imagine a radiologist looking at a scan. She has spent twenty years doing this. You ask her how she knows the small bright region in the upper lobe is a tumour, and she does not pull out a list of pixel weights. She says something like:

> "It looks like the case I had in 2019 — same shape, same edge, same patient profile. I'd biopsy."

Now imagine a lawyer arguing a case. She does not enumerate features of the situation and assign each a number. She says:

> "This is consistent with precedent in *Smith v. State*. The court reasoned there that…"

And finally a biologist looking at a deep-learning model that supposedly recognises animals. She does not ask which pixels were important. She asks:

> "Is this thing actually responding to the animal, or is it just locked onto stripes?"

Three different domains, three different experts, one shared habit: **they explain decisions through examples and concepts, not through feature scores.** Yet for the last decade, most of explainable AI has handed us tools that do the opposite. SHAP and LIME — which you have already met earlier in this book — return long lists of features and weights. That is fine for a data scientist staring at a tabular model. It is close to useless for a radiologist.

This chapter is about the family of XAI methods that try to close that gap. We will cover four ideas, in order of how they evolved:

1. **Case-Based Reasoning and Prototypes** — the classical foundation, much older than deep learning.
2. **ProtoPNet** — what happens when you bake prototype-thinking *into* the architecture of a neural network.
3. **Influence Functions** (Koh & Liang, 2017) — given a prediction, which training examples were most responsible for it?
4. **Concept Activation Vectors / TCAV** (Kim et al., 2018) — does a trained model actually understand the concept of "stripes"? Probe it and find out.

I will be honest about where each method shines and where it falls apart. By the end I will tell you what I personally took away from spending several weeks living inside these papers.

A small note on tone: I am going to assume you are comfortable with neural networks and have at least seen a gradient before. I am *not* going to assume you remember linear algebra fluently or have read the original papers. Where the maths matters, we will go through it slowly. Where it does not, we will skip it.

---

## 2. Why feature-based explanations were never going to be enough

Before we build anything new, it helps to understand why the existing tools fall short for human reasoners.

Suppose a SHAP analysis tells you that for a particular MRI scan classified as cancerous, the most important inputs were:

```
Pixel 24,581:   weight = +0.82
Pixel 12,003:   weight = -0.41
Pixel 89,127:   weight = +0.19
...
```

What does that *mean*? You cannot tell whether the model is paying attention to the tumour, the metadata bar at the corner of the scan, or a coffee stain on the digitiser. You can render a saliency map and squint at it, sure, but a 2018 study by Adebayo et al. showed that some popular saliency methods produce nearly identical maps for a trained network and a randomly-initialised one — meaning the maps were not really tracking what the model had learned at all.

There is a deeper problem too. Tim Miller, in his 2019 paper *Explanation in Artificial Intelligence: Insights from the Social Sciences*, surveyed decades of cognitive psychology and arrived at three findings worth remembering:

- **Explanations are contrastive.** People do not ask "why X?" — they ask "why X rather than Y?"
- **Explanations are selective.** A good explanation picks one or two relevant causes, not every cause.
- **Explanations are social.** They are a conversation, not a data dump.

Feature attributions are none of those things. A 4,000-element weight vector is not contrastive, not selective, and not social. So researchers started looking for explanation forms that *were*.

---

## 3. Case-Based Reasoning: the oldest trick in the book

Long before deep learning, AI had Case-Based Reasoning (CBR), formalised most famously by Aamodt and Plaza in 1994. CBR is almost embarrassingly simple, and that is the point. Given a new problem:

1. **Retrieve** similar past cases from a library.
2. **Reuse** the solution from the closest case, adapted to the new situation.
3. **Revise** the adapted solution if it does not quite work.
4. **Retain** the new case for next time.

This is how doctors and lawyers actually work. It is also how a k-Nearest-Neighbour classifier works, if you squint. The explanation is built in: "I predicted class A because the three closest cases were all class A, and here they are." There is nothing to interpret.

The problem is that CBR scales badly. With a million training images, "the three closest cases" might be a useless explanation if those cases all look like noise to a human. So researchers asked the next obvious question: out of all these cases, which ones are *prototypical*?

### Prototypes and criticisms

A 2016 paper by Kim, Khanna, and Koyejo at NeurIPS introduced a clean version of this idea. For each class, you learn two small sets of examples:

- **Prototypes** — the most representative members of the class. The model's mental snapshot of "what a golden retriever looks like."
- **Criticisms** — examples that the prototypes fail to cover. A hairless dog still correctly classified as a dog. A blurry, off-centre cat. The edge cases.

Prototypes tell you what the model has internalised as typical. Criticisms tell you where its mental image breaks down. Together, they give you something a textbook would call *model understanding* — a sense of the model's worldview, not just a single prediction.

This was a beautiful idea, but it was still a post-hoc analysis bolted onto an existing model. The next paper went further.

---

## 4. ProtoPNet: making interpretability part of the architecture

In 2019, Chen and colleagues at Duke published a paper at NeurIPS with the irresistible title **"This Looks Like That."** Their architecture, ProtoPNet, has a property that almost no other interpretability method can claim: **it cannot make a prediction without showing its work.**

Here is the rough idea. A standard CNN takes an image, computes a feature map, and pushes that through a classifier to get class probabilities. ProtoPNet adds a layer in between. During training, the network learns a small fixed number of *prototype patches* per class — say, 10 prototypes per class. Each prototype is a small region of feature space that corresponds to a particular visual pattern: the curve of a beak, the texture of a wing, the shape of an ear.

At test time, when you show the network a new image, it asks one question for every prototype:

> "How similar is the most similar patch in this image to me?"

It collects all those similarity scores, multiplies them by learned weights, and sums to a class score. A typical prediction looks like this:

```
This image is a clay-coloured sparrow because:

  - Patch in upper-left looks like prototype #3 of clay-coloured sparrow
    (similarity = 4.81, weight = 1.18)  →  contributes 5.66
  - Patch in middle looks like prototype #7 of clay-coloured sparrow
    (similarity = 3.92, weight = 1.05)  →  contributes 4.12
  - ...

Total class score for clay-coloured sparrow = 19.4 (winner)
```

The explanation is not bolted on after the fact. The explanation **is** the computation. If you delete the similarity scores, there is no prediction. Compare this with influence functions or TCAV, which we will meet in a moment — those are *post-hoc*, applied to an already-trained black box. ProtoPNet is *intrinsic*. Different category, different guarantees.

ProtoPNet has been followed by a small army of variants — ProtoTree, Deformable ProtoPNet, ProtoVAE — but the core idea has held up. A reference implementation lives at <https://github.com/cfchen-duke/ProtoPNet> if you want to play with it.

I will say more about ProtoPNet at the end. For now, hold onto one thing: it costs almost nothing extra at inference time. A prototype-similarity layer is cheap. If you can train it once, you get interpretability for free at every prediction.

---

## 5. Influence Functions: which training point is to blame?

Now we change gears. So far, the explanations have all been about the *input*: this image looks like that prototype. There is a stranger and arguably more revealing question we have not asked yet:

> Forget the input for a moment. Out of the *fifty thousand training points* this model saw, which ones are most responsible for the prediction it just made?

This is a fundamentally different question. It treats the training set as the explanation. And it gives you something nobody else does: a way to debug a model by debugging its data.

### The intuition

You are sitting an exam and you get a question wrong. You wonder: which flashcard, out of the hundreds you studied, was most responsible for that wrong answer? Maybe one specific card had a misleading example that planted the wrong intuition. If you could surgically remove that card from your past and re-study, would you have got the question right?

The naive way to answer this for a machine-learning model is brutal but obvious: **leave-one-out retraining.** Pick a training point, delete it, retrain the entire model from scratch, and see how the prediction on your test point changes. Repeat for every training point. With 50,000 examples, that is 50,000 full retrains. For a modern deep network, that is years of GPU time per question.

Koh and Liang's contribution was: you do not have to retrain. There is a closed-form approximation that gives you the same answer without ever touching the training loop again. It is the kind of result that, when you first see it, looks like it must be wrong.

### The maths, gently

If you trained your model by minimising a loss $L(\theta)$ over training points, and your fitted parameters are $\hat{\theta}$, then the influence of training point $z_i$ on the loss at test point $z_{\text{test}}$ is:

$$
\mathcal{I}_{\text{up,loss}}(z_i, z_{\text{test}}) = -\nabla_\theta L(z_{\text{test}}, \hat{\theta})^\top \, H_{\hat{\theta}}^{-1} \, \nabla_\theta L(z_i, \hat{\theta})
$$

That looks scary. Let us read it left to right, in plain English:

- $\nabla_\theta L(z_{\text{test}}, \hat{\theta})$ is the gradient of the test point's loss with respect to the model parameters. It says: *"to make the test loss go down, push the parameters in this direction."*
- $\nabla_\theta L(z_i, \hat{\theta})$ is the same thing for training point $z_i$: *"if I'd had a tiny bit more of this training point, the parameters would have moved in this direction."*
- $H_{\hat{\theta}}^{-1}$ is the inverse Hessian — the second derivative of the loss. Roughly: *"how curved is the loss surface around our solution?"* Dividing by it (which is what the inverse does) accounts for the fact that some parameter directions are more sensitive to perturbation than others.
- The whole product, with the minus sign, is: *"how much does the test loss change if I upweight this one training point a tiny bit?"*

If the score is **negative**, upweighting that training point makes the test loss go down — meaning it pulled the model in the right direction. That training point *helped* this prediction. If the score is **positive**, upweighting it makes the test loss go up — it *hurt*.

That is the whole idea. The machinery of the paper is mostly about making this tractable for big models (computing $H^{-1}$ exactly is expensive, so they use a trick called *stochastic conjugate gradient* or alternatively a method called *LiSSA*), but the core formula is what is doing the work.

### Seeing it run

Theory is one thing. Let me show you what this looks like in practice on a tiny example. Save the file `influence_demo.py` next to this chapter and run it; here is the heart of it:

```python
import numpy as np

rng = np.random.default_rng(0)

# A toy 2D dataset: two Gaussian blobs.
X0 = rng.normal(loc=[-2, 0], scale=1.0, size=(50, 2))
X1 = rng.normal(loc=[+2, 0], scale=1.0, size=(50, 2))
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(50), np.ones(50)])

# Inject a deliberately weird point: class 1, but sitting in class 0 territory.
X[0] = np.array([-3.0, 0.5])
y[0] = 1.0

# Train logistic regression with vanilla gradient descent.
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
theta = np.zeros(2)
for _ in range(2000):
    p = sigmoid(X @ theta)
    theta -= 0.1 * X.T @ (p - y) / len(X)

# Hessian of the logistic loss: H = (1/n) sum_i p_i (1 - p_i) x_i x_i^T
p = sigmoid(X @ theta)
H = (X.T * (p * (1 - p))) @ X / len(X)
H_inv = np.linalg.inv(H)

# Influence of training point i on a chosen test point.
def grad(x, y_true): return (sigmoid(x @ theta) - y_true) * x

x_test, y_test = np.array([1.5, 0.0]), 1.0
g_test = grad(x_test, y_test)

scores = [- g_test @ H_inv @ grad(X[i], y[i]) for i in range(len(X))]
```

When you run this, the most harmful training point — the one most responsible for *worsening* the prediction on the test point — is `idx=0`, with a score of **+3.96**. The next worst is around **+0.19**. Our deliberate bad apple is twenty times more harmful than anything else, and we found it without retraining once.

This is the whole pitch in miniature. Run influence functions across your training set, look at the points with extreme scores, and you have a tool for **catching mislabelled data, identifying bias, and debugging weird predictions** — all by inspecting the training set, not the model weights.

### What Koh and Liang showed it could do

In the original paper, they put this to work in four memorable case studies. The deck I taught from collected them, but they are worth restating because each one is a different *kind* of insight:

1. **Comparing models that agree.** They trained two different models — an Inception network and an SVM — on the same fish-vs-dog task. Both got the same test image right. But influence functions revealed that the two models cared about *completely different* training points to make that prediction. The Inception network had learned abstract semantic features; the SVM had latched onto raw pixel similarity. The predictions agreed; the reasoning did not.

2. **Adversarial training-set attacks.** They showed you can perturb a single training image *imperceptibly* — same eight-bit pixel values to a human eye — and flip the prediction on 57% of test points. This was, at the time, the first demonstration that you could attack a deep network through its training set without anyone being able to see the attack.

3. **Domain mismatch in medical data.** A model predicting hospital readmission was trained mostly on adult records. There were four children's records in the dataset. Those four points turned out to be **30 to 40 times more influential** than anything else, causing systematic errors. Looking directly at the model weights did not surface this. Influence functions did, immediately.

4. **Finding label noise.** They flipped 10% of the labels in a spam dataset. Then they ranked the training points by the magnitude of their influence and inspected the top-ranked ones. This recovered accuracy faster than both random inspection and inspecting the highest-loss points — meaning influence is genuinely picking up on something different from "the model got this one wrong."

That last one is, I think, the most practically valuable use of influence functions. Most real datasets are noisy. Influence functions give you a principled way to clean them.

### The catch

Koh and Liang were honest about the limitations in their own paper, and a lot of follow-up work has fleshed them out. The biggest one: **the approximation breaks down in deep networks**, the very setting where you most want it.

The derivation assumes the loss is convex around your solution. For logistic regression and linear models, fine. For a deep network with millions of parameters and a wildly non-convex loss landscape, less fine. Basu, Pope, and Feizi (2020) ran the experiment everyone should have run earlier: they compared influence-function predictions against *actual* leave-one-out retraining, on real deep networks. The two often disagreed badly. Bae et al. (NeurIPS 2022), in a paper with the cheeky title *"If Influence Functions are the Answer, Then What is the Question?"*, dug into why and showed the issue is fundamental, not a bug in the implementation.

So influence functions are not a magic wand. They are a tool that works beautifully in shallow regimes — logistic regression, linear models, perhaps the last few layers of a frozen deep net — and shakily everywhere else. Use them, but verify.

---

## 6. TCAV: do you actually understand stripes?

Now we change gears one more time. Influence functions ask: *which training data drove this prediction?* Concept Activation Vectors, introduced by Been Kim and colleagues at ICML 2018, ask something completely different. Rather than tracing predictions back to data points or input pixels, TCAV interrogates the model in the language a human would actually use: concepts. *Does this model rely on stripes when it predicts zebras? Does it rely on redness when it predicts fire engines? Does it lean on something it should not, like a demographic attribute, when it predicts an everyday object?*

It is the most elegant idea in this chapter, and to my eye also the most underused.

### The setup

Imagine you have trained a network to classify ImageNet images. You suspect, but cannot prove, that it relies on stripes to recognise zebras. How would you test this?

The naive way is to train a separate "is it striped?" classifier from scratch and run both on the same images. But that does not tell you anything about the *zebra* network. It tells you about the new classifier.

Kim et al.'s insight: you do not need to retrain anything. Take the zebra network and freeze it. Pick any layer in the middle — call it layer $L$. When you push an image through, layer $L$ produces an activation vector. That vector is the network's *internal representation* of the image. If the network has learned "stripes" as a concept, that information must live somewhere in layer $L$'s activations across many striped images.

So:

1. **Pick a concept.** "Striped." Collect about 30 images of striped things — zebras, tigers, candy canes, striped shirts, anything. Also collect 30 random non-striped images.
2. **Get activations.** Push all 60 through the network and record the activations at layer $L$.
3. **Train a tiny classifier.** A single linear classifier — a logistic regression, even — that takes a layer-$L$ activation and predicts "striped or not."
4. **The CAV is the normal vector to that classifier's decision boundary.** A direction in activation space that, if you move along it, makes things "more striped" according to the network's internal representation.
5. **Score it.** For every image of zebras (the class you care about), compute the directional derivative of the zebra-class output with respect to the CAV direction at layer $L$. If it is positive, this image is being pushed toward "zebra" by its striped-ness. The TCAV score is the fraction of zebra images for which this derivative is positive.

If TCAV($\text{striped}, \text{zebra}, L$) = 0.82, it means: for 82% of zebra images, "more striped" pushes the network toward "zebra" at layer $L$. The model has internalised stripes as zebra-relevant.

### Why this is so cool

There are three things I love about TCAV.

**One:** the concept is *yours*. You define it. Stripes, redness, polka-dots, "looks like a wheel," "presence of a tie," "diversity of colour." Anything you can collect 30 examples of. The model never had to be trained with these concepts as labels. They are imposed by you, after the fact, on a frozen network.

**Two:** it is global, not local. SHAP and LIME explain a single prediction. TCAV explains an entire class. *Across all zebras, how much does this model rely on stripes?* That is the kind of question a domain expert, regulator, or auditor actually wants to ask.

**Three:** it surfaces bias in a way nothing else can. In the 2018 paper, Kim et al. ran TCAV against ImageNet classes using gender concepts. The "female" concept was strongly influential for the **apron** class. The "male" concept was strongly influential for **rugby ball**. A race-related concept correlated with predictions of **ping-pong balls**. None of these demographic attributes were *labels* in ImageNet. The model was nonetheless using them. TCAV did not infer this from the model's parameters or its outputs alone — it asked the model directly, in concept-space, and the model answered.

Nothing else in the XAI toolbox can do that as cleanly.

### Sanity check from the original paper

A skeptic might say: this whole pipeline is fragile. You are training a tiny linear classifier on 30 noisy activation vectors and reading meaning into a derivative. Why should I trust it?

Kim et al. anticipated this and did something I think every XAI paper should do — they ran a controlled human study. They presented 50 humans with a binary task: given a saliency map, identify the dominant concept the network was using. **The humans were correct 52% of the time.** A coin flip. Saliency maps had effectively no diagnostic power for the question they were supposedly designed to answer.

TCAV, on the same controlled task, tracked ground truth. That is the most damning empirical comparison in the XAI literature, and it is buried in a single paragraph in the appendix.

### The catch (again)

TCAV is not perfect. The two main limitations:

- **Concept quality is the new bottleneck.** Garbage concepts in, garbage scores out. If your "striped" set is actually full of vertically-oriented objects of any kind, your TCAV scores measure something other than stripes. The methodology is only as careful as your data collection.
- **Confounded concepts get muddled.** TCAV can struggle to distinguish visually similar concepts that travel together — "striped tiger" vs "striped zebra" — because the linear separator in activation space picks up on both.

So, like influence functions: useful, but verify.

---

## 7. A side-by-side comparison

Now that we have all four ideas in the room, here is how they line up. This is the most important table in the chapter, and I would bookmark it.

| | **Prototypes / ProtoPNet** | **Influence Functions** | **TCAV** | **SHAP / LIME** (for contrast) |
|---|---|---|---|---|
| What it explains | Why a prediction was made | Why a prediction was made | What the model learned about a class | Why a prediction was made |
| Unit of explanation | Visual patches | Training examples | Human-defined concepts | Input features / pixels |
| Local or global? | Local | Local | **Global** | Local |
| Post-hoc or intrinsic? | Intrinsic (ProtoPNet) | Post-hoc | Post-hoc | Post-hoc |
| Needs retraining? | Yes (to use ProtoPNet) | No | No | No |
| Cost at inference | **Cheap** | N/A (offline analysis) | N/A (offline analysis) | Cheap (LIME) to expensive (KernelSHAP) |
| Best at surfacing | Visual reasoning patterns | Bad data, label noise, domain mismatch | Hidden bias, conceptual reliance | Feature importance for tabular data |
| Where it breaks | Needs careful prototype design | Deep, non-convex models | Confounded concepts, bad concept sets |  |

Notice that these methods *do not really compete*. They answer different questions with different units of explanation at different scopes. A rigorous audit of a deployed model would use **all of them**:

- ProtoPNet tells you *what visual patterns* the model is matching against.
- Influence functions tell you *what data* shaped those patterns.
- TCAV tells you *what concepts* the model learned along the way.
- SHAP/LIME tells you, for the boring tabular models that still run most of the world, *which input features* drove a single decision.

If a regulator asked me to certify a medical-imaging model tomorrow, I would want answers from at least three of these methods before signing off. Any one of them on its own is a partial view.

---

## 8. The honest list of caveats

I want to list the limitations of this whole family of methods in one place, because it is too easy to come away from a chapter like this thinking the problem is solved.

1. **Faithful explanations of biased models are still dangerous.** Influence functions and TCAV both faithfully describe what the model has actually learned. If what the model has learned is biased — if it is using "female" to predict "apron" — then the explanation is *correct*, and that correctness might be exactly what makes deploying the model worse, because someone will read the explanation and say "well, at least we understand it now." Understanding a bad system is not the same as fixing it.

2. **Influence functions are mathematically local.** They measure the effect of *infinitesimally* upweighting one training point. If you actually want to know what would happen if you removed an entire hospital's worth of records from the training set, the linear approximation breaks down. The paper itself notes this in its discussion section.

3. **TCAV depends on you choosing the right concepts.** If a model is using a concept you never thought to test, TCAV will not find it. It is a powerful microscope, but you have to point it somewhere.

4. **Prototypes can be misleading too.** A prototype is just an example the model considers typical. If the training distribution is biased, the prototypes will inherit that bias. ProtoPNet does not magically clean your data.

5. **All of these methods compete for the same scarce resource: human attention.** A model with 10 prototypes per class, ranked influence scores, and TCAV scores against 20 concepts produces a *lot* of artifacts. Whether anyone has time to look at all of them is its own question.

---

## 9. What I actually think after working through these papers

I will close with my own view, because the rubric for this book asks us to engage critically and not just summarise.

After going through numerous papers covering all the topics laid out above — re-reading Koh & Liang, working through the TCAV derivation by hand, reimplementing the influence-function demo you ran above, and studying the ProtoPNet architecture in detail — this is what I have come to believe.

**These methods are not in competition.** I started this chapter expecting to pick a winner. I do not have one. Influence functions, TCAV, and prototype-based methods solve genuinely different problems, and the right answer for a real audit is to use the combination. The framing of "which XAI method is best" is, I think, the wrong question. Each of them illuminates one face of the model. You need several to see the shape.

**But I do think ProtoPNet deserves a default seat at the table.** Here is my reasoning. Influence functions are an offline analysis tool — you run them when something has gone wrong. TCAV is an audit tool — you run it when you want to interrogate the model. Both are valuable, but neither is "always on." ProtoPNet is. The interpretability cost is paid once, at training time. From then on, every single prediction the network makes comes with a free, faithful, "this looks like that" explanation, at essentially zero inference overhead. You do not have to remember to ask for it.

In a world where a lot of models will be deployed by people who are *not* going to remember to run a post-hoc audit — most of the world, in other words — having interpretability built in by default is a good architectural choice. It is the difference between a car that has airbags and a car where you can install airbags later if you fill out a form. You want the first car.

I am not saying ProtoPNet should replace influence functions or TCAV. I am saying the architectural choice — *intrinsic interpretability over post-hoc* — should be the default whenever it is feasible, and ProtoPNet is the cleanest existing demonstration that it is feasible. Use influence functions when you suspect a data problem. Use TCAV when you want to probe what the model has learned. But build the model itself, when you can, in a way that explains itself for free.

**There is one more thing I have come to believe**, and it is bigger than the choice between any two methods on the table. I do not think real interpretability can be reached with a one-size-fits-all toolbox. The methods in this chapter are general; they work on any classifier, with any concept set, on any training set. That generality is their strength as research contributions, but it is also their ceiling as deployed tools. A radiologist does not need a generic explanation. She needs an explanation in the language of her field — anatomical landmarks, lesion morphology, the differential diagnoses she is trained to consider. A loan officer needs an explanation in the language of credit risk, not in the language of pixel patches or activation vectors.

Real interpretability, I think, will come from customising the model — and the explanation interface around it — together with experts in the field the model serves. Co-designing with clinicians, lawyers, regulators, and domain scientists will always produce something more accurate and more useful than dropping a general-purpose XAI method on top of an off-the-shelf network. The methods in this chapter are the building blocks. The actual interpretable system, the one a professional will trust on a Tuesday afternoon during a hard case, will be built *with* those professionals, *for* their problem. A general method is a starting point. A bespoke system, designed alongside the people who will actually use it, is the destination.

That is my answer. It came out of running the code, sitting with the failure modes, and asking which approach I would actually trust if I were the radiologist at the start of this chapter.

---

## 10. Where to go next

If you want to go deeper, here is a short, opinionated reading list, in the order I would tackle them.

**Start here:**
- Tim Miller (2019), *Explanation in Artificial Intelligence: Insights from the Social Sciences.* The best survey of why human explanation is contrastive, selective, and social. Read this before any technical paper. <https://arxiv.org/abs/1706.07269>
- Doshi-Velez & Kim (2017), *Towards a Rigorous Science of Interpretable Machine Learning.* The setup paper for this whole field. <https://arxiv.org/abs/1702.08608>

**The two papers this chapter covers:**
- Koh & Liang (2017), *Understanding Black-box Predictions via Influence Functions.* ICML. <https://arxiv.org/abs/1703.04730>
- Kim et al. (2018), *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV).* ICML. <https://arxiv.org/abs/1711.11279>

**Going further:**
- Chen et al. (2019), *This Looks Like That: Deep Learning for Interpretable Image Recognition.* NeurIPS. The ProtoPNet paper. Code at <https://github.com/cfchen-duke/ProtoPNet>.
- Kim, Khanna & Koyejo (2016), *Examples are not Enough, Learn to Criticize! Criticism for Interpretability.* NeurIPS. The prototype-and-criticism paper.
- Han, Wallace & Tsvetkov (2020), *Explaining Black Box Predictions via Influence Functions.* ACL. The follow-up that brings influence functions to NLP.

**Read these to stay honest:**
- Basu, Pope & Feizi (2020), *Influence Functions in Deep Learning Are Fragile.* The paper that empirically broke deep-network influence functions. <https://arxiv.org/abs/2006.14651>
- Bae et al. (2022), *If Influence Functions are the Answer, Then What is the Question?* NeurIPS. The theoretical follow-up.
- Adebayo et al. (2018), *Sanity Checks for Saliency Maps.* NeurIPS. The paper that broke a lot of feature-attribution methods. A useful check on overconfidence.

**Watch:**
- Been Kim's ICML 2018 talk on TCAV. Search YouTube for *"TCAV ICML 2018 Been Kim"*. Worth the half-hour.
- Yannic Kilcher's paper walkthroughs of both Koh & Liang and Kim et al. He is excellent on the maths.

**Run code:**
- The full `influence_demo.py` referenced in this chapter is in this directory. Open it in Colab, run it top to bottom, and try changing the `x_test` point. Also try removing the `X[0] = ...` line that injects the bad apple — you should see the harmful-influence ranking become much less dramatic, which is exactly what should happen.

---

## Acknowledgement of AI use

Per the course AI use policy, I want to be transparent about how I used LLM tooling in preparing this chapter. I used Claude to help structure the chapter outline, refine prose, and check the influence-function demo for arithmetic correctness. All technical content, the choice of papers, the limitations, the comparison table, and the conclusion in section 9 reflect my own reading and judgement after working through the primary sources. Every paper cited has been read directly, not paraphrased through an LLM summary. Where I describe what a paper "showed" or "proved," I am describing what I read in the paper itself.

---

To cite this, please use the following bibtex:

```bibtex
@misc{abdelgawad_2026_XAI,
  author       = {Mahmoud Abdelgawad},
  title        = {Interpreting Machine Learning: A Gentle Introduction, Chapter 5},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/interpreting_machine_learning}},
}
```
