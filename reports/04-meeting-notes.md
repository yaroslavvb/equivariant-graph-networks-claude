# The meeting that prompted this

This report exists because of a reading-group discussion held on **23 July 2026**, in a café in San
Francisco, by a group that calls itself *Gradient Dissent*. Five people, about two and a quarter
hours, roughly forty per cent of it technical.

The account below is a **technical extraction**, not a transcript. The original recording is a
private conversation among named individuals and contains a great deal of personal material —
health data, finances, third parties who were not in the room and did not consent to publication.
None of that appears here. Speakers are identified by role rather than by name, for two reasons:
the automatic diarisation in the source is demonstrably unreliable (it assigns nine speaker
identities to five people and swaps them mid-thread), so any named attribution would carry a real
chance of being wrong; and the technical content stands on its own without them.

Quotations are from an automatic transcription and are **approximate**. Where the transcriber
garbled a technical term, the repair is shown in brackets. Read them as faithful to the sense and
not to the syllable.

---

## What the group was doing

They had set out to survey what machine learning has actually accomplished in materials science.
The reading list had been assembled by asking a language model for recent work, and the session
opened with a paper about a fine-tuned language model for materials text. It did not survive
contact. The methodological objection was sharp and correct:

> "They fine-tuned their models but did not [fine-tune the baseline] … they did that fine tuning on
> theirs, but not on the baseline model … so they didn't really even help the baseline."

The group abandoned it within about five minutes and switched to reading a survey article in
silence. That decision is why the discussion that followed ranged so widely: nobody was anchored to
a single paper.

---

## The five threads that matter

### 1. Somebody opened a playground, and the room understood the point immediately

Partway through, one participant pulled up an interactive demonstration of a machine-learned
interatomic potential in the browser and the table played with it for several minutes. The reactions
were physical rather than analytical — the atoms "look like bowling balls", you can throw things at
the structure, it responds in real time. Then the sentence that did the actual work:

> "Oh, cool. So it's essentially an approximated DFT."

That is the correct one-line summary of the entire field, arrived at by direct manipulation. The
group also noticed, with some surprise, that the team behind it was small — "not that many, eight
people or something" — and that the organisation responsible was a social-media company.

Worth recording precisely: **the model was never named in the room, and its architecture was never
discussed.** The demo functioned as an intuition pump for "quantum chemistry is expensive, this is
instantaneous" and nothing further. Identifying it as Meta's UMA is an editorial reconstruction,
not something anyone said.

### 2. Most of the room did not know what DFT was, and said so

This is the most useful thing in the recording, because it is the honest starting point of any
interdisciplinary group. Asked who was familiar with density functional theory, the answers were
"I used it a long time ago, [in] grad school" and "it's a field that I know nothing about."

The novice's question was the right one:

> "Just like PDEs, or … is it Schrödinger equation solving? I guess it's like an approximation of
> … [the] quantum Schrödinger equation. Like, what makes it hard? How scalable is it?"

The answer given was "it scales [as] $n$ cubed, [where] $n$ is the number of particles" — roughly
right for Kohn–Sham DFT. The thread reopened thirty minutes later with the same question, and this
time reached: "Instead of solving the full [many-body] electronic [problem, which] would be
literally impossible, you make this [density] approximation."

So the room converged on a correct-if-shallow model: DFT is a tractable approximation to the
many-electron problem, it scales cubically, and it is itself approximate. Nobody could speak to
exchange–correlation functionals, basis sets, or the regimes where DFT is known to fail — which
matters for the next thread.

### 3. The chromium dimer, and a story nobody could check

The materials thread was opened with an anecdote:

> "Two chromium atoms — if you want to figure out how much force [is] between them, you have to
> measure it. You couldn't actually predict it."

and its punchline, which is what actually motivated the room:

> "The point is, these are so expensive, even for just two atoms, we often can't afford to calculate
> the force between them."

Two things should be said about this. Rhetorically it worked perfectly: it gave a table of
non-chemists a reason to care about learned potentials. Technically, as told, it conflates two
different difficulties. Cr₂ is the textbook *multireference* system, and DFT's problem there is a
qualitative accuracy failure arising from strong static correlation, not merely a cost problem;
recent progress came from quantum Monte Carlo and DMRG-class methods rather than from making DFT
cheaper. The transcript's "efficiently simulate DFT" elides that distinction.

Nobody challenged it — precisely because, per the previous thread, nobody in the room had the
background to. This is a small, clean example of how a group's expertise gaps determine which claims
survive a discussion.

### 4. Symmetry — taught well, and never connected to architecture

The best-executed segment of the meeting was a live tutorial on crystallographic symmetry:

> "There's 230 [space] groups. So in 2D, there is only 17. So these are called the wallpaper groups."

> "If you look at what transformation you can do to keep this the same — so you can translate it, or
> flip it, or rotate it, or you can [do a] glide reflection, you can move [it] one and flip it. So
> actually there's only 17 total."

> "In 3D there's only 230 … so every crystal is one of those 230 groups."

The room worked through examples on screen, someone asked why a pentagon does not appear, and
another participant supplied the right intuition — "every rotation you do there, then it's like,
how can you repeat exactly?" Someone pulled up a reaction–diffusion demonstration that imposes a
wallpaper group and then evolves the PDE, so every frame respects the symmetry, and the table played
"guess the group". It closed on Conway's book and the observation that all seventeen appear in the
Alhambra.

**And it never crossed over.** Symmetry was discussed as a property of crystals and as a generative
constraint on wallpaper. It was never discussed as a constraint on a network architecture. The word
*equivariance* does not appear anywhere in the recording. The nearest approach is a single sentence
ninety lines earlier, about the leaderboard, which is the reason this entire report exists:

> "It's called Graph Attention Transformer that is now the leading one … which incorporates
> rotational symmetry."

Two hours of group theory sat in the same room as that sentence and the connection was never made.
That gap is exactly what [the tutorial](../tutorial/index.html) and
[the genealogy report](02-genealogy.md) are for.

### 5. Where machine learning is actually worth it

The sharpest analytical remark of the morning, and the one that best justifies the group's interest
in this area at all:

> "Linear equations are actually kind of almost too simple for neural networks. You have existing
> [classical] solvers which are really fast — but this DFT: [there,] the existing solvers are too
> slow."

That is a clean statement of the value criterion for scientific machine learning. The payoff from a
learned surrogate is set by **the gap the incumbent solver leaves**, not by the difficulty of the
underlying physics. Linear heat diffusion has fast, reliable classical solvers, so there is no room;
DFT does not, so there is. The remark was made from first principles, without reference to neural
operators or any of the surrounding literature.

---

## What the group got right without the vocabulary

Three of the group's instincts turn out to be the standard critiques of this field, arrived at
independently.

**Class imbalance makes accuracy meaningless.** Presented with a stability-classification accuracy
above ninety per cent, the immediate response was scepticism — "that sounds [dubious]" — followed by
the correct methodological fix: "if it's [im]balanced, then F1 [is] maybe a better score." This is
precisely why Matbench Discovery reports F1 and the discovery acceleration factor rather than
accuracy. [Chapter 8](../tutorial/index.html#hull) makes the point quantitative.

**The candidate set was doing suspicious work.** On a set of roughly 200,000 "potentially stable"
structures, the deflating observation was that it was "just like very individual **ionic
substitutions**". That is an accurate description of how the WBM test set is generated — by
elemental substitution into known prototypes — and the concern that this is enumeration rather than
discovery is one of the benchmark's genuinely open criticisms. See
[the leaderboard report](03-leaderboard.md).

**Thermodynamic stability is not existence.** The recurring worry was "the odds of coming out with
[something] physically impossible". The group grounded this in synthesis: one participant described
the Berkeley autonomous laboratory, and then the caveat that undercuts stability-as-a-target —
"what goes into making sure it crystallizes one way or [another] — [it's] random things: temperature,
[pressure] schedules, air pressure." Another, from undergraduate materials science: "we did a lot of
phase transition diagrams — so it's what temperature [and] pressure you're at." The conclusion was
that a useful model "would also give you … recipes."

The distinction they were circling — between compositional validity, low predicted energy,
thermodynamic stability, phonon stability, kinetic accessibility, and actual synthesisability — is
real, and those are not interchangeable notions.

---

## The disagreement that stayed open

The liveliest exchange was about how to condition a generative model for materials. One participant
had read a generative-materials paper and found the conditioning signal weak — it was "a textual
description from a chemistry book" — and argued for physics instead:

> "I would expect that you'd condition on a vector of property values … not text."
>
> "In the end, they're still [a] set of physical properties that are measurable, so why not just go
> straight there?"

The counterargument, the bitter-lesson one, was raised in the same breath and then picked up by
others: "Everything gets encoded in the end … I'm thinking of image generation, which is
conditioning on a vector. You could give a description [and] the machine will automatically
classify" — and fine-tuning a hand-specified property vector may end up worse than letting a
sufficiently capable model do its own classification from text.

A third contribution went to mechanism, and is the most technically substantive thing anyone said
about generation:

> "The natural conditioning: if you have an encoding over [a] crystal, just fix one part of the
> encoding and sample from the rest. … there's like five ways to encode a crystal [in] that overview
> paper. So once you have the encoding … if you have the **Wyckoff positions**, you could encode
> 'I want two of these atoms', and then you sample the rest."

The thread converged not on an answer but on a well-posed problem:

> "The more likely thing is 'I want a superconductor, I don't care [about the rest]'. So that's the
> interesting technical part: how do you attach this extra thing to your generative model, and then
> how do you condition on that?"

---

## Where the energy dropped

About sixty-eight minutes in, after a great deal of material:

> "None of the papers that we touched today were [ones where] I learned a new type of model … it's
> not going to give you a new architecture, probably."

paired with a crowded-field worry — "it's sort of like AlphaFold, where everyone's going [after] it"
— and a statement of what the speaker actually wanted: "what I'm very interested in, I really enjoy
… ways of encoding structures."

That is a precise diagnosis, and it is the reason this report is organised the way it is. The group
had been reading *applications*. What they wanted was the *mechanism*. NequIP is the paper where the
mechanism is visible, and where "ways of encoding structures" is the entire subject.

---

## The one concrete research proposal

Worth recording because it is genuinely open:

> "Real properties are not just about whether it's [stable] … but rather how they [change] over time
> — [like] batteries and their ability to change … I don't know if that's covered in any of these
> benchmarks, so maybe [we] need our own benchmark for that. That could be a useful contribution."

with a suggested technical route by analogy to generative video: "add a time [axis] to make video
instead of [images]." Existing benchmarks do evaluate molecular dynamics and thermal conductivity,
but a benchmark centred on *degradation and cycling over time* is not something the current
leaderboard measures.

---

## Reading order from here

The sentence about rotational symmetry is the thread this repository pulls.

1. [NequIP, read closely](01-nequip-anatomy.md) — the paper where the mechanism is visible.
2. [The genealogy of an idea](02-genealogy.md) — where "incorporates rotational symmetry" came from,
   and what has happened to it since.
3. [Reading the leaderboard](03-leaderboard.md) — what the benchmark actually measures, and why
   ranking by different metrics produces different winners.
4. [The clickable tutorial](../tutorial/index.html) — ten chapters, each with a toy you can break.
