# DSM500 CW2 Video Script

**Spoken text:** 294 words. Allow roughly four minutes at 90 words per minute, including short slide transitions. Rehearse with a timer and finish under five minutes at your natural speaking pace.
**Slides:** `DSM500-Presentation-200153474_presentation.html` in the repository root. Press → or Space at each advance cue.
**Reading:** use this shortened script; the HTML presenter notes retain the longer version. Do not read headings or cues aloud.

## Slide 1: Title

> START — Begin recording with slide 1 visible.

Hello, I’m Guillermo Olmos Ranalli. Synthetic Markets, Real Decisions asks whether synthetic financial data can help us choose trading strategies.

> ADVANCE TO SLIDE 2

## Slide 2: The problem

Testing many strategies on one historical market can produce lucky winners. Synthetic data offers alternative histories. But matching the appearance of real markets does not guarantee useful decisions. That distinction is the focus of this project.

> ADVANCE TO SLIDE 3

## Slide 3: Question and approach

Does a strategy’s ranking on synthetic data predict its ranking on later real data? I measure agreement using Spearman correlation, with USD/EUR data from 2018 to 2021 for development and 2022 to 2024 for testing.

> ADVANCE TO SLIDE 4

## Slide 4: Study design

I compare GARCH, window bootstrap and two saved GAN models, alongside historical validation. Each generator supplies one thousand paths. The evaluation uses 500 unique rules, mainly moving averages; 484 have defined scores across every method.

> ADVANCE TO SLIDE 5

## Slide 5: What the paths show

Bootstrap closely reproduces historical statistics, and GARCH captures clustered volatility. The saved GANs have fidelity problems: the MLP produces too little variation, while the LSTM generates excessive autocorrelation and extreme returns.

> ADVANCE TO SLIDE 6

## Slide 6: Main result

No generator demonstrates reliable positive ranking utility. GARCH’s correlation is about 0.04, LSTM about 0.03 and MLP about minus 0.02. Bootstrap is about minus 0.31, and historical validation minus 0.29. Matching history did not preserve later strategy rankings.

> ADVANCE TO SLIDE 7

## Slide 7: Scope and checks

Comparisons using the same rules do not clearly separate GARCH from either GAN. Closely related rules may also make uncertainty intervals too narrow. These results support caution, rather than a claim that any method predicts successful trading.

> ADVANCE TO SLIDE 8

## Slide 8: Limitations

The study covers one currency and one test period. The GAN checkpoints retain pooled, zero-filled training. Transaction costs and repeated training seeds are not evaluated, so conclusions apply to this experiment.

> ADVANCE TO SLIDE 9

## Slide 9: Conclusion

The main takeaway is simple: realistic synthetic data must still be tested against the decision it supports. Better trained models, broader rules and additional periods are the next steps. Thank you.

> END — Hold the final slide briefly, then stop recording.
