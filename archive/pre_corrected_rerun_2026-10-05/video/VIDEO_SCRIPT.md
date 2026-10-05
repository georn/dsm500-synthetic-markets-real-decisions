# DSM500 CW2 Video Script

**Spoken text:** 558 words. Allow about 4.3 minutes at 130 words per minute, plus pauses. Keep the recording under five minutes and use your own voice.
**Slides:** `video/presentation.html`. Press → or Space at each advance cue; press **N** to show these notes.
**Presenter cues:** quoted lines are instructions for you and are not spoken. Timing guides assume roughly 130 words per minute plus a two-second transition per slide; confirm with a timed rehearsal.

## Slide 1: Title (0:00–0:13)

> START ON SLIDE 1 — Title. Begin recording with this slide visible. Do not read presenter cues aloud.

Hello, I’m Guillermo Olmos Ranalli. My project, Synthetic Markets, Real Decisions, asks whether synthetic financial data can help us judge trading strategies more reliably.

> ADVANCE TO SLIDE 2 — The problem. Press → or Space after the paragraph above, then continue below.

## Slide 2: The problem (0:13–0:44)

A backtest measures a strategy on one historical path. If we try enough rules on that path, some will look successful by chance. That is backtest overfitting. Synthetic data offers additional market paths to test. But realistic looking data is not automatically useful for a decision. This project evaluates generators by whether their strategy rankings agree with performance on later real data.

> ADVANCE TO SLIDE 3 — Question and approach. Press → or Space after the paragraph above, then continue below.

## Slide 3: Question and approach (0:44–1:14)

The main question is: does a rule’s ranking on synthetic data predict its ranking on unseen market data? I compare the rankings using Spearman’s rank correlation. Generators use daily data from 2018 to 2021; the held-out period is 2022 to 2024.

Yahoo’s currency series is USD/EUR: euros per US dollar. The long-only rules therefore take dollar exposure against the euro.

> ADVANCE TO SLIDE 4 — Study design. Press → or Space after the paragraph above, then continue below.

## Slide 4: Study design (1:14–1:43)

The report compares four generators: a GARCH volatility model, a window bootstrap, an MLP GAN and an LSTM GAN. Each supplies one thousand synthetic paths. The rule set began with 500 specifications, but shared names caused some to be merged. The analysis has 185 distinct rules, 180 of them moving-average crossovers. The GANs have defined scores for 181 rules.

> ADVANCE TO SLIDE 5 — What the paths show. Press → or Space after the paragraph above, then continue below.

## Slide 5: What the paths show (1:43–2:14)

The bootstrap resamples observed windows. GARCH generates new paths with clustered volatility, though its parameters were not calibrated to this currency series. The MLP GAN paths are much less volatile than the real data, and some LSTM paths collapse toward straight lines. These checks matter when interpreting the rankings: the deep generators were not well matched to the data in this run.

> ADVANCE TO SLIDE 6 — Main result. Press → or Space after the paragraph above, then continue below.

## Slide 6: Main result (2:14–2:59)

GARCH has the highest rank correlation, about 0.54. The window bootstrap is about 0.34, the LSTM GAN about 0.30, and the MLP GAN about 0.19. All four correlations are positive, so each contains some ranking information in this pilot.

A paired resampling comparison uses the 181 rules common to all four generators. It puts GARCH ahead of each other generator; the other three cannot be separated by that test. This supports a limited conclusion: here, this simple classical model ranked these mostly moving-average rules better than the tested GANs.

> ADVANCE TO SLIDE 7 — Scope and checks. Press → or Space after the paragraph above, then continue below.

## Slide 7: Scope and checks (2:59–3:30)

Moving-average crossovers account for 180 of the 185 distinct rules. There are too few momentum and RSI rules for meaningful correlations. Many rules are close variants, so the resampling intervals may be too narrow. The report also finds that closely matching the training data’s moments did not guarantee the best strategy ranking. These findings apply to this currency series and test period.

> ADVANCE TO SLIDE 8 — Limitations. Press → or Space after the paragraph above, then continue below.

## Slide 8: Limitations (3:30–4:05)

The results are a pilot with important limitations. The as-run backtester calculated trade returns incorrectly, GARCH was not fitted to the currency series, and the invalid walk-forward comparison is excluded. The GANs were trained on pooled, zero-filled data and produced poor-quality paths. These issues were found in final review. The corrected pipeline has tests, but the reported results have not been regenerated with it. The ranking could change after a rerun.

> ADVANCE TO SLIDE 9 — Conclusion. Press → or Space after the paragraph above, then continue below.

## Slide 9: Conclusion (4:05–4:37)

The project demonstrates a task-based way to evaluate synthetic market data: check its statistical fidelity and whether it supports useful decisions on unseen data. In this pilot, GARCH ranked the tested rules best, but the evidence is preliminary and specific to mostly moving-average rules on USD/EUR. A corrected rerun, better-trained generators and a broader rule set are needed before wider conclusions. Thank you.

> END — Keep the conclusion slide visible briefly, then stop recording.
