# DSM500 CW2 Video Script

**Spoken text:** 505 words; about 3.9 minutes at 130 words per minute, plus brief slide transitions. Rehearse once and keep the recording under five minutes.
**Slides:** `video/presentation.html`. Press → or Space at each advance cue. Press **N** to show notes.
**Presenter cues:** quoted instructions are for you and are not spoken.

## Slide 1: Title

> START — Begin recording with slide 1 visible. Do not read cues aloud.

Hello, I’m Guillermo Olmos Ranalli. My project, Synthetic Markets, Real Decisions, asks whether synthetic financial data can help us judge trading strategies more reliably.

> ADVANCE TO SLIDE 2 — The problem. Press → or Space, then continue below.

## Slide 2: The problem

A backtest measures a strategy on one historical path. If we try enough rules, some will look successful by chance. Synthetic data offers alternative market histories, but realistic looking data is not automatically useful for a decision. My project evaluates generators by whether their strategy rankings agree with performance on later real data.

> ADVANCE TO SLIDE 3 — Question and approach. Press → or Space, then continue below.

## Slide 3: Question and approach

The main question is whether a rule’s ranking on synthetic data predicts its ranking on unseen market data. I measure this with Spearman’s rank correlation. The development period is 2018 to 2021, and the held-out period is 2022 to 2024. Yahoo’s currency series is USD/EUR: euros per US dollar. The long-only rules therefore take dollar exposure against the euro.

> ADVANCE TO SLIDE 4 — Study design. Press → or Space, then continue below.

## Slide 4: Study design

I compare GARCH, a window bootstrap, an MLP GAN and an LSTM GAN, with one thousand synthetic paths each. A historical validation comparison uses the earlier real period directly. The corrected evaluation has 500 uniquely named rules, including 480 moving-average variants. Defined real scores are available for 488 rules, and 484 rules enter the comparison shared by all methods.

> ADVANCE TO SLIDE 5 — What the paths show. Press → or Space, then continue below.

## Slide 5: What the paths show

The bootstrap reproduces many training-data statistics closely. GARCH is now calibrated to the currency’s return variance and generates clustered volatility. The saved MLP GAN produces returns with much less variation than real data. The saved LSTM GAN introduces strong autocorrelation and extreme returns. These are the existing trained models, so the study evaluates those checkpoints rather than newly trained architectures.

> ADVANCE TO SLIDE 6 — Main result. Press → or Space, then continue below.

## Slide 6: Main result

The corrected results are a negative finding. GARCH’s rank correlation is about 0.04, the LSTM GAN about 0.03, and the MLP GAN about minus 0.02. None shows reliable positive ranking performance. Window bootstrap is about minus 0.31, and historical validation about minus 0.29. Their rankings are negatively associated with later performance. These results replace the earlier pilot numbers, which were affected by evaluation defects.

> ADVANCE TO SLIDE 7 — Scope and checks. Press → or Space, then continue below.

## Slide 7: Scope and checks

Paired resampling compares the same 484 rules. It does not clearly separate GARCH from either GAN. These methods compare more favourably with the negatively ranked historical and bootstrap baselines, but that does not establish useful positive prediction. Closely related rules make uncertainty estimates optimistic. Matching historical statistics also does not guarantee useful strategy selection.

> ADVANCE TO SLIDE 8 — Limitations. Press → or Space, then continue below.

## Slide 8: Limitations

The rerun corrects trade accounting, daily strategy returns, duplicated names and the GARCH variance recursion. Remaining limitations include pooled, zero-filled GAN training, one currency series, one held-out period and a rule set dominated by moving-average variants. Transaction costs and repeated training seeds are not evaluated. A synthetic market with known ground truth would provide a stronger control. Those limits prevent wider claims about GAN architectures or profitable trading.

> ADVANCE TO SLIDE 9 — Conclusion. Press → or Space, then continue below.

## Slide 9: Conclusion

The project provides a reproducible way to assess synthetic data through both statistical fidelity and downstream decisions. In this corrected study, no generator demonstrates reliable positive ranking utility. The negative result shows why a plausible synthetic market must still be tested against the decision it is meant to support. Broader rules, additional periods and better trained models are the next research steps. Thank you.

> END — Hold the conclusion slide briefly, then stop recording.
