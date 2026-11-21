# Supplementary calculations for the poster

*Early detection of Parkinson's*, NEURO Group 5, 27020 Interdisciplinary Bioengineering, DTU, Fall 2026.

This page shows every step behind the numbers in the poster's **KPIs and conclusion** box:
the inputs, where each one comes from, the formula and the result. Model results come from
our own analysis in this repository (see the [README](README.md)). Reference numbers in
square brackets match the poster.

---

## 1. Screening: what a positive result would mean

**Poster text:** *If 10,000 Danes aged 70–79 used the app, 160 would have Parkinson's [9].
The app would flag 154 of them and 1,850 healthy people, so 1 in 13 flagged has Parkinson's.*

### Inputs

| Input | Value | Source |
|---|---|---|
| Group screened | 10,000 people aged 70–79 | Chosen for illustration |
| Parkinson's prevalence, age 70–79 | 1,601 per 100,000 (1.6%) | Pringsheim et al. 2014 [9], pooled estimate for Europe, North America and Australia |
| Share of Parkinson's found (sensitivity) | 96.4% (53 of 55) | Our test set, model "watch + questions" |
| Share of healthy wrongly flagged (false-alarm rate) | 18.8% (3 of 16) | Our test set, model "watch + questions" |

### Calculation

```
People with Parkinson's      = 10,000 × 0.01601            = 160
Healthy people               = 10,000 − 160                = 9,840
Parkinson's found            = 160 × 0.964                 = 154
Healthy people flagged       = 9,840 × 0.188               = 1,850
All people flagged           = 154 + 1,850                 = 2,004
Share of flagged who are ill = 154 / 2,004                 = 7.7 %   (about 1 in 13)
```

### How sensitive is this to our assumptions?

| Scenario | Prevalence | False-alarm rate | Share of flagged who are ill |
|---|---|---|---|
| Poster (Europe, age 70–79) | 1.6% | 18.8% | 7.7% (1 in 13) |
| Worldwide average, age 70–79 [9] | 1.1% | 18.8% | 5.3% (1 in 19) |
| Age 60–69 [9] | 0.43% | 18.8% | 2.2% (1 in 46) |
| Watch only (no questions) | 1.6% | 25.0% | 5.1% (1 in 20) |
| Best case of our 95% interval for false alarms | 1.6% | 4% | 28% (about 1 in 4) |
| Worst case of our 95% interval for false alarms | 1.6% | 46% | 3.3% (1 in 30) |

The false-alarm rate is the least certain input, because only 16 healthy people were in
our test set (95% interval 4–46%, see the README). The conclusion that the app works as a
**first filter before a doctor's check, not as a diagnosis**, holds across all scenarios.

### Limits of this calculation

- Prevalence counts everyone who has Parkinson's, including people who are already
  diagnosed. A real screening programme would mostly look for undiagnosed cases, which are
  fewer, so the share of flagged people who are ill would be lower.
- Everyone with Parkinson's in our data was already diagnosed (on average about 9 years
  earlier), so the 96.4% may be lower for early, undiagnosed disease.

---

## 2. Scale: do older Danes have the hardware?

**Poster text (conclusion):** *feasible today on the smartphones most older Danes already use [10].*

| Age group | Use a mobile phone or smartphone to go online |
|---|---|
| 15–69 | over 90% |
| 70–74 | 81% |
| 75–79 | 72% |
| 80–84 | 62% |
| 85–89 | 54% |

Source: Ældre Sagen, *Ældres anvendelse af IT i 2024* (2025) [10], figure 3. The question
was asked to people who had been online in the last three months, so the shares are of
internet users, not of the whole age group. The app needs no new hardware for most people
in the target group. The oldest age groups are the weakest point.

---

## 3. Cost: what it costs to run

**Poster text:** *about 4.7 million DKK a year to run, paid by municipalities, not users.*

The app is free for users. Municipalities, the state or research projects pay for running
the company. Only the salary level has an external source. The other lines are our own
estimates.

| Item | Assumption | DKK per year |
|---|---|---|
| Staff: 2 app developers, 1 data scientist, 1 for regulation and quality | 65,000 DKK per month in salary, about the average for IDA members in the private sector (61,912 women, 67,878 men) [IDA 2024], plus 20% for pension and other employer costs: 65,000 × 12 × 1.2 = 936,000 per person | 3,744,000 |
| Approval and quality system | A screening app is most likely a medical device in the EU (MDR, Rule 11, at least class IIa). That means CE marking through a notified body and yearly audits of the quality system | 500,000 |
| Servers and data security | The analysis runs on the phone, so only small summaries are stored centrally | 200,000 |
| Other | Insurance, legal advice, office | 300,000 |
| **Total** | | **about 4,700,000** |

### Per user and in context

```
Per user per year at 100,000 users = 4,744,000 / 100,000 = 47 DKK
Per user per year at  10,000 users = 4,744,000 /  10,000 = 474 DKK
```

For scale only, not as a claim of savings: 4.7 million DKK is about 680,000 USD. The
economic burden of Parkinson's in the US is 82.2 billion USD a year for about 1.2 million
people [3], which is about 68,500 USD per person. One year of running the company costs about
the same as the yearly burden of 10 people with Parkinson's.

---

## References

Numbers match the poster's reference list.

- [3] The Lewin Group. *Economic burden of Parkinson's and atypical parkinsonism in the
  United States.* New York: Michael J. Fox Foundation; 2026.
- [9] Pringsheim T, Jette N, Frolkis A, Steeves TDL. The prevalence of Parkinson's disease:
  a systematic review and meta-analysis. *Mov Disord.* 2014;29(13):1583–1590.
- [10] Ældre Sagen. *Ældres anvendelse af IT i 2024.* Copenhagen: Ældre Sagen; 2025.
- Not on the poster: IDA. *Køn og løn 2024.* Copenhagen: Ingeniørforeningen IDA; 2024.
- Not on the poster: Regulation (EU) 2017/745 on medical devices (MDR), Annex VIII, Rule 11.
