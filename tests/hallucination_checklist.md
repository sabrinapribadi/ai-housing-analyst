# Hallucination Evaluation Checklist

## For Each Agent Response, Check:

### 1. Factual Accuracy
- [ ] Are prices reported correctly (within 10% of actual)?
- [ ] Are neighborhood names spelled correctly?
- [ ] Are statistics (counts, averages) accurate?
- [ ] Are dates and timeframes correct?

### 2. Data Presence
- [ ] Does the agent claim data exists when it doesn't?
- [ ] Does the agent claim data is missing when it's available?
- [ ] Are tool outputs correctly interpreted?

### 3. Reasoning Quality
- [ ] Are conclusions logically derived from data?
- [ ] Are correlations correctly explained?
- [ ] Are recommendations based on actual data?

### 4. Confidence Calibration
- [ ] Is the agent overconfident about uncertain claims?
- [ ] Does the agent hedge when data is limited?
- [ ] Is uncertainty clearly communicated?

## Test Questions to Run:

| Question | Expected Answer | Agent Response | Hallucination? |
|----------|----------------|----------------|----------------|
| Average price | ¥24,782 | | |
| Most expensive neighborhood | Chuo Ku (¥44,342) | | |
| Cheapest neighborhood | Higashiyamato Shi (¥5,136) | | |
| Total listings | 27,945 | | |
| Most common room type | Entire home/apt (23,759) | | |
