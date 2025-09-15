# PRD: Customer Experience Insight Discovery  
## Project: Disneyland Review Analysis  

---

## 1. Background  
Visitor reviews at Disneyland contain valuable, often unstructured feedback. Analyzing these reviews can reveal key pain points and drive meaningful improvements to the guest experience.

---

## 2. Objective  
Produce a one-time analysis of 1,000 reviews to uncover **actionable insights** that help the **Customer Experience team** enhance satisfaction and service quality.

---

## 3. Stakeholders

| Role                    | Responsibility                      |
|-------------------------|--------------------------------------|
| Customer Experience Team | Consume insights and act on them     |
| Data Scientist (you)     | Execute analysis and summarize findings |

---

## 4. User Needs

- Identify drivers of low ratings and negative sentiment  
- Understand patterns by time or reviewer location  
- Receive a concise, quote-supported summary with actionable recommendations  

---

## 5. Requirements

### 5.1 Functional
- Process and clean review data  
- Translate non-English text  
- Extract sentiment, theme, and topic via LLM  
- Analyze trends across time, sentiment, theme, and location  
- Deliver executive summary  

### 5.2 Non-Functional

| Area         | Requirement                          |
|--------------|---------------------------------------|
| Timeline     | Must complete in a few hours          |
| Output       | Executive summary (Markdown/PDF)      |
| Language     | Translate to English if needed        |
| LLM Usage    | Use OpenAI `gpt-4o-mini`              |
| Reproducibility | Save enriched data to CSV           |

---

## 6. Success Criteria

| Goal                  | Metric                                      |
|-----------------------|---------------------------------------------|
| Insight Quality       | 3–5 high-impact recommendations             |
| Report Clarity        | Business-friendly summary with examples     |
| Reuse of Enriched Data| Saved, structured CSV with new features     |

---

## 7. Risks and Mitigations

| Risk                     | Mitigation                          |
|--------------------------|--------------------------------------|
| LLM overhead or slowness | Cache responses; avoid recomputation |
| Language noise           | Auto-translate where necessary       |
| Misclassification        | Manually check sample outputs        |
