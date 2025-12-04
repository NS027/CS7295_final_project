# Human Evaluation Handbook (A/B Study)

## Off-the-Shelf LLM vs. Our Visualization Pipeline

------------------------------------------------------------------------

## 1. Purpose of This Study

You will compare: - **Condition A:** An off-the-shelf Large Language
Model (LLM) such as ChatGPT or Claude - **Condition B:** Our custom
Question--Visualization pipeline (Streamlit App)

You will evaluate the **overall quality of questions and
visualizations** generated under each condition.

------------------------------------------------------------------------

## 2. Dataset

All participants must use the **same penguins dataset** provided by the
organizer. Do not modify, clean, or filter the dataset manually.

------------------------------------------------------------------------

## 3. Condition A -- Baseline LLM (Use This Prompt Exactly)

### ✅ Baseline Prompt (Copy--Paste)

You are a senior data analyst.

Generate 5 questions that would help me better understand this dataset.

For each question, also generate a corresponding data visualization image to improve understanding.

------------------------------------------------------------------------

## 4. Condition B -- Our Pipeline

1.  Open the Streamlit application.
2.  Load the same penguins dataset.
3.  Use the app to generate approximately **five Question--Visualization
    outputs**.
4.  Observe:
    -   The generated questions
    -   The actual charts displayed in the interface

------------------------------------------------------------------------

## 5. Scoring Procedure

After you review all outputs:

-   Give **one overall score for Condition A**
-   Give **one overall score for Condition B**

Use the **same 8 evaluation dimensions** for both conditions. Each
dimension is scored from **1 (Poor) to 5 (Excellent)**.

------------------------------------------------------------------------

## 6. Evaluation Dimensions and Scoring Anchors (1--5 Scale)

### 1. Question Clarity

Measures whether the question is clearly stated and unambiguous. - 1 =
Very unclear or confusing - 2 = Somewhat unclear - 3 = Generally clear,
but imprecise - 4 = Clear and well stated - 5 = Very clear, precise, and
unambiguous

### 2. Analytical Depth

Measures whether the question reflects meaningful analytical thinking. -
1 = Purely descriptive or trivial - 2 = Very limited analytical value -
3 = Moderate analytical value - 4 = Strong analytical intent - 5 =
Expert-level analytical depth

### 3. Data Answerability

Measures whether the question can be answered using the given dataset. -
1 = Cannot be answered with the dataset - 2 = Heavily dependent on
assumptions or external data - 3 = Partially answerable - 4 = Mostly
answerable - 5 = Clearly and fully answerable

### 4. Visualization Appropriateness

Measures whether the selected chart type is suitable for the question
and data. - 1 = Completely inappropriate - 2 = Poor choice - 3 =
Acceptable but not ideal - 4 = Appropriate - 5 = Optimal and
professional choice

### 5. Encoding Correctness

Measures whether fields are correctly mapped to visual channels. - 1 =
Major encoding errors - 2 = Several incorrect mappings - 3 = Mostly
correct with minor issues - 4 = Correct with clear logic - 5 = Fully
correct and rigorous mapping

### 6. Readability & Visual Design

Measures whether the visualization is easy to read and interpret. - 1 =
Very difficult to read - 2 = Difficult to interpret - 3 = Acceptable
readability - 4 = Clear and readable - 5 = Very clear and intuitive

For Condition A, judge based on the clarity of the visualization
description. For Condition B, judge based on the actual rendered chart.

### 7. Question--Visualization Alignment

Measures whether the visualization truly answers the question. - 1 = No
alignment - 2 = Very weak alignment - 3 = Partial alignment - 4 = Strong
alignment - 5 = Perfect alignment

### 8. Overall Usefulness

Measures the practical value of the outputs in a real analysis
scenario. - 1 = Not useful at all - 2 = Low practical value - 3 =
Moderately useful - 4 = Useful in practice - 5 = Highly useful for real
decision-making

------------------------------------------------------------------------

## 7. Important Rules

-   Do NOT discuss your ratings with others.
-   Score A and B independently and fairly.
-   Do NOT attempt to infer the study's expected outcome.
-   Base your scores only on what you observe.

------------------------------------------------------------------------

## 8. Submission

Please complete the provided CSV scoring sheet and return it to the
organizer.

Thank you for your participation.

---
## 9. Dataset

We evaluate our system using three real-world, small-to-medium scale datasets to ensure diversity in data characteristics while maintaining comparable data volume.

1. **Palmer Penguins Dataset**  
   https://github.com/allisonhorst/palmerpenguins  
   A biological measurement dataset containing morphological measurements of penguins across different species, islands, and years.

2. **Iris Dataset**  
   https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv  
   A classic multivariate dataset containing sepal and petal measurements of three species of iris flowers.

3. **Tips Dataset (Restaurant Tips)**  
   https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv  
   A behavioral dataset containing restaurant bills and gratuities with categorical attributes such as sex, smoking status, day, and time.