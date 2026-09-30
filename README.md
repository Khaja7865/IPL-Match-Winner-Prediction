# IPL Match Winner Prediction Using Machine Learning

## Project Overview

This project develops an end-to-end Machine Learning system to predict the winner of an IPL cricket match using historical IPL match data.

The project analyzes historical match information and creates meaningful features such as team win rate, recent form, Elo rating, head-to-head performance, and toss advantage.

Three Machine Learning classification algorithms are trained, evaluated, and tuned to identify the best-performing model.

---

## Problem Statement

Predict the likely winner of an IPL match based on historical match information, team performance, recent form, head-to-head records, Elo ratings, venue, and toss-related information.

The objective is to develop a complete Machine Learning pipeline from data analysis to final prediction.

---

## Objectives

- Understand and analyze historical IPL match data.
- Perform statistical analysis and Exploratory Data Analysis (EDA).
- Handle missing and categorical data.
- Create meaningful features for prediction.
- Split the dataset into training and testing data.
- Train multiple Machine Learning classification models.
- Evaluate model performance using different metrics.
- Perform hyperparameter tuning using GridSearchCV.
- Select the best-performing model.
- Save the final trained model.
- Demonstrate prediction on an unseen test match.

---

## Dataset

The project uses historical IPL match data containing information such as:

- Season
- Date
- City
- Venue
- Team 1
- Team 2
- Toss Winner
- Toss Decision
- Match Winner

Dataset file:

```text
Dataset/matches.csv