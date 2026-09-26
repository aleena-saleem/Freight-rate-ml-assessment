# Freight Rate Prediction

Machine learning solution for predicting freight load rates from shipment, route, equipment, distance, weight, market, and quote information.

## Overview

This project was developed for the Machine Learning Engineer assessment.

The development dataset contains 48,000 historical freight loads with a known `posted_rate`. The final validation dataset contains 12,000 loads without the target value. The objective is to train a regression model on the labeled development data and generate a predicted rate for every load in the validation dataset.

The final prediction file is:

`validation_predictions.csv`

The model used for the final predictions is CatBoost regression.

## Dataset

### Development data

The labeled dataset contains 48,000 rows covering January 2025 through October 2025.

The target variable is:

`posted_rate`

Input variables include:

- pickup
- delivery
- pickup_lat
- pickup_lon
- delivery_lat
- delivery_lon
- distance
- equipment
- weight
- date
- market_index
- quote_signal

### Validation data

The validation dataset contains 12,000 loads covering November 2025 and December 2025.

The validation file does not contain `posted_rate`, so the final 12,000 predictions cannot be evaluated locally against ground truth.

## Data Quality

The initial analysis identified:

- 300 missing values in `weight`
- 374 missing values in `market_index`
- No duplicate rows
- No missing target values
- 64 unique pickup locations
- 64 unique delivery locations
- Three equipment categories: Dry Van, Reefer, and Flatbed

Missing numerical values were handled during preprocessing using values calculated from the training data.

## Feature Engineering

The model uses the original shipment variables together with additional features derived from the available data.

Date features include:

- month
- day
- day of week
- day of year

A route feature was created from the pickup and delivery locations.

A logarithmic distance feature was also tested because freight rates showed a strong relationship with shipment distance.

The final feature set includes the original numerical and categorical variables together with the engineered date, route, and distance features.

## Model Selection

Three regression approaches were evaluated on the temporal validation set:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Random Forest | $206.85 | $711.67 | 0.7832 |
| HistGradientBoosting | $156.29 | $660.39 | 0.8133 |
| CatBoost | $121.26 | $647.13 | 0.8208 |

CatBoost was selected for the final model because it produced the strongest validation performance among the tested models and handles categorical shipment information directly.

The final configuration used during model development included:

- depth: 8
- learning rate: 0.05
- iterations: 1500
- random seed: 42

## Validation Strategy

A temporal validation strategy was used because the final prediction period occurs after the labeled development period.

Instead of randomly mixing observations from different dates, earlier months were used for training and later months were used for validation.

### Validation 1

Training period:

January 2025 to August 2025

Validation period:

September 2025 to October 2025

Results:

- MAE: $112.27
- RMSE: $632.99
- R²: 0.8279

### Validation 2

Training period:

January 2025 to September 2025

Validation period:

October 2025

Results:

- MAE: $121.26
- RMSE: $647.13
- R²: 0.8208

These tests provide a check of how the model performs when predicting later observations using earlier historical data.

## Final Prediction

After model validation, the final model was trained using all 48,000 labeled development rows.

The trained model was then used to predict the 12,000 rows in `validation.csv`.

The resulting file contains exactly two columns:

```text
load_id,predicted_rate
The final prediction file contains 12,000 rows.

December Prediction Scenario

The assessment provides a fixed December scenario for visualization.

The fixed inputs are:

Pickup: Lexington
Delivery: Fort Wayne
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000 lb
Date: December 1 to December 31, 2025

Only the date changes between predictions.

The provided scoring script was used to validate these predictions and generate the December chart.

Output Validation

The provided scoring script successfully validated:

12,000 final validation predictions
31 December predictions
prediction IDs
prediction column names
prediction row count
positive prediction values
December dates and fixed input values

The generated chart is available at:

scorer_results/candidate_december.png

Project Structure
freight-rate-ml-assessment/
│
├── model.py
├── december_pred.py
├── temporal_validation.py
├── score.py
├── requirements.txt
├── README.md
│
├── train-test.csv
├── validation.csv
├── validation-predictions-template.csv
├── validation_predictions.csv
├── december_chart_inputs.csv
│
├── scorer_results/
│   └── candidate_december.png
│
└── freight-rate-ml-assessment.pdf
Running the Project

Create and activate the virtual environment, then install the dependencies:

pip install -r requirements.txt

Run the final model:

python model.py

Generate the December prediction inputs:

python december_pred.py

Validate the final outputs and generate the December chart:

python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv

A successful scoring run validates the 12,000 final predictions and the 31 December predictions.

Final Files

The main submission files are:

validation_predictions.csv

freight-rate-ml-assessment.pdf

scorer_results/candidate_december.png

The final prediction metrics for the 12,000 validation loads are calculated by the assessment platform because the ground truth posted_rate values for those loads are not provided.
