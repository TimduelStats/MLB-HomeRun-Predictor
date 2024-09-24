# MLB Home Run Predictor

This project uses an XGBoost machine learning model to predict MLB home runs based on batter statistics, pitcher matchups, and recent performance data. The prediction process runs within an AWS Lambda function, which is deployed using Docker and AWS CDK. The results are stored in Amazon S3.

## Key Features
- **Home Run Prediction:** Uses XGBoost to predict home runs for MLB games.
- **AWS Lambda & Docker:** The prediction function runs in a containerized environment on AWS Lambda.
