Roulette Prediction Model
============================================================================================================================
This repository contains a Python script for predicting roulette outcomes based on a RandomForestClassifier model. The model is trained on a dataset provided in the file roulette_predictions.csv.


Requirements:
------------------------------------------------------

Python 3
- NumPy
- pandas
- scikit-learn
- Install the required dependencies using:

Bash
- pip install numpy pandas scikit-learn

Usage
------------------------------------------------------

Ensure you have the required dependencies installed.
Download or clone the repository.
Bash
- git clone https://github.com/your-username/roulette-prediction.git
- cd roulette-prediction

Run the script:
------------------------------------------------------
Bash
- python roulette_prediction.py
- Follow the on-screen prompts to input the starting and final numbers. Type 'exit' to end the prediction.

Model Details:
------------------------------------------------------
The script uses a RandomForestClassifier with 50,000 estimators to predict the likely outcomes. The trained model is saved to the file roulette_predictions.csv. The features used for prediction are the starting numbers, and the target is the corresponding final number.

Additional Functions:
------------------------------------------------------
- roulette_color(number): Determines the color (red, black, or green) of a given roulette number.
- roulette_third(number): Identifies the third (1st, 2nd, or 3rd) to which a roulette number belongs.
- predict_top_3_numbers_with_probabilities(model, data_row): Predicts the top 3 most likely numbers and their probabilities.
- calculate_color_third_probabilities(indices, probabilities): Calculates color and third probabilities for the predicted numbers.

Contributions:
------------------------------------------------------
Feel free to contribute by providing enhancements, fixing issues, or expanding the capabilities of the prediction model. Create a pull request, and let's make this roulette prediction script even more accurate and versatile!

Please just cite that you got your data and base code from me!
------------------------------------------------------







