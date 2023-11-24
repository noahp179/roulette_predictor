import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

file_path = 'roulette_predictions.csv'

try:
    data = pd.read_csv(file_path, header=None)
except FileNotFoundError:
    print(f"File {file_path} not found. Please check the file path.")
    exit()

if data.empty:
    print("Data is empty. Please load the data correctly.")
    exit()

# Assuming the target is the actual number (second column)
y = data.iloc[:, 1].astype(int)
X = data.iloc[:, 0].values.reshape(-1, 1)

if len(data) > 1:
    model = RandomForestClassifier(n_estimators=50000, random_state=42)
    model.fit(X, y)
else:
    print("Not enough data to train the model.")
    model = None

def roulette_color(number):
    red_numbers = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
    return 'red' if number in red_numbers else ('green' if number == 0 else 'black')

def roulette_third(number):
    if 1 <= number <= 12:
        return '1st third'
    elif 13 <= number <= 24:
        return '2nd third'
    elif 25 <= number <= 36:
        return '3rd third'
    return 'None'

def predict_top_3_numbers_with_probabilities(model, data_row):
    if model:
        features = data_row.reshape(1, -1)
        probabilities = model.predict_proba(features)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_probabilities = probabilities[top_3_indices]
        return top_3_indices, top_3_probabilities
    return None, None

def calculate_color_third_probabilities(indices, probabilities):
    color_prob = {'red': 0, 'black': 0, 'green': 0}
    # Add '0th third' to your dictionary to handle the specific case
    third_prob = {'0th third': 0, '1st third': 0, '2nd third': 0, '3rd third': 0}

    for i, prob in zip(indices, probabilities):
        color = roulette_color(i)
        third = roulette_third(i)

        # Check if the index corresponds to '0th third'
        if i == 0:
            third = '0th third'

        color_prob[color] += prob
        third_prob[third] += prob

    return color_prob, third_prob

while True:
    starting_input = input("Enter the starting number (type 'exit' to end): ")
    if starting_input.lower() == 'exit':
        break

    starting_number = int(starting_input) if starting_input.isdigit() else 0

    if model:
        features = np.array([starting_number]).reshape(1, -1)
        top_3_indices, top_3_probabilities = predict_top_3_numbers_with_probabilities(model, features)
        
        if top_3_indices is not None:
            print(' ')
            print(f"Top 3 most likely numbers and their probabilities:")
            for i, prob in zip(top_3_indices, top_3_probabilities):
                color = roulette_color(i)
                third = roulette_third(i)
                print(f"Number: {i}, Probability: {prob:.2f}, Color: {color}, Third: {third}")

            color_prob, third_prob = calculate_color_third_probabilities(top_3_indices, top_3_probabilities)
            print('')
            print(f"Color Probabilities: {color_prob}")
            print(f"Third Probabilities: {third_prob}")

        print('')
        final_input = input("Enter the final number landed on: ")
        final_number = int(final_input) if final_input.isdigit() else 0
        
        final_input = int(final_input)
        if len(top_3_indices) >= 3 and (top_3_indices[0] == final_input or top_3_indices[1] == final_input or top_3_indices[2] == final_input):
            number_correct = 1
        else:
            number_correct = 0 # Placeholder for third correctness
        
        # Find the color and third with the highest probability
        highest_prob_color = max(color_prob, key=color_prob.get)
        highest_prob_third = max(third_prob, key=third_prob.get)

        # Check if the final number has the same color and third
        color_correct = 1 if roulette_color(final_number) == highest_prob_color else 0
        third_correct = 1 if roulette_third(final_number) == highest_prob_third else 0
        
        new_row = [starting_number, final_number] + list(top_3_indices)
        data.loc[len(data)] = new_row
        data.to_csv(file_path, index=False, header=False)
        
        new_results = [third_correct, color_correct, number_correct]
        
        with open('correct.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_results)
            print(     )
                
    else:
        print("Model not available for prediction.")
