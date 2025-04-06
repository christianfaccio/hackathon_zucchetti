import pandas as pd

def predict_next_year(data):
    """
    Predict the next year's value for the dismissed model.
    Always predicts 0 for every month.
    """
    return 0

if __name__ == "__main__":
    input_history = pd.read_csv('input/01_input_history.csv')
    prediction = predict_next_year(input_history)
    with open('output/01_output_predictions_2179.csv', 'a') as f:
        f.write("Default,Default,Default,{}\n".format(prediction))

