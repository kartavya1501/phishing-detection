from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model

def main():
    # Paths
    data_path = "data/dataset.csv"

    # Load and preprocess data
    df = load_data(data_path)
    X, y = preprocess_data(df)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate model (prints metrics + saves plot + report)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()