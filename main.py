from config import csv_path, user_id_input
import data_loader
import model_utils
import recommendation_engine

def main(user_id):
    # Load the cleaned data
    ratings_df = data_loader.load_ratings_data(csv_path)
    model = model_utils.train_model(ratings_df)
    recommendations = recommendation_engine.return_result(user_id, ratings_df, model, 20)

    return recommendations


if __name__ == '__main__':
    result = main(user_id_input)
    print(result)
