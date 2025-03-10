import os
import glob
import logging
import argparse
import pandas as pd
import numpy as np
from ievals.settings import categories, subject2category


def get_parser():
    parser = argparse.ArgumentParser(description="Run TMMLU+ score aggregate")
    parser.add_argument("files", nargs="+", help="TSV files to process")
    return parser


def process_single_file(filename, inverted_categories):
    df = pd.read_csv(filename, delimiter="\t", header=None)
    df.columns = ["model_name", "subject", "accuracy"]
    
    model_name = df.model_name.iloc[0]
    results = {
        "humanities": [],
        "social sciences": [],
        "STEM": [],
        "Others": []
    }
    total = 0
    
    for _, row in df.iterrows():
        total += 1
        subject = row["subject"]
        category = inverted_categories[subject2category[subject]]
        if category == 'other (business, health, misc.)':  # Fix category name
            category = "Others"
        results[category].append(row["accuracy"])
    
    data = {"model_name": model_name}
    avg_scores = 0
    for category, scores in results.items():
        if scores:  # Only calculate if we have scores
            data[category] = np.mean(scores)
            avg_scores += np.mean(scores)
    data["Average"] = avg_scores / 4
    
    if total != 66:
        logging.warning(f"Warning: {model_name} has {total} subjects instead of expected 66")
    
    return data


def format_leaderboard(model_scores):
    # Fixed categories order
    categories = ["humanities", "social sciences", "STEM", "Others", "Average"]
    
    # Calculate max lengths for formatting
    max_model_length = max(len(score["model_name"]) for score in model_scores)
    header = f"{'Model':^{max_model_length}} | " + " | ".join(f"{cat:^8}" for cat in categories)
    separator = "-" * len(header)
    
    leaderboard = [header, separator]
    for score in model_scores:
        model_name = score["model_name"]
        try:
            scores = [f"{score[cat]:^8.2f}" for cat in categories]
            row = f"{model_name:<{max_model_length}} | " + " | ".join(scores)
            leaderboard.append(row)
        except KeyError as e:
            print(e)
    return "\n".join(leaderboard)


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Create inverted categories mapping
    inverted_categories = {}
    for big_cat, cats in categories.items():
        for cat in cats:
            inverted_categories[cat] = big_cat

    # Get all matching files
    all_files = args.files
    if not all_files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Process each file
    model_scores = []
    for filename in all_files:
        data = process_single_file(filename, inverted_categories)
        model_scores.append(data)

    # Sort by average score
    model_scores = sorted(model_scores, key=lambda x: x["Average"], reverse=True)
    # Print formatted leaderboard
    print("\nTMMLU+ Leaderboard:")
    print(format_leaderboard(model_scores))
    
    return model_scores


if __name__ == "__main__":
    main()