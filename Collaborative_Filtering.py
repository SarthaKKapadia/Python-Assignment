import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "User": ["A", "A", "A", "B", "B", "B",
             "C", "C", "C", "D", "D"],
    "Movie": ["Avatar", "Titanic", "Inception",
              "Avatar", "Titanic", "Interstellar",
              "Avatar", "Inception", "Interstellar",
              "Titanic", "Inception"],
    "Rating": [5, 4, 5, 5, 4, 5, 4, 5, 4, 5, 4]
}

df = pd.DataFrame(data)

print("Movie Rating Dataset:")
print(df)

matrix = df.pivot_table(
    index="User",
    columns="Movie",
    values="Rating"
)

print("\nRating Matrix:")
print(matrix)

matrix = matrix.fillna(0)

similarity = cosine_similarity(matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=matrix.index,
    columns=matrix.index
)

print("\nUser Similarity:")
print(similarity_df.round(2))


def recommend_movies(user):
    recommendations = {}

    for movie in matrix.columns:
        if movie in df[df["User"] == user]["Movie"].values:
            continue

        score = 0
        count = 0

        for other in matrix.index:
            rating = matrix.loc[other, movie]
            sim = similarity_df.loc[user, other]

            if other != user and rating > 0 and sim > 0:
                score += rating * sim
                count += sim

        if count > 0:
            recommendations[movie] = score / count

    return sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )[:2]


user = "A"

print(f"\nRecommendations for User {user}:")

for movie, score in recommend_movies(user):
    print(f"{movie}: Predicted Rating = {score:.2f}")