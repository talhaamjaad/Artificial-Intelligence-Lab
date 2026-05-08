movies = [
    ("Eternal Sunshine of the Spotless Mind", 20000000),
    ("Memento", 9000000),
    ("Requiem for a Dream", 4500000),
    ("Pirates of the Caribbean: On Stranger Tides", 379000000),
    ("Avengers: Age of Ultron", 365000000),
    ("Avengers: Endgame", 356000000),
    ("Incredibles 2", 200000000)
]

count = int(input("How many movies do you want to add? "))

for _ in range(count):
    name = input("Enter movie name: ")
    budget = int(input("Enter movie budget: "))
    movies.append((name, budget))

total_budget = 0

for movie in movies:
    total_budget += movie[1]

average_budget = total_budget / len(movies)

print("Average Budget:", average_budget)

higher_count = 0

for movie in movies:
    if movie[1] > average_budget:
        difference = movie[1] - average_budget
        print(movie[0], "is over budget by", difference)
        higher_count += 1

print("Number of movies above average budget:", higher_count)