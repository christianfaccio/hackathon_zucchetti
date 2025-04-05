import csv

def mean(csv_file):
    data = []
    somma = 0

    # Primo passaggio: raccolgo i dati e calcolo la somma totale
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            country = row['Country']
            capacity = int(row['Monthly Capacity'])
            data.append((country, capacity))
            somma += capacity

    # Secondo passaggio: calcolo la percentuale per ciascuna riga
    result = []
    for country, capacity in data:
        percentuale = capacity / somma
        result.append((country, capacity, percentuale))

    return result


risultati = mean('02_input_capacity.csv')
for country, capacity, percentuale in risultati:
    print(f"{country}: Capacity = {capacity}, Percentuale = {percentuale:.9%}")

