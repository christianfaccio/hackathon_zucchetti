import csv
from collections import defaultdict

def calcola_richieste_globali(file_path='02_input_target.csv'):
    # Dizionario: {prodotto: {mese: quantità_totale}}
    richieste_globali = defaultdict(lambda: defaultdict(int))

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mese = row['Month']
            prodotto = row['Product']
            quantità = int(row['Quantity'])
            richieste_globali[prodotto][mese] += quantità

    # Stampa il dizionario
    for prodotto, mesi in richieste_globali.items():
        print(f"{prodotto}:")
        for mese, totale in mesi.items():
            print(f"  {mese}: {totale}")

    return richieste_globali

# Chiamata della funzione
calcola_richieste_globali()
