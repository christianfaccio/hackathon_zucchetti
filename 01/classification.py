import csv

def leggi_csv_4d(file_path):
    """
    Legge il file CSV input_history: Country, Product, Month, Quantity
    
    return: dizionario tridimensionale: data[country][product][year][month]
    """

    data = {}
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            country = row['Country']
            product = row['Product']
            month_year = row['Month']
            
            # Estrai il mese (i primi 3 caratteri) e l'anno (il resto)
            if len(month_year) < 4:
                continue  # ignora eventuali righe con formato errato
            month = month_year[:3]
            year = month_year[3:]
            
            try:
                quantity = float(row['Quantity'])
            except ValueError:
                continue  # se la quantità non è un numero, salta la riga

            # Inizializza i livelli del dizionario se non esistono già
            if country not in data:
                data[country] = {}
            if product not in data[country]:
                data[country][product] = {}
            if year not in data[country][product]:
                data[country][product][year] = {}
            if month not in data[country][product][year]:
                data[country][product][year][month] = 0
            
            # Somma la quantità (utile se ci sono più righe per le stesse categorie)
            data[country][product][year][month] += quantity

    return data

def categorize_products(data):
    """
    Riceve in input un dizionario strutturato come:
      data[country][product][year][month] = quantità venduta
    e restituisce un dizionario che per ogni paese e prodotto assegna una categoria:
      - "Prodotti dismessi"
      - "Prodotti venduti solo in alcuni mesi dell'anno"
      - "Prodotti ancora in vendita"
      
    Il criterio utilizzato si basa sui dati dell'anno più recente:
      - Se in tutti i 12 mesi la quantità venduta è 0 -> prodotto dismesso
      - Se in alcuni mesi la quantità è > 0, ma non in tutti -> venduto solo in alcuni mesi
      - Se in tutti i 12 mesi la quantità è > 0 -> ancora in vendita
    """
    categories = {}
    # Lista ordinata dei mesi considerati (le chiavi attese nel CSV)
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for country, products in data.items():
        if country not in categories:
            categories[country] = {}
        for product, years in products.items():
            # Determiniamo l'anno più recente: assumiamo che le chiavi degli anni possano essere convertite in interi
            try:
                latest_year = max(years.keys(), key=lambda y: int(y))
            except ValueError:
                continue  # se l'anno non è nel formato corretto, saltiamo il prodotto
            
            monthly_data = years[latest_year]
            # Per ogni mese controlliamo le vendite; se il mese non è presente, assumiamo 0
            sales_counts = [monthly_data.get(month, 0) for month in months_order]
            
            # Calcoliamo il numero di mesi in cui sono state registrate vendite (maggiore di 0)
            positive_months = sum(1 for s in sales_counts if s > 0)
            
            if positive_months == 0:
                category = "dismissed"
            elif positive_months < 12:
                category = "intervals"
            else:
                category = "regular"
            
            categories[country][product] = category


    return categories

import csv

def write_categories_to_csv(categories, output_file):
    """
    Scrive il dizionario 'categories' in un file CSV.
    
    Il dizionario 'categories' ha la seguente struttura:
      {
        'Country1': {
            'Product1': 'Categoria1',
            'Product2': 'Categoria2',
            ...
        },
        'Country2': {
            ...
        },
        ...
      }
    
    L'output CSV avrà le colonne: Country, Product, Category.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Scrive l'intestazione del file CSV
        writer.writerow(['Country', 'Product', 'Category'])
        
        # Itera attraverso il dizionario e scrive ogni riga
        for country, products in categories.items():
            for product, category in products.items():
                writer.writerow([country, product, category])

def classify(group_data):
    """
    Dummy classification function.
    Uses the mean of 'Quantity' from group_data to select a model.
    Returns one of 'dismissed', 'intervals', or 'regular'.
    """
    if group_data.empty:
        return 'regular'
    mean_quantity = group_data['Quantity'].mean()
    if mean_quantity > 100:
        return 'dismissed'
    elif mean_quantity > 50:
        return 'intervals'
    else:
        return 'regular'

if __name__ == "__main__":
    # Existing code that writes categories.csv should run only when executed directly.
    file_path = "../chiavetta/data/01_input_history.csv"  # adjust if necessary
    categories = categorize_products(leggi_csv_4d(file_path))
    write_categories_to_csv(categories, "categories.csv")
