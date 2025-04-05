import csv

def leggi_csv_4d(file_path):
    data = {}
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            country = row['Country']
            product = row['Product']
            month_year = row['Month']
            
            if len(month_year) < 4:
                continue
            month = month_year[:3]
            year = month_year[3:]
            
            try:
                quantity = float(row['Quantity'])
            except ValueError:
                continue

            if country not in data:
                data[country] = {}
            if product not in data[country]:
                data[country][product] = {}
            if year not in data[country][product]:
                data[country][product][year] = {}
            if month not in data[country][product][year]:
                data[country][product][year][month] = 0
            
            data[country][product][year][month] += quantity

    return data

def categorize_products(data):
    categories = {}
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for country, products in data.items():
        categories[country] = {}
        for product, years in products.items():
            try:
                latest_year = max(years.keys(), key=lambda y: int(y))
            except ValueError:
                continue
            
            monthly_data = years[latest_year]
            sales_counts = [monthly_data.get(month, 0) for month in months_order]
            positive_months = sum(1 for s in sales_counts if s > 0)
            
            if positive_months == 0:
                category = "dismissed"
            elif positive_months < 12:
                category = "intervals"
            else:
                category = "regular"
            
            categories[country][product] = category

    return categories

def aggiungi_categoria_al_csv(input_file, output_file, categories):
    with open(input_file, newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['Category']  # aggiungiamo la nuova colonna
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in reader:
            country = row['Country']
            product = row['Product']
            category = categories.get(country, {}).get(product, "unknown")
            row['Category'] = category
            writer.writerow(row)

# === USO ===
file_input = "C:\\Users\\hp\\Desktop\\data\\01_input_history.csv"
file_output = "C:\\Users\\hp\\Desktop\\data\\01_input_history_with_category.csv"

data = leggi_csv_4d(file_input)
categorie = categorize_products(data)
aggiungi_categoria_al_csv(file_input, file_output, categorie)