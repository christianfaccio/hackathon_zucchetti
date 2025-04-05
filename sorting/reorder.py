import csv

def crea_dizionario_request(csv_file):
    request = {}

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            stato = row['Country']
            prodotto = row['Product']
            mese = row['Month']
            quantita = int(row['Quantity'])

            if stato not in request:
                request[stato] = {}

            if prodotto not in request[stato]:
                request[stato][prodotto] = {}

            if mese not in request[stato][prodotto]:
                request[stato][prodotto][mese] = [0, 0]

            request[stato][prodotto][mese][0] += quantita

    return request

csv_file = '02_input_target.csv'
request = crea_dizionario_request(csv_file)
print(request)