import csv
import pandas
import threading

df = pandas.read_csv('./Test1Reviews.csv', usecols=['ProductId', 'UserId', 'Score'])
fieldnames = ['']
fieldrow = []

# print(df)
def Get_product(data):
    arrProduct = data['ProductId']   
    for x in arrProduct:
        if len(fieldnames) <= 1:
            fieldnames.append(x)
        else:
            flag = False
            for y in fieldnames:
                if x == y:
                    flag = True
            if flag == False:
                fieldnames.append(x)

def Get_user(data):
    arrUser = data['UserId']
    for x in arrUser:
        if len(fieldrow) <= 1:
            fieldrow.append(x)
        else:
            flag = False
            for y in fieldrow:
                if x == y:
                    flag = True
            if flag == False:
                fieldrow.append(x)

def Generator(data, userid, productid):
    with open('./Matrix1.csv', mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=productid)
        writer.writeheader()
        for x in userid:
            writer.writerow({'': x})
    #d = pandas.read_csv('./Matrix1.csv')
    with open('./Matrix1.csv') as csv_file:
        d = [row for row in csv.reader(csv_file)]
        print('\n\n')
    for x in userid:
        for y in data.itertuples():
            score = getattr(y, "Score")
            product = getattr(y, "ProductId")
            user = getattr(y, "UserId")
            
            if user == x:
                row = userid.index(user) + 1
                col = productid.index(product)           
                d[row][col] = score
                continue
    for x in userid:
        row = userid.index(x) + 1 
        for y in productid:
            col = productid.index(y)
            if d[row][col] == '':
                d[row][col] = 0
    with open('./Matrix1.csv', mode='w', newline='') as csv_file:
        w = csv.writer(csv_file)
        for row in d:
            w.writerow(row) 
    print('Completed')
        
thread1 = threading.Thread(Get_product(df))
print(len(fieldnames))
thread2 = threading.Thread(Get_user(df))
print(len(fieldrow))
thread3 =  threading.Thread(Generator(df, fieldrow, fieldnames))
