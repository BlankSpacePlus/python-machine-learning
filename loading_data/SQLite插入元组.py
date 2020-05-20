import sqlite3

try:
    connection = sqlite3.connect('sqlite_load_test.db')
    cursor = connection.cursor()
    print("Opened database successfully")
    cursor.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (1, 'Sam', 32, 'BeiJing', 20000.00)")
    cursor.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (2, 'Bob', 25, 'ShangHai', 15000.00)")
    cursor.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (3, 'Jack', 23, 'Norway', 20000.00)")
    cursor.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (4, 'Tom', 25, 'GuangZhou', 65000.00)")
    connection.commit()
    print("Records created successfully")
finally:
    connection.close()
