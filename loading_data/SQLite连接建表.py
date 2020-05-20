import sqlite3

try:
    connection = sqlite3.connect('sqlite_load_test.db')
    print("Opened database successfully")
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE COMPANY
           (ID INT PRIMARY KEY     NOT NULL,
           NAME           TEXT    NOT NULL,
           AGE            INT     NOT NULL,
           ADDRESS        CHAR(50),
           SALARY         REAL);''')
    print("Table created successfully")
    connection.commit()
finally:
    connection.close()
