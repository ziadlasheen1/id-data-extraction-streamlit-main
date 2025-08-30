import sqlite3

def insert_to_db(data):
    conn = sqlite3.connect("egypt_id_data.db")
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS ids (
        name TEXT, id TEXT, dob TEXT, gender TEXT,
        address TEXT, issue_date TEXT, governorate TEXT, religion TEXT
    )""")

    cursor.execute("INSERT INTO ids VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
        (data['name'], data['id'], data['dob'], data['gender'],
         data['address'], data['issue_date'], data['governorate'], data['religion'])
    )
    conn.commit()
    conn.close()
