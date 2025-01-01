import sqlite3

from fastcore.xtras import globtastic

# Paths to your databases

# Update with your database file paths
source_databases = globtastic('dbs', file_glob='*.db', skip_file_re='test_1|master')

# Update with your master database file path
master_database = "dbs/master.db"

def merge_databases(source_dbs, master_db):
    # Connect to the master database (create if it doesn't exist)
    master_conn = sqlite3.connect(master_db)
    master_cursor = master_conn.cursor()

    # Create the annotations table in the master database
    master_cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            video_name TEXT,
            frame_name TEXT,
            label TEXT
        )
    """)
    master_conn.commit()

    for db in source_dbs:
        print(f"Merging database: {db}")
        source_conn = sqlite3.connect(db)
        source_cursor = source_conn.cursor()

        # Read all rows from the source database
        source_cursor.execute("SELECT video_name, frame_name, label FROM annotations")
        rows = source_cursor.fetchall()

        # Insert rows into the master database
        try:

            master_cursor.executemany("""
                INSERT INTO annotations (video_name, frame_name, label)
                VALUES (?, ?, ?)
            """, rows)
            master_conn.commit()

        except sqlite3.IntegrityError:
            print(f"Skipping duplicate rows in {db}")

        source_conn.close()

    # Close the master connection
    master_conn.close()
    print(f"Merge completed. All data is in {master_db}")

merge_databases(source_databases, master_database)
