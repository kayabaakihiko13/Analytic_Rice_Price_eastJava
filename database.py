import sqlite3
import pandas as pd
from typing import Optional


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)

    def execute_query(self, query: str, fetch: bool = False):
        if self.connection is None:
            raise ConnectionError(
                "Database not connected. Please call connect() first."
            )

        cursor = self.connection.cursor()
        cursor.execute(query)

        if fetch:
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame(result, columns=columns)
        else:
            self.connection.commit()
            return None

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
