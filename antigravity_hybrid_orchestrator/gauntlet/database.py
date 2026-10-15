import sqlite3
from typing import List, Tuple, Optional

class DatabaseManager:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                role TEXT,
                email TEXT
            )
        """)
        self.conn.commit()

    def add_user(self, username: str, role: str, email: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, role, email) VALUES (?, ?, ?)",
            (username, role, email),
        )
        self.conn.commit()

    def get_user_by_username(self, username: str) -> Optional[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, username, role, email FROM users WHERE username = ?",
            (username,),
        )
        return cursor.fetchone()

    def get_users_by_role(self, role: str) -> List[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, username, role, email FROM users WHERE role = ?",
            (role,),
        )
        return cursor.fetchall()

    def close(self) -> None:
        self.conn.close()
