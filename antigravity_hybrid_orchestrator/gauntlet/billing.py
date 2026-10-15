import time
import threading
from typing import Dict

class BillingSystem:
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self._lock = threading.Lock()

    def create_account(self, account_id: str, initial_balance: float = 0.0) -> None:
        with self._lock:
            self.balances[account_id] = initial_balance

    def get_balance(self, account_id: str) -> float:
        with self._lock:
            return self.balances.get(account_id, 0.0)

    def withdraw(self, account_id: str, amount: float) -> bool:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        with self._lock:
            current_balance = self.balances.get(account_id, 0.0)
            if current_balance >= amount:
                time.sleep(0.005)
                self.balances[account_id] = current_balance - amount
                return True
            return False
