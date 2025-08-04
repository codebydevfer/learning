class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited ${amount}. New balance: ${self.balance}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self.balance}")
        else:
            print("Insufficient funds.")

    def get_balance(self):
        print(f"Current balance: ${self.balance}")
        return self.balance

    def __str__(self):
        return f"BankAccount(owner: {self.owner}, balance: ${self.balance})"


# Create a new bank account
account = BankAccount("Sara", 100)

# Check balance
account.get_balance()

# Deposit money
account.deposit(50)

# Withdraw money
account.withdraw(30)

# Try to withdraw too much
account.withdraw(200)

# Print account info
print(account)