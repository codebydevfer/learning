# Simple Tip Splitter

total_bill = float(input("Enter the total bill amount: "))
tip_percent = float(input("Enter the tip percentage (e.g. 15 for 15%): "))
num_people = int(input("Enter the number of people splitting the bill: "))

tip_amount = total_bill * (tip_percent / 100)
total_with_tip = total_bill + tip_amount
amount_per_person = total_with_tip / num_people

print(f"\nTotal tip: ${tip_amount:.2f}")
print(f"Total bill with tip: ${total_with_tip:.2f}")
print(f"Each person should pay: ${amount_per_person:.2f}")