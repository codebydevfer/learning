# Simple BMI Calculator
import math

user_weight = float(input("Enter your weight in Kilograms: "))
user_height_cm = float(input("Enter your height in Centimeters: "))
user_height_m = user_height_cm / 100  # Convert to meters

user_bmi = user_weight / math.pow(user_height_m, 2)

if user_bmi < 18.5:
    print(f"Your BMI is: {user_bmi:.2f} - Underweight")
elif 18.5 <= user_bmi <= 24.9:
    print(f"Your BMI is: {user_bmi:.2f} - Normal")
elif 25 <= user_bmi <= 29.9:
    print(f"Your BMI is: {user_bmi:.2f} - Overweight")
elif 30 <= user_bmi <= 34.9:
    print(f"Your BMI is: {user_bmi:.2f} - Obese")
else:
    print(f"Your BMI is: {user_bmi:.2f} - Extremely Obese")
