#Variables

a = 1
b = "Hello World!"
c = ['Example', 'Example']
d = {'Example', 'Example'}

#Lists

#1
primes = [2, 3, 5, 7]

#2
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

planets[0]

#3
hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Comma after the last element is optional)
]
# (I could also have written this on one line, but it can get hard to read)
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]

#4
my_favourite_things = [32, 'raindrops on roses', help]

#List Functions

len(planets)

sorted(planets)

primes = [2, 3, 5, 7]
sum(primes)

max(primes)


#Dictionaries

#1
numbers = {'one':1, 'two':2, 'three':3}

#Adding an element to the dict
numbers['eleven'] = 11
numbers

#Changing values
numbers['one'] = 'Pluto'
numbers

#2
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}
planet_to_initial

#Loop in a dict
for k in numbers:
    print("{} = {}".format(k, numbers[k]))
    
#3
# Get all the initials, sort them alphabetically, and put them in a space-separated string.
' '.join(sorted(planet_to_initial.values()))

#4
for planet, initial in planet_to_initial.items():
    print("{} begins with \"{}\"".format(planet.rjust(10), initial))
    
#To know more about dictionaries and other python elements, we can use 'help(dict)'

#Loops

#Ex - 1
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ') # print all on same line
    
#Ex - 2
multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
product

#Ex - 3
s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='')        
        
#Range

for i in range(5):
    print("Doing important work. i =", i)
    
#While Loops

i = 0
while i < 10:
    print(i, end=' ')
    i += 1 # increase the value of i by 1
    
#Loop & List

squares = [n**2 for n in range(10)]
squares

#Same example without list

squares = []
for n in range(10):
    squares.append(n**2)
squares

#Adding if statements

short_planets = [planet for planet in planets if len(planet) < 6]
short_planets

#Functions

#Ex - 1
def least_difference(a, b, c):
    """Return the smallest difference between any two numbers
    among a, b and c.
    
    >>> least_difference(1, 5, -5)
    4
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)

#Ex - 2
def greet(who="Colin"):
    print("Hello,", who)
    
greet()
greet(who="Kaggle")
# (In this case, we don't need to specify the name of the argument, because it's unambiguous.)
greet("world")

#Output
# Hello, Colin
# Hello, Kaggle
# Hello, world

#Functions that don't return

def least_difference(a, b, c):
    """Return the smallest difference between any two numbers
    among a, b and c.
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    min(diff1, diff2, diff3)
    
print(
    least_difference(1, 10, 100),
    least_difference(1, 10, 10),
    least_difference(5, 6, 7),
)

#Functions in Functions

def mult_by_five(x):
    return 5 * x

def call(fn, arg):
    """Call fn on arg"""
    return fn(arg)

def squared_call(fn, arg):
    """Call fn on the result of calling fn on arg"""
    return fn(fn(arg))

print(
    call(mult_by_five, 1),
    squared_call(mult_by_five, 1), 
    sep='\n', # '\n' is the newline character - it starts a new line
)







#Kaggle Content