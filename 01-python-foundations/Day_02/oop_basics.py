#OOP Basics

#Classes

#A Class is like an object constructor, or a "blueprint" for creating objects.

#Creating a class

class MyClass:
  x = 5
  
#Creating objects

p1 = MyClass()
print(p1.x)


#Methods

#Use the __init__() method to assign values to object properties, or other operations that are necessary to do when the object is being created

class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

print(p1.name)
print(p1.age)

#The __str__() method controls what should be returned when the class object is represented as a string.
#If the __str__() method is not set, the string representation of the object is returned

#Without __string__ method:
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

print(p1)

#With __string__ method:
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def __str__(self):
    return f"{self.name}({self.age})"

p1 = Person("John", 36)

print(p1)

#Creating methods

#You can create your own methods inside objects. Methods in objects are functions that belong to the object.
#Let us create a method in the Person class

class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def myfunc(self):
    print("Hello my name is " + self.name)

p1 = Person("John", 36)
p1.myfunc()

#Self parameter

#The self parameter is a reference to the current instance of the class, and is used to access variables that belong to the class.
#It does not have to be named self, you can call it whatever you like, but it has to be the first parameter of any function in the class

class Person:
  def __init__(mysillyobject, name, age):
    mysillyobject.name = name
    mysillyobject.age = age

  def myfunc(abc):
    print("Hello my name is " + abc.name)

p1 = Person("John", 36)
p1.myfunc()

#Delete object properties

del p1.age

#Delete object

del p1

#Pass Statement

#class definitions cannot be empty, but if you for some reason have a class definition with no content, put in the pass statement to avoid getting an error.

class Person:
  pass

#__new__

# This method is responsible for creating a new instance of a class. It allocates memory and returns the new object. It is called before __init__.

class ClassName:
    def __new__(cls, parameters):
        instance = super(ClassName, cls).__new__(cls)
        return instance
    
#__init__

# This method initializes the newly created instance and is commonly used as a constructor in Python. It is called immediately after the object is created by __new__ method and is responsible for initializing attributes of the instance.

class ClassName:
    def __init__(self, parameters):
        self.attribute = value

#Constructors

#Default constructor

#A default constructor does not take any parameters other than self. It initializes the object with default attribute values.

class Car:
    def __init__(self):

        #Initialize the Car with default attributes
        self.make = "Toyota"
        self.model = "Corolla"
        self.year = 2020

# Creating an instance using the default constructor
car = Car()
print(car.make)
print(car.model)
print(car.year)

#Parameterized Constructor

#A parameterized constructor accepts arguments to initialize the object's attributes with specific values.

class Car:
    def __init__(self, make, model, year):
      
        #Initialize the Car with specific attributes.
        self.make = make
        self.model = model
        self.year = year

# Creating an instance using the parameterized constructor
car = Car("Honda", "Civic", 2022)
print(car.make)
print(car.model)
print(car.year)







#W3Schools Content
#Geeksforgeeks Content