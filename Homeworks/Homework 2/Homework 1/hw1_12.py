import math

#Create a list of prime numbers
prime_number_list = [2]

#Set the maximum limit
MAX_LIMIT = 10000

#initialize iterator
i = 3

#Iteration to check all the numbers for being a prime number or not
while i < MAX_LIMIT: #i is the number to be evaluated
    temp = math.sqrt(i)
    for j in prime_number_list:
        if j <= temp : #Condition to check the prime factors less than the square root of the given number i
            if i%j == 0: break
        else : 
            prime_number_list.append(i)
            break
    i+=2 #Even numbers do not constitute prime numbers, skip the even numbers
print(prime_number_list)