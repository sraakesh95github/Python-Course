import sys
import math
# Program to calculate the time taken for a ball to drop from a height for a given initial velocity

#Set a constant value for acceleration due to gravity
g = 9.81;

#Get the height from which the ball is dropped
s = int(input("Enter height : "));

#Check for the height range
if s<10 or s>1000 :
    print("Bad height specified. Please try again.");
    sys.exit()

#Get the intial velocity with which the ball is thrown / dropped
u = float(input("Enter initial upward velocity : "));
#Check for the initial velocity range
if u<-20 or u>20 :
    print("Initial velocity too large ! Slow down !");
    sys.exit()
    
    
#Calculate the time taken for the ball to drop to the groud Negative initial velocity is given as the ball is moving downward
t = ((math.sqrt((u*u)+(2*g*s)) + u) / g);

#Print time taken
print("time to hit ground " + "{:.2f}".format(t) + " seconds");

