
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import signal
from sklearn.metrics import root_mean_squared_error
import datetime

def Predict_fit(a1, b1, a2, b2, W1, epsilon):
    min_x = int(min(a1,a2))
    min_y = int(min(b1,b2))
        
    max_x = int(a1+a2+10)
    max_y= int(b1+b2+10)
    W2 = 1-W1
    
    delta = 10 ** -(4)
    big_grid = np.arange(delta,1,delta)

    Beta1 = stats.beta.pdf(big_grid, a1, b1, loc=0, scale=W1)
    Beta2 = stats.beta.pdf(big_grid,a2, b2, loc=0, scale=W2)
    
    stop_limit = Epsilon
    
    conv_pmf = signal.fftconvolve(Beta1,Beta2,'full')
    size_xaxis = int( (1/delta)-1)
    conv_pmf = conv_pmf[:size_xaxis]
    area3 = np.trapz(conv_pmf)
    conv_pmf = conv_pmf/(area3*delta)

    print("Printing convoluted distribution!!!")
    
    plt.plot(big_grid, Beta1)
    plt.plot(big_grid, Beta2)
    plt.plot(big_grid, conv_pmf)
    string1 = str(W1)+"*Beta(" +str(a1) +"," +str(b1)+")"
    string2 = str(round(W2, 2))+"*Beta(" +str(a2) +"," +str(b2)+")"
    string3 = "C(" +str(a1) +"," +str(b1)+ ","+ str(a2) +"," +str(b2) +"," +str (W1)+")"
    plt.legend([string1, string2, string3], loc="upper right")
    plt.show()
    
    
    print("Fitting Beta on convoluted distribution...")
    
    
    max = 1000
    best_x =0
    best_y = 0
    best2_x =0
    best2_y = 0
    best3_x =0
    best3_y = 0
    best4_x =0
    best4_y = 0

    for x in range(min_x, 10*(max_x), 1):
        for y in range(min_y, 10*(max_y), 1):
            Beta_test = stats.beta.pdf(big_grid,x/10, y/10, loc=0, scale=1)
            rms = root_mean_squared_error(conv_pmf, Beta_test)
            if(rms < max):
                max = rms
                best_x = x/10
                best_y = y/10
                
            if(rms < stop_limit):
                print("Fitted Beta(" +str(best_x) +","+str(best_y)+"), with RMSE score: "+str(max))
                plt.plot(big_grid, conv_pmf)
                plt.plot(big_grid, Beta_test)
                string2 = "Fitted Beta(" +str(best_x) +"," +str(best_y)+")"
                string3 = "C(" +str(a1) +"," +str(b1)+ ","+ str(a2) +"," +str(b2) +"," +str (W1)+")"
                plt.legend([string2, string3], loc="upper right")
                plt.show()
                return
   
    range_low_x = int(100*best_x-20)  
    range_high_x = int(100*best_x+20)  
    range_low_y = int(100*best_y-20)  
    range_high_y = int(100*best_y+20)  
    max = 1000    

    for x2 in range(range_low_x, range_high_x, 1):
        for y2 in range(range_low_y, range_high_y, 1):
            Beta_test = stats.beta.pdf(big_grid,x2/100, y2/100, loc=0, scale=1)
            rms = root_mean_squared_error(conv_pmf, Beta_test)
            if(rms < max):
                max = rms
                best2_x = x2/100
                best2_y = y2/100
            
            if(rms < stop_limit):
                print("Fitted Beta(" +str(best2_x) +","+str(best2_y)+"), with RMSE score: "+str(max))
                plt.plot(big_grid, conv_pmf)
                plt.plot(big_grid, Beta_test)
                string2 = "Fitted Beta(" +str(best2_x) +"," +str(best2_y)+")"
                string3 = "C(" +str(a1) +"," +str(b1)+ ","+ str(a2) +"," +str(b2) +"," +str (W1)+")"
                plt.legend([string2, string3], loc="upper right")
                plt.show()
                return
                
    range_low_x2 = int(1000*best2_x-10)  
    range_high_x2 = int(1000*best2_x+10)  
    range_low_y2 = int(1000*best2_y-10)  
    range_high_y2 = int(1000*best2_y+10)  
    max = 1000
    
    for x3 in range(range_low_x2, range_high_x2, 1):
        for y3 in range(range_low_y2, range_high_y2, 1):
            Beta_test = stats.beta.pdf(big_grid,x3/1000, y3/1000, loc=0, scale=1)
            rms = root_mean_squared_error(conv_pmf, Beta_test)
            if(rms < max):
                max = rms
                best3_x = x3/1000
                best3_y = y3/1000
                
            if(rms < stop_limit):
                print("Fitted Beta(" +str(best3_x) +","+str(best3_y)+"), with RMSE score: "+str(max))
                string2 = "Fitted Beta(" +str(best3_x) +"," +str(best3_y)+")"
                string3 = "C(" +str(a1) +"," +str(b1)+ ","+ str(a2) +"," +str(b2) +"," +str (W1)+")"
                plt.plot(big_grid, conv_pmf)
                plt.plot(big_grid, Beta_test)
                plt.legend([string2, string3], loc="upper right")
                plt.show()
                return
                
    range_low_x3 = int(10000*best3_x-10)  
    range_high_x3 = int(10000*best3_x+10)  
    range_low_y3 = int(10000*best3_y-10)  
    range_high_y3 = int(10000*best3_y+10)  
    max = 1000
    
    for x4 in range(range_low_x3, range_high_x3, 1):
        for y4 in range(range_low_y3, range_high_y3, 1):
            Beta_test = stats.beta.pdf(big_grid,x4/10000, y4/10000, loc=0, scale=1)
            rms = root_mean_squared_error(conv_pmf, Beta_test)
            if(rms < max):
                max = rms
                best4_x = x4/10000
                best4_y = y4/10000

    print("Fitted Beta(" +str(best4_x) +","+str(best4_y)+"), with RMSE score: "+str(max))
    Beta_test = stats.beta.pdf(big_grid,best4_x, best4_y, loc=0, scale=1)
    plt.plot(big_grid, conv_pmf)
    plt.plot(big_grid, Beta_test)
    string2 = "Fitted Beta(" +str(best4_x) +"," +str(best4_y)+")"
    string3 = "C(" +str(a1) +"," +str(b1)+ ","+ str(a2) +"," +str(b2) +"," +str (W1)+")"
    plt.legend([string2, string3], loc="upper right")
    plt.show()
    return



print("Welcome to the numerical Beta fitter for Weighted Beta Convolution!!!")
a_test1 = 0
while a_test1 <= 1:
    a_test1 = float(input("Enter alpha_1 (number greater than 1): "))

b_test1 = 0
while b_test1 <= 1:
    b_test1 = float(input("Enter beta_1 (number greater than 1): "))
    
a_test2 = 0
while a_test2 <= 1:
    a_test2 = float(input("Enter alpha_2 (number greater than 1): "))

b_test2 = 0
while b_test2 <= 1:
    b_test2 = float(input("Enter beta_2 (number greater than 1): "))
    
W = 0
while W >= 1 or W <= 0:
    W = float(input("Enter W (number less than 1, but greater than 0): "))
    
Epsilon = -1
while Epsilon < 0:
    Epsilon = float(input("Enter Epsilon, the minimum limit you accept or 0 for fittest Beta we can find (non-negative number): "))
    
Predict_fit(a_test1, b_test1, a_test2, b_test2, W, Epsilon)