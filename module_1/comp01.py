""" 
Author: XXX xxx 
Date: XXXX-XX-XX 
Version: 1.0 
""" 
 
# import libraries 
import numpy as np 
 
class Generator: 
    def __init__(self): 
        """ 
        Constructor 
        """ 
        self.lowBoundary = 0 
        self.highBoundary = 10 
 
    def getRandomData( self, valNum=100 ): 
        """Generate a numpy array with random numbers 
 
        Args: 
            valNum (int, optional): number of element in the array. Defaults to 100. 
 
        Returns: 
            np.array: random numbers 
        """ 
         
 
        resArr = np.random.randint( self.lowBoundary, self.highBoundary, valNum ) 
 
        return resArr

# This code section will be called only when this file is run directly with: 
#  python filename  
if __name__ == "__main__": 
    # create generator object 
    gen1 = Generator() 
    # call method to generating a numpy array with 5 random numbers 
    sampleArr = gen1.getRandomData(5) 
    # print array 
    print(sampleArr) 