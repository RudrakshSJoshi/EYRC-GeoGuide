'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2D of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ eYRC#GG#3047 ]
# Author List:		[  SANSKRITI, AADITYA PORWAL,ARNAB MITRA,  RUDRAKSH SACHIN JOSHI ]
# Filename:			Task_2D.py
# Functions:	    [ read_csv, write_csv, tracker, main ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
import csv
import time
# Additional Imports

# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
path1 = [11, 14, 13, 18, 19, 20, 23, 21, 22, 33, 30, 35, 32, 31, 34, 40, 36, 38, 37, 39, 41, 50, 4, 6, 52, 7, 8, 1, 2, 11]
path2 = [11, 14, 13, 10, 9, 51, 53, 0, 39, 37, 38, 28, 25, 54, 5, 3, 19, 20, 17, 12, 15, 16, 27, 26, 24, 29, 40, 34, 31, 32, 35, 30, 33, 22, 21, 23, 20, 19, 18, 13, 14, 11]
###################################################################################################

# Declaring Variables
path1 = [str(x) for x in path1]
path2 = [str(x) for x in path2]

lat_long = "C:/Users/Rudraksh/OneDrive/Desktop/Task_2C/Task_2C/lat_long.csv"
live_data = "C:/Users/Rudraksh/OneDrive/Desktop/Task_2C/Task_2C/live_data.csv"

def read_csv(csv_name):
    lat_lon = {}

    with open(csv_name, 'r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)

        # Read the header row
            # Iterate through each row in the CSV
        for row in csv_reader:
            id, lat, lon = (row[0]), (row[1]), (row[2])
            lat_lon[id] = [lat, lon]
    return lat_lon



def write_csv(location, csv_name):
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["lat", "lon"])

        if location:
            writer.writerow(location)
        else:
            print("Coordinate is None. Skipping write to CSV.")

def tracker(ar_id, lat_lon):
    coordinate = lat_lon.get(str(ar_id)) 
    
    coordinate = [str(x) for x in coordinate]

    if coordinate is not None:
        # Write to live_data.csv
        write_csv(coordinate, "live_data.csv")
    else:
        print("Coordinate is None. Skipping write to CSV.")
        
    return coordinate

# ADDITIONAL FUNCTIONS

'''
If there are any additonal functions that you're using, you shall add them here. 

'''

###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################


def main():
    ###### reading csv ##############
    lat_lon = read_csv('lat_long.csv')
    print("###############################################")
    print(f"Received lat, lons : {len(lat_lon)}")
    if len(lat_lon) != 48:
        print(f"Incomplete coordinates received.")
        print("###############################################")
        exit()
    ###### Test case 1 ##############
    print("########## Executing first test case ##########")
    start = time.time()
    passed = 0
    traversedPath1 = []
    for i in path1:
        t_point = tracker(i, lat_lon)
        traversedPath1.append(t_point)
        time.sleep(0.5)
    end = time.time()
    if None in traversedPath1:
        print(f"Incorrect path travelled.")
        exit()
    print(f"{len(traversedPath1)} points traversed out of {len(path1)} points")
    print(f"Time taken: {int(end-start)} sec")
    if len(traversedPath1) != len(path1):
        print("Test case 1 failed. Travelled path is incomplete")
    else:
        print("Test case 1 passed !!!")
        passed = passed+1
    print("########## Executing second test case ##########")
    ###### Test case 2 ##############
    start = time.time()
    traversedPath2 = []
    for i in path2:
        t_point = tracker(i, lat_lon)
        traversedPath2.append(t_point)
        time.sleep(0.5)
    end = time.time()
    if None in traversedPath2:
        print(f"Incorrect path travelled.")
        exit()
    print(f"{len(traversedPath2)} points traversed out of {len(path2)} points")
    print(f"Time taken: {int(end-start)} sec")
    if len(traversedPath2) != len(path2):
        print("Test case 2 failed. Travelled path is incomplete")
    else:
        print("Test case 2 passed !!!")
        passed = passed+1
    print("###############################################")
    if passed==0:
        print("0 Test cases passed please check your code.")
    elif passed==1:
        print("Partialy correct, look for any logical erro ;(")
    else:
        print("Congratulations!")
        print("You've succesfully passed all the test cases \U0001f600")
    print("###############################################")
if __name__ == "__main__":
    main()
