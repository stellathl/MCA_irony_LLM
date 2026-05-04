# test the code for the HPC
import csv
import os

print("Script is running!")

# Show where you are (important on HPC)
print("Current working directory:", os.getcwd())

# Write a CSV file
with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["hello world!"])

print("File written: output.csv")