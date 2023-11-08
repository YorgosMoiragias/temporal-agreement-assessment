#! C:\Users\user\Desktop\temporal-agreement-assessment-of-listening-panel\Scripts\python.exe
import os
import panel_evaluation

# Define the csv file containing the temporal Envelopment ratings
file_path="C:/Users/user/Desktop/temporal-agreement-assessment-of-listening-panel/Temporal Envelopment Data/Demo_Temporal_Envelopment_Assessment.csv"

# Find the assessors that are in agreement with the majority of the listening panel and those who are not
AssessorToHold, AssessorsToBeRemoved=panel_evaluation.agreement_evaluation(file_path,True)

# Remove the assessors not in agreement and create an updated csv file
out=panel_evaluation.remove_assessors(file_path, AssessorsToBeRemoved, True)
