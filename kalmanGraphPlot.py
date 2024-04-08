import pandas as pd
import matplotlib.pyplot as plt
"""

Auther: Koray Aman Arabzadeh
Thesis: Mid Sweden University.
Bachelor Thesis - Bachelor of Science in Engineering, Specialisation in Computer
Engineering
Main field of study: Computer Engineering
Credits: 15 hp (ECTS)
Semester, Year: Spring, 2024
Supervisor: Emin Zerman
Examiner: Stefan Forsström
Course code: DT099G
Programme: Degree of Bachelor of Science with a major in Computer Engineering


This script plots individual detections and predictions as distinct points, 
helping the reader distinguish between what has been measured (detections) and what is being predicted (future positions). 
The legend is placed outside the plot area so it doesn't obscure the data, 
and tight_layout() is called to ensure all elements of the plot are neatly arranged.

"""
# Load the data from CSV
df = pd.read_csv('tracking_and_predictions.csv')

# Filter data for a specific class, e.g., 'person'
class_name = 'person'
df_filtered = df[df['class_name'] == class_name]

# Plotting
plt.figure(figsize=(12, 8))

# Plot detected positions with blue circles
plt.scatter(df_filtered['det_x'], df_filtered['det_y'], c='blue', label=f'Detected Path ({class_name})', zorder=2)

# Plot predicted future positions with red crosses
plt.scatter(df_filtered['pred_x'], df_filtered['pred_y'], c='red', marker='x', label=f'Predicted Future Position ({class_name})', zorder=1)

plt.title(f'Object Tracking and Future Position Prediction for {class_name}')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend outside of the plot
plt.grid(True)
plt.tight_layout()  # Adjust the padding between and around subplots
plt.show()
