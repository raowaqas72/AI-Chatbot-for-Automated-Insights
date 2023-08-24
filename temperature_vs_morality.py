import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('uae_city_data_rao.csv',encoding='ISO-8859-1')

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Average Temperature (°C)'], data['Mortality Rate (%)'], color='blue')
plt.title('Temperature vs. Mortality Rate')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Mortality Rate (%)')
plt.tight_layout()

# Show the plot
plt.show()
