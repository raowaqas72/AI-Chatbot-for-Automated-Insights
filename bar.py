import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('uae_city_data_rao.csv',encoding='ISO-8859-1')

# Group data by city and calculate total population
population_by_city = data.groupby('City')['Total Population'].sum()

# Create a bar chart
plt.figure(figsize=(10, 6))
population_by_city.plot(kind='bar', color='blue')
plt.title('Population by City')
plt.xlabel('City')
plt.ylabel('Total Population')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
