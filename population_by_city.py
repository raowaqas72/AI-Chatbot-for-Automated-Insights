import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('uae_city_data_rao.csv',encoding='ISO-8859-1')
# Group data by city and calculate male and female populations
gender_population_by_city = data.groupby('City')[['Male Population', 'Female Population']].sum()

# Create a grouped bar chart
plt.figure(figsize=(10, 6))
gender_population_by_city.plot(kind='bar', stacked=True)
plt.title('Gender Population by City')
plt.xlabel('City')
plt.ylabel('Population')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.legend(title='Gender')
plt.show()
