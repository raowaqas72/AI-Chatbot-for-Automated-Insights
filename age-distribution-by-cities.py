import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('uae_city_data_rao.csv',encoding='ISO-8859-1')

# Group data by city and calculate age group distribution
age_groups = ['Age 0-18', 'Age 19-35', 'Age 36-50', 'Age 51+']
age_distribution_by_city = data.groupby('City')[age_groups].sum()

# Create a stacked bar chart
plt.figure(figsize=(10, 6))
age_distribution_by_city.plot(kind='bar', stacked=True)
plt.title('Age Distribution by City')
plt.xlabel('City')
plt.ylabel('Population')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.legend(title='Age Groups')
plt.show()
