import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('uae_city_data_rao.csv',encoding='ISO-8859-1')


# Group data by city and calculate average healthcare expenditure
average_expenditure_by_city = data.groupby('City')['Healthcare Expenditure (% of Income)'].mean()

# Create a line chart
plt.figure(figsize=(10, 6))
average_expenditure_by_city.plot(kind='line', marker='o', color='blue')
plt.title('Average Healthcare Expenditure by City')
plt.xlabel('City')
plt.ylabel('Healthcare Expenditure (% of Income)')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
