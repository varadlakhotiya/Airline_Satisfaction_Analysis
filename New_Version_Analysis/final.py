import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# Set a clean style for the plots
sns.set(style="whitegrid")

# -----------------------------
# 1. Load the dataset
# -----------------------------
# Replace 'airline_data.csv' with the path to your CSV file.
df = pd.read_csv('Cleaned_Data.csv')

# -----------------------------
# 2. Identify Service-Related Columns
# -----------------------------
# These are the service rating columns used in our analysis.
service_cols = [
    "Seat_Comfort", "InFlight_Entertainment", "Food_Quality", "Cleanliness",
    "Cabin_Staff_Service", "Legroom", "Baggage_Handling", "CheckIn_Service",
    "Boarding_Process", "WiFi_Service"
]

# -----------------------------
# 3. Compute Average Ratings and Identify Improvement Areas
# -----------------------------
# Calculate the average rating for each service along with the count of ratings used
avg_service_ratings = df[service_cols].mean()
rating_counts = df[service_cols].count()  # Count non-null values used for each average

# Compute an overall average rating across all services
overall_avg = avg_service_ratings.mean()

# Identify services with average ratings below the overall average (indicating possible improvement areas)
services_to_improve = avg_service_ratings[avg_service_ratings < overall_avg]

# Display the average ratings and counts
print("=== Airlines Customer Satisfaction Dashboard ===\n")
print("Key Findings:")
print("1. Overall average service rating: {:.2f}".format(overall_avg))

# Display the average ratings with the count of ratings used
for service, rating in avg_service_ratings.items():
    print(f"   - {service}: Average Rating: {rating:.2f}, Count of Ratings: {rating_counts[service]}")

if not services_to_improve.empty:
    print("2. Services needing improvement (average rating below overall average):")
    for service, rating in services_to_improve.items():
        print("   - {}: {:.2f}".format(service, rating))
else:
    print("2. All services are rated above the overall average.")

# -----------------------------
# 4. Display Key Findings
# -----------------------------
print("=== Airlines Customer Satisfaction Dashboard ===\n")
print("Key Findings:")
print("1. Overall average service rating: {:.2f}".format(overall_avg))
if not services_to_improve.empty:
    print("2. Services needing improvement (average rating below overall average):")
    for service, rating in services_to_improve.items():
        print("   - {}: {:.2f}".format(service, rating))
else:
    print("2. All services are rated above the overall average.")

# -----------------------------
# 5. Saving Key Findings to CSV
# -----------------------------
key_findings = {
    "Service": avg_service_ratings.index,
    "Average Rating": avg_service_ratings.values
}

key_findings_df = pd.DataFrame(key_findings)
key_findings_df.to_csv("key_findings.csv", index=False)

# Save services needing improvement to another CSV file
improvement_findings = {
    "Service": services_to_improve.index,
    "Average Rating": services_to_improve.values
}

improvement_df = pd.DataFrame(improvement_findings)
improvement_df.to_csv("services_to_improve.csv", index=False)

# -----------------------------
# 6. Visualization: Average Service Ratings
# -----------------------------
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(
    x=avg_service_ratings.index, 
    y=avg_service_ratings.values, 
    palette="viridis",
    legend=False
)
plt.axhline(
    overall_avg, 
    color='red', 
    linestyle='--', 
    label=f"Overall Average Rating ({overall_avg:.2f})"
)
plt.title("Average Service Ratings")
plt.ylabel("Average Rating")
plt.xlabel("Service")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('average_service_ratings.png')  # Save image

# -----------------------------
# 7. Visualization: Correlation Heatmap
# -----------------------------
# Include overall satisfaction to see which service metrics have the strongest impact.
corr_cols = service_cols + ["Overall_Satisfaction"]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix: Service Ratings vs. Overall Satisfaction")
plt.tight_layout()
plt.savefig('correlation_matrix.png')  # Save image

# -----------------------------
# 8. Additional Visualizations
# -----------------------------
# Mapping the class codes to actual names
class_mapping = {0: 'Business', 1: 'Economy', 2: 'First Class'}
df['Class'] = df['Class'].map(class_mapping)

# Boxplot: Overall Satisfaction by Flight Class (if available)
if 'Class' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Class", y="Overall_Satisfaction", data=df, palette="Set2")
    plt.title("Overall Satisfaction by Flight Class")
    plt.xlabel("Flight Class")
    plt.ylabel("Overall Satisfaction")
    plt.tight_layout()
    plt.savefig('overall_satisfaction_by_class.png')  # Save image

# Scatter Plot: Departure Delay vs. Overall Satisfaction (if available)
if 'Departure_Delay' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Departure_Delay", y="Overall_Satisfaction", data=df, alpha=0.6)
    plt.title("Departure Delay vs. Overall Satisfaction")
    plt.xlabel("Departure Delay (minutes)")
    plt.ylabel("Overall Satisfaction")
    plt.tight_layout()
    plt.savefig('departure_delay_vs_satisfaction.png')  # Save image

# Scatter Plot: Arrival Delay vs. Overall Satisfaction (if available)
if 'Arrival_Delay' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Arrival_Delay", y="Overall_Satisfaction", data=df, alpha=0.6)
    plt.title("Arrival Delay vs. Overall Satisfaction")
    plt.xlabel("Arrival Delay (minutes)")
    plt.ylabel("Overall Satisfaction")
    plt.tight_layout()
    plt.savefig('arrival_delay_vs_satisfaction.png')  # Save image

# -----------------------------
# 9. Generate HTML Report
# -----------------------------
html_template = """
<html>
<head>
    <title>Airlines Customer Satisfaction Dashboard</title>
</head>
<body>
    <h1>Airlines Customer Satisfaction Dashboard</h1>
    
    <h2>Key Findings</h2>
    <p>1. Overall average service rating: {{ overall_avg }}</p>
    <p>2. Services needing improvement:</p>
    <ul>
        {% for service, rating in services_to_improve.items() %}
            <li>{{ service }}: {{ rating }}</li>
        {% endfor %}
    </ul>
    
    <h2>Visualizations</h2>
    <p>Average Service Ratings:</p>
    <img src="average_service_ratings.png" alt="Average Service Ratings">
    
    <p>Correlation Matrix:</p>
    <img src="correlation_matrix.png" alt="Correlation Matrix">
    
    <h2>Additional Visualizations</h2>
    <p>Overall Satisfaction by Flight Class:</p>
    <img src="overall_satisfaction_by_class.png" alt="Overall Satisfaction by Flight Class">
    
    <p>Departure Delay vs. Overall Satisfaction:</p>
    <img src="departure_delay_vs_satisfaction.png" alt="Departure Delay vs. Overall Satisfaction">
    
    <p>Arrival Delay vs. Overall Satisfaction:</p>
    <img src="arrival_delay_vs_satisfaction.png" alt="Arrival Delay vs. Overall Satisfaction">
    
    <h2>Actionable Recommendations</h2>
    <ul>
        {% for service, rating in services_to_improve.items() %}
            <li>Focus on improving {{ service }} (Avg. Rating: {{ rating }})</li>
        {% endfor %}
        <li>Review operational factors such as delays, as they may negatively impact customer satisfaction.</li>
        <li>Consider tailored strategies for different flight classes based on observed satisfaction differences.</li>
        <li>Prioritize enhancements in service areas that show a strong correlation with overall satisfaction.</li>
    </ul>
    
</body>
</html>
"""

# Prepare data for the template
template = Template(html_template)
html_content = template.render(
    overall_avg=round(overall_avg, 2),
    services_to_improve=services_to_improve.to_dict()
)

# Save the HTML report
with open('customer_satisfaction_report.html', 'w') as file:
    file.write(html_content)

print("Report saved as 'customer_satisfaction_report.html'")
print("Visualizations and data saved as images and CSV files.")
