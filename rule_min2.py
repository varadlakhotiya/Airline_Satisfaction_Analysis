import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# Load the cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# Step 1: Prepare Transaction Data for Association Rule Mining
binarized_columns = [col for col in df.columns if col.startswith('Low_')]
transaction_data = df[binarized_columns]

# Step 2: Apply the Apriori Algorithm
frequent_itemsets = apriori(transaction_data, min_support=0.1, use_colnames=True)

# Step 3: Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Step 4: Filter and Analyze Rules
# Lower the lift threshold to 1.0
filtered_rules = rules[rules['lift'] > 1.0]

# Display frequent itemsets and rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)

print("\nFiltered Association Rules (Lift > 1.0):")
print(filtered_rules)

# Step 5: Visualize Association Rules
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x='support', y='confidence', size='lift', data=rules, hue='lift', palette='viridis')
plt.title('Association Rules: Support vs. Confidence')
plt.show()

# Step 6: Save Results
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)
filtered_rules.to_csv('filtered_association_rules.csv', index=False)

print("Association rule mining completed. Results saved to CSV files.")

import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph from the association rules
G = nx.DiGraph()

# Add edges to the graph
for _, rule in filtered_rules.iterrows():
    G.add_edge(str(rule['antecedents']), str(rule['consequents']), weight=rule['lift'])

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray', width=1.5, alpha=0.8)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title('Association Rules Network Graph', fontsize=14)
plt.show()