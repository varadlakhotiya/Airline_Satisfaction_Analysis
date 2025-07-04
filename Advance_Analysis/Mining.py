import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class AdvancedFrequentItemsetMiner:
    def __init__(self, filepath):
        """
        Initialize the Advanced Frequent Itemset Miner
        
        Parameters:
        - filepath: Path to the CSV file
        """
        # Read the dataset
        self.df = pd.read_csv(filepath)
        
        # Preprocessing methods
        self.preprocess_data()
    
    def preprocess_data(self):
        """
        Comprehensive preprocessing of the dataset
        - Categorize continuous variables
        - Prepare categorical columns for FIM
        """
        # Delay categorization
        self.df['Departure_Delay_Category'] = pd.cut(
            self.df['Departure_Delay'], 
            bins=[-float('inf'), 15, 45, float('inf')],
            labels=['On-time', 'Minor Delay', 'Significant Delay']
        )
        
        self.df['Arrival_Delay_Category'] = pd.cut(
            self.df['Arrival_Delay'], 
            bins=[-float('inf'), 15, 45, float('inf')],
            labels=['On-time', 'Minor Delay', 'Significant Delay']
        )
        
        # Flight Duration Categorization (modify to handle HH:MM format)
        def categorize_duration(duration):
            hours, minutes = map(int, duration.split(':'))
            total_hours = hours + minutes / 60
            
            if total_hours < 1:
                return 'Short'
            elif 1 <= total_hours < 2:
                return 'Medium'
            else:
                return 'Long'
        
        self.df['Flight_Duration_Category'] = self.df['Flight_Duration'].apply(categorize_duration)
    
    def create_transactions(self):
        """
        Create transactions for Frequent Itemset Mining
        
        Returns:
        - List of transactions
        """
        # Carefully selected columns for comprehensive analysis
        selected_columns = [
            # Demographic & Travel
            'Gender', 'Nationality', 'Travel_Purpose', 
            
            # Flight & Service Information
            'Airline_Name', 'Class', 'Departure_City', 
            'Arrival_City', 'Flight_Route_Type',
            'Departure_Delay_Category', 'Arrival_Delay_Category',
            'Flight_Duration_Category',
            
            # Service Ratings
            'Seat_Comfort', 'InFlight_Entertainment', 
            'Food_Quality', 'Cleanliness', 
            'Cabin_Staff_Service', 'Legroom', 
            'Baggage_Handling', 'CheckIn_Service', 
            'Boarding_Process', 'WiFi_Service',
            
            # Customer Outcomes
            'Overall_Satisfaction', 'Recommendation', 
            'Complaint_Submitted', 'Complaint_Type',
            
            # Loyalty & Transactional Behavior
            'Seat_Type', 'Loyalty_Membership', 
            'Frequent_Flyer', 'Airline_Loyalty_Program',
            'Baggage_Lost', 'Compensation_Received',
            'Booking_Channel', 'Seat_Upgrade', 
            'Special_Assistance', 'Discount_Received',
            'Preferred_Airline', 'Frequent_Route', 
            'Payment_Method'
        ]
        
        # Create transactions
        transactions = self.df[selected_columns].apply(
            lambda row: [f"{col}_{val}" for col, val in row.items() if pd.notna(val)], 
            axis=1
        ).tolist()
        
        return transactions
    
    def perform_frequent_itemset_mining(
        self, 
        transactions, 
        min_support=0.05, 
        max_len=3
    ):
        """
        Perform Frequent Itemset Mining using Apriori algorithm
        
        Parameters:
        - transactions: List of transactions
        - min_support: Minimum support threshold
        - max_len: Maximum length of itemsets
        
        Returns:
        - Frequent itemsets
        - Association rules
        """
        # Transform transactions to one-hot encoded format
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        te_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(
            te_df, 
            min_support=min_support, 
            use_colnames=True, 
            max_len=max_len
        )
        
        # Generate association rules
        rules = association_rules(
            frequent_itemsets, 
            metric="confidence", 
            min_threshold=0.5
        )
        
        return frequent_itemsets, rules
    
    def visualize_network_graph(self, frequent_itemsets, rules):
        """
        Create a network graph of frequent itemsets and associations
        
        Parameters:
        - frequent_itemsets: DataFrame of frequent itemsets
        - rules: DataFrame of association rules
        """
        plt.figure(figsize=(20, 20))
        G = nx.Graph()
        
        unique_items = set()
        
        # Add nodes from top frequent itemsets
        top_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(30)
        for itemset in top_itemsets['itemsets']:
            for item in itemset:
                cleaned_item = item.replace('_', ' ')
                unique_items.add(cleaned_item)
                G.add_node(cleaned_item)
        
        # Add edges from association rules
        top_rules = rules.sort_values('lift', ascending=False).head(200)
        for _, rule in top_rules.iterrows():
            for ant in rule['antecedents']:
                for con in rule['consequents']:
                    ant_name = ant.replace('_', ' ')
                    con_name = con.replace('_', ' ')
                    
                    # Add nodes if not already present
                    if ant_name not in G.nodes:
                        G.add_node(ant_name)
                    if con_name not in G.nodes:
                        G.add_node(con_name) 
                    
                    # Add edge with lift as weight
                    G.add_edge(ant_name, con_name, weight=rule['lift'])
        
        # Draw the graph
        pos = nx.spring_layout(G, k=0.9, iterations=100)
        
        # Draw the graph with more detailed settings
        plt.figure(figsize=(20, 20))
        
        # Draw edges first with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v].get('weight', 1) for u, v in edges]
    
        # Normalize edge weights for visualization
        max_weight = max(weights) if weights else 1
        normalized_weights = [2 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(
            G, 
            pos, 
            edge_color='gray', 
            width=normalized_weights, 
            alpha=0.5
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color='lightblue', 
            node_size=500, 
            alpha=0.8
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, 
            pos, 
            font_size=8, 
            font_weight='bold'
        )
    
        plt.title('Network of Frequent Itemsets and Associations', fontsize=16)
        plt.axis('off')  # Hide axes
        plt.tight_layout()
        plt.savefig('itemset_network_graph.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_heatmap(self, frequent_itemsets):
        """
        Create a heatmap of frequent itemsets
        
        Parameters:
        - frequent_itemsets: DataFrame of frequent itemsets
        """
        # Prepare data for heatmap
        top_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(20)
        
        # Extract unique items
        all_items = set()
        for itemset in top_itemsets['itemsets']:
            all_items.update(item.replace('_', ' ') for item in itemset)
        
        # Create heatmap matrix
        heatmap_data = np.zeros((len(top_itemsets), len(all_items)))
        items_list = list(all_items)
        
        for i, row in enumerate(top_itemsets['itemsets']):
            for item in row:
                item_name = item.replace('_', ' ')
                j = items_list.index(item_name)
                heatmap_data[i, j] = 1
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            heatmap_data, 
            xticklabels=items_list, 
            yticklabels=[' & '.join(list(x)) for x in top_itemsets['itemsets']],
            cmap='YlGnBu',
            cbar_kws={'label': 'Presence in Itemset'}
        )
        plt.title('Frequent Itemsets Composition Heatmap', fontsize=16)
        plt.xlabel('Items', fontsize=12)
        plt.ylabel('Itemsets', fontsize=12)
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig('itemsets_heatmap.png', dpi=300)
        plt.close()
    
    def visualize_association_rules(self, rules):
        """
        Create a scatter plot of association rules
        
        Parameters:
        - rules: DataFrame of association rules
        """
        plt.figure(figsize=(15, 10))
        plt.scatter(
            rules['support'], 
            rules['confidence'], 
            alpha=0.5, 
            c=rules['lift'], 
            cmap='viridis'
        )
        plt.colorbar(label='Lift')
        plt.xlabel('Support', fontsize=12)
        plt.ylabel('Confidence', fontsize=12)
        plt.title('Association Rules: Support vs Confidence', fontsize=16)
        plt.tight_layout()
        plt.savefig('association_rules_scatter.png', dpi=300)
        plt.close()
    
    def generate_human_readable_insights(self, frequent_itemsets, rules):
        """
        Translate technical mining results into human-friendly insights
        
        Parameters:
        - frequent_itemsets: DataFrame of frequent itemsets
        - rules: DataFrame of association rules
        
        Returns:
        - Comprehensive human-readable report
        """
        # Initialize report sections
        report = {
            'summary': [],
            'customer_segments': [],
            'service_quality': [],
            'operational_patterns': [],
            'key_recommendations': []
        }
        
        # Overall Summary
        report['summary'].append(
            f"Our analysis examined {len(self.df)} flight records, "
            f"discovering {len(frequent_itemsets)} significant patterns "
            f"and {len(rules)} meaningful associations."
        )
        
        # Customer Segmentation Insights
        demographic_itemsets = frequent_itemsets[
            frequent_itemsets['itemsets'].apply(
                lambda x: any('Gender_' in item or 'Nationality_' in item or 'Travel_Purpose_' in item for item in x)
            )
        ].sort_values('support', ascending=False).head(5)
        
        customer_segments_insights = []
        for _, row in demographic_itemsets.iterrows():
            itemset = list(row['itemsets'])
            support = row['support']
            # Translate technical itemset to readable text
            readable_itemset = [item.replace('_', ' ') for item in itemset]
            insight = f"We found that {' and '.join(readable_itemset)} appears together in {support:.2%} of our flight records."
            report['customer_segments'].append(insight)
            customer_segments_insights.append(insight)
        
        # Service Quality Correlations
        service_rules = rules[
            rules['antecedents'].apply(
                lambda x: any('Seat_Comfort_' in item or 'InFlight_Entertainment_' in item or 'Food_Quality_' in item for item in x)
            ) &
            rules['consequents'].apply(
                lambda x: any('Overall_Satisfaction_' in item or 'Recommendation_' in item for item in x)
            )
        ].sort_values('lift', ascending=False).head(5)
        
        service_quality_insights = []
        for _, rule in service_rules.iterrows():
            antecedents = [item.replace('_', ' ') for item in list(rule['antecedents'])]
            consequents = [item.replace('_', ' ') for item in list(rule['consequents'])]
            confidence = rule['confidence']
            lift = rule['lift']
            
            insight = (
                f"Interestingly, when passengers experience {' and '.join(antecedents)}, "
                f"they are {confidence:.2%} likely to result in {' and '.join(consequents)}. "
                f"This correlation is {lift:.2f} times more probable than random chance."
            )
            report['service_quality'].append(insight)
            service_quality_insights.append(insight)
        
        # Operational Patterns
        operational_itemsets = frequent_itemsets[
            frequent_itemsets['itemsets'].apply(
                lambda x: any('Departure_Delay_Category_' in item or 'Arrival_Delay_Category_' in item or 'Flight_Duration_Category_' in item for item in x)
            )
        ].sort_values('support', ascending=False).head(5)
        
        operational_patterns_insights = []
        for _, row in operational_itemsets.iterrows():
            itemset = list(row['itemsets'])
            support = row['support']
            readable_itemset = [item.replace('_', ' ') for item in itemset]
            insight = f"We observed that {' and '.join(readable_itemset)} consistently occur together in {support:.2%} of our flight records."
            report['operational_patterns'].append(insight)
            operational_patterns_insights.append(insight)
        
        # Dynamic Recommendation Generation
        recommendations = []
        
        # Recommendations based on Customer Segments
        if customer_segments_insights:
            recommendations.append(
                f"Develop targeted marketing strategies for {customer_segments_insights[0].split(' appears')[0]}."
            )
        
        # Recommendations based on Service Quality
        if service_quality_insights:
            key_service_aspects = [insight.split(' when passengers experience ')[1].split(' they are')[0] for insight in service_quality_insights]
            if key_service_aspects:
                recommendations.append(
                    f"Prioritize improvements in {', '.join(key_service_aspects)} to enhance overall customer satisfaction."
                )
        
        # Recommendations based on Operational Patterns
        if operational_patterns_insights:
            key_operational_issues = [insight.split(' consistently occur')[0] for insight in operational_patterns_insights]
            if key_operational_issues:
                recommendations.append(
                    f"Address operational challenges related to {', '.join(key_operational_issues)} to improve flight experience."
                )
        
        # Generic recommendations if specific ones are not generated
        if not recommendations:
            recommendations = [
                "Conduct further in-depth analysis to uncover specific improvement opportunities.",
                "Continue monitoring and collecting detailed customer feedback.",
                "Invest in comprehensive staff training and service quality enhancement."
            ]
        
        # Add diverse and context-specific recommendations
        additional_recommendations = [
            "Leverage data-driven insights for continuous service improvement.",
            "Implement personalized customer experience strategies.",
            "Develop a proactive approach to managing customer expectations."
        ]
        
        # Combine and select unique recommendations
        report['key_recommendations'] = list(set(recommendations + additional_recommendations))
        
        return report
    
    def run_advanced_fim_analysis(self, min_support=0.05, max_len=3):
        """
        Run comprehensive Frequent Itemset Mining analysis
        
        Parameters:
        - min_support: Minimum support threshold
        - max_len: Maximum length of itemsets
        """
        # Create transactions
        transactions = self.create_transactions()
        
        # Perform FIM
        frequent_itemsets, rules = self.perform_frequent_itemset_mining(
            transactions, 
            min_support=min_support, 
            max_len=max_len
        )
        
        # Generate human-readable insights
        insights = self.generate_human_readable_insights(frequent_itemsets, rules)
        
        # Create visualizations
        print("\n=== GENERATING VISUALIZATIONS ===")
        self.visualize_network_graph(frequent_itemsets, rules)
        print("✓ Network Graph of Itemsets (itemset_network_graph.png)")
        
        self.visualize_heatmap(frequent_itemsets)
        print("✓ Frequent Itemsets Heatmap (itemsets_heatmap.png)")
        
        self.visualize_association_rules(rules)
        print("✓ Association Rules Scatter Plot (association_rules_scatter.png)")
        
        # Print insights (previous implementation)
        print("\n=== FLIGHT DATA INSIGHTS ===\n")
        
        print("1. OVERALL SUMMARY:")
        for summary in insights['summary']:
            print(f"   • {summary}")
        
        print("\n2. CUSTOMER SEGMENTS:")
        for segment in insights['customer_segments']:
            print(f"   • {segment}")
        
        print("\n3. SERVICE QUALITY INSIGHTS:")
        for quality in insights['service_quality']:
            print(f"   • {quality}")
        
        print("\n4. OPERATIONAL PATTERNS:")
        for pattern in insights['operational_patterns']:
            print(f"   • {pattern}")
        
        print("\n5. KEY RECOMMENDATIONS:")
        for recommendation in insights['key_recommendations']:
            print(f"   • {recommendation}")
        
        return insights

# Main execution
def main():
    # File path - modify as needed
    filepath = 'final_updated_dataset.csv'
    
    # Initialize and run Advanced FIM
    fim_miner = AdvancedFrequentItemsetMiner(filepath)
    fim_miner.run_advanced_fim_analysis(
        min_support=0.05,  # 5% minimum support 
        max_len=3  # Itemsets up to 3 items
    )

if __name__ == '__main__':
    main()