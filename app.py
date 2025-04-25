#!/usr/bin/env python
# coding: utf-8

# Burushaski Language Network Centrality Analysis
# ==============================================
# This program performs centrality analysis on a Burushaski language network

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import requests
from bs4 import BeautifulSoup
import random
import re
import os
import seaborn as sns

# Try to import Google Colab specific libraries
try:
    from google.colab import drive
    from google.colab import auth
    from googleapiclient.discovery import build
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Install missing dependencies if needed
try:
    import tabulate
except ImportError:
    import pip
    pip.main(['install', 'tabulate'])
    import tabulate

# Constants
FOLDER_NAME = "BURUSHASHKI DATASET"
FILE_NAME = "new-roman-burushaski-words-v1"
SHEET_ID = "1zXjVkMNb459J6_fVrsSBGfYm8F9pEXkv9qJwh5TJi18"

WORLD_LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian",
    "Chinese", "Arabic", "Hindi", "Bengali", "Japanese", "Korean", "Turkish",
    "Persian", "Urdu", "Swahili", "Hebrew", "Greek", "Latin", "Dutch", "Polish",
    "Swedish", "Norwegian", "Finnish", "Czech", "Hungarian", "Romanian", "Thai",
    "Vietnamese", "Indonesian", "Tagalog", "Malay", "Zulu", "Xhosa", "Tamil",
    "Telugu", "Kannada", "Gujarati", "Punjabi", "Marathi", "Armenian", "Georgian",
    "Kazakh", "Uzbek", "Azerbaijani", "Pashto", "Kurdish", "Yoruba", "Igbo"
]

class BurushaskiAnalyzer:
    def __init__(self):
        """Initialize the analyzer with authentication and setup."""
        self.initialize_google_drive()
        self.WORLD_LANGUAGES = WORLD_LANGUAGES

    def initialize_google_drive(self):
        """Authenticate and mount Google Drive if in Colab environment."""
        if IN_COLAB:
            print("Authenticating with Google Drive...")
            auth.authenticate_user()
            drive.mount('/content/drive')
            print("Drive mounted successfully.")
        else:
            print("Not running in Google Colab. Skipping Drive mount.")

    def load_burushaski_data(self):
        """
        Load Burushaski words from Google Drive or Sheets.

        Returns:
            set: A set of Burushaski words in lowercase
        """
        print("\nLoading Burushaski words...")

        if IN_COLAB:
            # Try to load from Google Drive file first
            drive_path = f"/content/drive/MyDrive/{FOLDER_NAME}/{FILE_NAME}"
            if os.path.exists(drive_path):
                try:
                    if drive_path.endswith('.xlsx'):
                        df = pd.read_excel(drive_path)
                    else:  # Assume CSV
                        df = pd.read_csv(drive_path)
                    words = set(df.iloc[:, 0].dropna().str.lower().str.strip())
                    print(f"Loaded {len(words)} words from Google Drive file")
                    return words
                except Exception as e:
                    print(f"Couldn't read file directly: {e}")

            # Fall back to Google Sheets API
            print("Accessing via Google Sheets API...")
            try:
                sheet_service = build('sheets', 'v4')
                sheet = sheet_service.spreadsheets()
                result = sheet.values().get(
                    spreadsheetId=SHEET_ID,
                    range="A:Z"
                ).execute()
                values = result.get('values', [])

                if not values:
                    raise ValueError("No data found in spreadsheet")

                # Convert to DataFrame (first row as headers)
                df = pd.DataFrame(values[1:], columns=values[0])
                words = set(df.iloc[:, 0].dropna().str.lower().str.strip())
                print(f"Loaded {len(words)} words from Google Sheets")
                return words

            except Exception as e:
                print(f"Failed to load data: {e}")

        # If not in Colab or all methods failed, use sample data
        print("Using sample Burushaski word data...")
        # Generate some sample Burushaski words for testing
        sample_words = {f"burushaski_word_{i}" for i in range(100)}
        return sample_words

    def fetch_language_words(self, language, max_words=500):
        """
        Fetch words for a given language from online sources.

        Args:
            language (str): Name of the language to fetch
            max_words (int): Maximum number of words to return

        Returns:
            set: Words in the target language
        """
        print(f"Fetching words for {language}...")
        words = set()

        # Try multiple sources
        sources = [
            self._fetch_from_wiktionary,
            self._fetch_from_multi_translate,
            self._fetch_from_freelang
        ]

        for source in sources:
            try:
                if len(words) >= max_words:
                    break
                new_words = source(language)
                if new_words:
                    words.update(new_words)
            except Exception as e:
                print(f"Error with {source.__name__} for {language}: {e}")

        # If online fetching failed or not enough words, generate random ones for testing
        if len(words) < 20:
            print(f"Generating sample words for {language}...")
            words.update({f"{language.lower()}_word_{i}" for i in range(200)})

        # Limit the number of words
        if len(words) > max_words:
            words = set(random.sample(list(words), max_words))

        return words

    def _fetch_from_wiktionary(self, language):
        """Fetch words from Wiktionary frequency lists."""
        url = f"https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/{language.title()}_wordlist"
        response = requests.get(url, timeout=10)
        words = set()

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract from tables
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        word = cells[0].get_text().strip().lower()
                        if self._is_valid_word(word):
                            words.add(word)

            # Extract from lists
            for list_elem in soup.find_all(['ul', 'ol']):
                for item in list_elem.find_all('li'):
                    text = item.get_text().strip()
                    if text:
                        word = text.split()[0].lower()
                        if self._is_valid_word(word):
                            words.add(word)

        return words

    def _fetch_from_multi_translate(self, language):
        """Fetch words from various translation websites."""
        urls = [
            f"https://1000mostcommonwords.com/1000-most-common-{language.lower()}-words/",
            f"https://www.101languages.net/common-words/{language.lower()}-words/"
        ]
        words = set()

        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for table in soup.find_all('table'):
                        for row in table.find_all('tr'):
                            for cell in row.find_all(['td', 'th']):
                                word = cell.get_text().strip().lower()
                                if self._is_valid_word(word):
                                    words.add(word)
            except:
                continue

        return words

    def _fetch_from_freelang(self, language):
        """Fetch words from freelang.net dictionaries."""
        url = f"https://www.freelang.net/dictionary/{language.lower()}.php"
        words = set()

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for elem in soup.find_all(['li', 'p', 'span']):
                    text = elem.get_text().strip()
                    for word in re.findall(r'\b\w+\b', text):
                        word = word.lower()
                        if self._is_valid_word(word):
                            words.add(word)
        except:
            pass

        return words

    def _is_valid_word(self, word):
        """Check if a word is valid for inclusion."""
        return (len(word) > 1 and
                not word.isdigit() and
                not bool(re.search(r'[0-9]', word)) and
                word.isalpha())

    def build_network(self, burushaski_words, language_words_dict):
        """
        Build a network graph connecting languages and shared words.

        Args:
            burushaski_words (set): Burushaski vocabulary
            language_words_dict (dict): Other languages and their words

        Returns:
            networkx.Graph: The constructed language network
        """
        print("\nBuilding language network...")
        G = nx.Graph()

        # Add Burushaski node
        G.add_node("Burushaski", type='language', color='red', size=800)

        # Add Burushaski words
        for word in tqdm(burushaski_words, desc="Adding Burushaski words"):
            G.add_node(word, type='word', color='green', size=30)
            G.add_edge("Burushaski", word)

        # Add other languages and their connections
        for lang, words in tqdm(language_words_dict.items(), desc="Adding other languages"):
            G.add_node(lang, type='language', color='red', size=800)
            common_words = burushaski_words.intersection(words)

            for word in common_words:
                G.add_edge(lang, word)

        print(f"\nNetwork built with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G

    def visualize_network(self, G, title="Language Network", layout='spring'):
        """Visualize the language-word network."""
        plt.figure(figsize=(16, 12))

        # Get node attributes
        colors = [data['color'] for _, data in G.nodes(data=True)]
        sizes = [data['size'] for _, data in G.nodes(data=True)]
        labels = {node: node if data['type'] == 'language' else ''
                 for node, data in G.nodes(data=True)}

        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)

        # Draw the network
        nx.draw_networkx(
            G, pos,
            node_color=colors,
            node_size=sizes,
            edge_color='gray',
            alpha=0.7,
            with_labels=True,
            labels=labels,
            font_size=10,
            font_weight='bold'
        )

        plt.title(title, fontsize=18)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def analyze_similarities(self, G):
        """Calculate and visualize language similarity matrix."""
        languages = [n for n in G.nodes() if G.nodes[n]['type'] == 'language']
        similarity = np.zeros((len(languages), len(languages)))

        # Calculate Jaccard similarity
        for i, lang1 in enumerate(languages):
            lang1_words = set(nx.neighbors(G, lang1))
            for j, lang2 in enumerate(languages):
                if i < j:
                    lang2_words = set(nx.neighbors(G, lang2))
                    intersection = len(lang1_words & lang2_words)
                    union = len(lang1_words | lang2_words)
                    similarity[i,j] = intersection / union if union else 0
                    similarity[j,i] = similarity[i,j]

        # Create similarity DataFrame
        sim_df = pd.DataFrame(similarity, index=languages, columns=languages)

        # Visualize
        plt.figure(figsize=(12, 10))
        plt.imshow(similarity, cmap='viridis')
        plt.colorbar(label='Jaccard Similarity')
        plt.xticks(range(len(languages)), languages, rotation=90)
        plt.yticks(range(len(languages)), languages)
        plt.title('Language Similarity Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()

        return sim_df

    def run_analysis(self, num_languages=10):
        """Run complete analysis pipeline."""
        # Load Burushaski data
        burushaski_words = self.load_burushaski_data()
        if not burushaski_words:
            print("Error: Failed to load Burushaski words")
            return

        # Select languages to compare
        selected_langs = random.sample(WORLD_LANGUAGES, min(num_languages, len(WORLD_LANGUAGES)))
        print(f"\nSelected {len(selected_langs)} languages for comparison:")
        print(", ".join(selected_langs))

        # Fetch words for each language
        lang_words = {}
        for lang in tqdm(selected_langs, desc="Fetching language data"):
            words = self.fetch_language_words(lang)
            if words:
                lang_words[lang] = words

        # Build and visualize network
        G = self.build_network(burushaski_words, lang_words)
        self.visualize_network(G, "Burushaski Language Connections")

        # Analyze similarities
        sim_df = self.analyze_similarities(G)

        # Show top connections
        print("\nTop language connections:")
        connections = []
        for lang in lang_words:
            common = len(burushaski_words & lang_words[lang])
            connections.append((lang, common))

        connections.sort(key=lambda x: x[1], reverse=True)
        for lang, count in connections[:10]:
            print(f"{lang}: {count} shared words")

        return G


class BurushaskiCentralityAnalyzer(BurushaskiAnalyzer):
    """
    Extended class for analyzing centrality measures in the Burushaski language network.
    """

    def calculate_centralities(self, G):
        """
        Calculate various centrality measures for all nodes in the network.

        Args:
            G (networkx.Graph): The network graph

        Returns:
            pd.DataFrame: DataFrame containing centrality values for each node
        """
        print("\nCalculating centrality measures...")

        # Dictionary to store centrality measures
        centrality_measures = {}

        # 1. Degree Centrality
        # Reference: NetworkX library - https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.degree_centrality.html
        print("Computing Degree Centrality...")
        centrality_measures['degree'] = nx.degree_centrality(G)

        # 2. Betweenness Centrality
        # Reference: NetworkX library - https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
        print("Computing Betweenness Centrality...")
        centrality_measures['betweenness'] = nx.betweenness_centrality(G)

        # 3. Closeness Centrality
        # Reference: NetworkX library - https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
        print("Computing Closeness Centrality...")
        centrality_measures['closeness'] = nx.closeness_centrality(G)

        # 4. Eigenvector Centrality
        # Reference: NetworkX library - https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html
        print("Computing Eigenvector Centrality...")
        try:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, max_iter=300)
        except nx.PowerIterationFailedConvergence:
            print("Warning: Eigenvector centrality failed to converge. Using default values.")
            centrality_measures['eigenvector'] = {node: 0.01 for node in G.nodes()}

        # 5. PageRank Centrality
        # Reference: NetworkX library - https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
        print("Computing PageRank Centrality...")
        centrality_measures['pagerank'] = nx.pagerank(G, alpha=0.85)

        # Convert to DataFrame
        centrality_df = pd.DataFrame(centrality_measures)

        # Normalize centrality values (already normalized by NetworkX for most measures)
        # but ensuring consistency across all measures
        for column in centrality_df.columns:
            if centrality_df[column].max() > 0:
                centrality_df[column] = centrality_df[column] / centrality_df[column].max()

        return centrality_df

    def aggregate_centralities(self, centrality_df):
        """
        Aggregate the centrality measures by taking the average of normalized values.

        Args:
            centrality_df (pd.DataFrame): DataFrame with normalized centrality measures

        Returns:
            pd.DataFrame: DataFrame with original and aggregated centrality values
        """
        print("\nAggregating centrality measures...")

        # Calculate the aggregated centrality (mean of all centrality measures)
        centrality_df['aggregated'] = centrality_df.mean(axis=1)

        return centrality_df

    def visualize_centralities(self, G, centrality_df, measure='aggregated', top_n=20):
        """
        Visualize the network with node sizes based on centrality values.

        Args:
            G (networkx.Graph): The network graph
            centrality_df (pd.DataFrame): DataFrame with centrality values
            measure (str): Centrality measure to visualize
            top_n (int): Number of top nodes to label
        """
        plt.figure(figsize=(16, 12))

        # Create a copy of the graph
        viz_graph = G.copy()

        # Set node sizes based on centrality values
        centrality_values = centrality_df[measure].to_dict()

        # Scale centrality values for visualization (min_size=100, max_size=1000)
        min_size, max_size = 100, 2500
        min_val, max_val = min(centrality_values.values()), max(centrality_values.values())

        # Calculate scaled sizes
        for node in viz_graph.nodes():
            if node in centrality_values:
                if max_val > min_val:
                    scaled_size = min_size + (centrality_values[node] - min_val) * (max_size - min_size) / (max_val - min_val)
                else:
                    scaled_size = min_size
                viz_graph.nodes[node]['size'] = scaled_size
            else:
                viz_graph.nodes[node]['size'] = min_size

        # Extract color information
        colors = [data['color'] for _, data in viz_graph.nodes(data=True)]
        sizes = [data['size'] for _, data in viz_graph.nodes(data=True)]

        # Get positions using spring layout
        pos = nx.spring_layout(viz_graph, k=0.5, iterations=100, seed=42)

        # Get top N nodes by centrality
        top_nodes = centrality_df.nlargest(top_n, measure).index.tolist()

        # Create labels only for top nodes
        labels = {node: node if node in top_nodes else '' for node in viz_graph.nodes()}

        # Draw the network
        nx.draw_networkx(
            viz_graph, pos,
            node_color=colors,
            node_size=sizes,
            edge_color='lightgray',
            alpha=0.8,
            with_labels=True,
            labels=labels,
            font_size=10,
            font_weight='bold'
        )

        plt.title(f'Network Visualization: {measure.capitalize()} Centrality', fontsize=18)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def report_top_nodes(self, centrality_df, measures=None, top_n=20):
        """
        Report the top N nodes for each centrality measure.

        Args:
            centrality_df (pd.DataFrame): DataFrame with centrality values
            measures (list): List of centrality measures to report
            top_n (int): Number of top nodes to report

        Returns:
            dict: Dictionary with top nodes for each measure
        """
        print(f"\nTop {top_n} nodes by centrality:")

        if measures is None:
            measures = centrality_df.columns.tolist()

        results = {}

        for measure in measures:
            top_nodes = centrality_df.nlargest(top_n, measure)
            print(f"\n--- {measure.capitalize()} Centrality ---")
            # Use the tabulate module for prettier output
            from tabulate import tabulate
            print(tabulate(
                top_nodes[[measure]].reset_index().rename(columns={'index': 'Node'}),
                headers='keys',
                tablefmt='pretty',
                showindex=True
            ))
            results[measure] = top_nodes.index.tolist()

        return results

    def visualize_centrality_distributions(self, centrality_df):
        """
        Visualize the distribution of centrality values.

        Args:
            centrality_df (pd.DataFrame): DataFrame with centrality values
        """
        plt.figure(figsize=(15, 10))

        # Set up subplots for each centrality measure
        measures = centrality_df.columns.tolist()
        n_measures = len(measures)
        n_rows = (n_measures + 1) // 2

        for i, measure in enumerate(measures, 1):
            plt.subplot(n_rows, 2, i)
            sns.histplot(centrality_df[measure], kde=True)
            plt.title(f'{measure.capitalize()} Centrality Distribution')
            plt.xlabel('Centrality Value')
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def save_results(self, centrality_df, filename='burushaski_centrality_results.csv'):
        """
        Save the centrality results to a CSV file.

        Args:
            centrality_df (pd.DataFrame): DataFrame with centrality values
            filename (str): Name of the file to save
        """
        centrality_df.to_csv(filename)
        print(f"\nResults saved to {filename}")

    def compare_language_centralities(self, G, centrality_df):
        """
        Compare centrality values specifically for language nodes.

        Args:
            G (networkx.Graph): The network graph
            centrality_df (pd.DataFrame): DataFrame with centrality values

        Returns:
            pd.DataFrame: DataFrame with centrality values for language nodes only
        """
        # Extract language nodes
        language_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'language']

        # Filter centrality DataFrame for language nodes
        language_centralities = centrality_df.loc[language_nodes]

        print("\nLanguage Node Centralities:")
        from tabulate import tabulate
        print(tabulate(
            language_centralities.reset_index().rename(columns={'index': 'Language'}),
            headers='keys',
            tablefmt='pretty',
            showindex=False
        ))

        # Plot language centrality comparison
        plt.figure(figsize=(14, 8))
        language_centralities_sorted = language_centralities.sort_values('aggregated', ascending=False)

        ax = sns.heatmap(
            language_centralities_sorted,
            cmap='viridis',
            annot=True,
            fmt='.3f',
            linewidths=.5
        )
        plt.title('Language Node Centrality Comparison', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return language_centralities

    def run_centrality_analysis(self, G=None, num_languages=10):
        """
        Run complete centrality analysis pipeline.

        Args:
            G (networkx.Graph, optional): Pre-built network graph
            num_languages (int): Number of languages to include if building a new graph

        Returns:
            tuple: (networkx.Graph, pd.DataFrame) - The graph and centrality DataFrame
        """
        # If no graph is provided, build one using the parent class methods
        if G is None:
            # Load Burushaski data
            burushaski_words = self.load_burushaski_data()
            if not burushaski_words:
                print("Error: Failed to load Burushaski words")
                return None, None

            # Select languages to compare
            selected_langs = random.sample(self.WORLD_LANGUAGES, min(num_languages, len(self.WORLD_LANGUAGES)))
            print(f"\nSelected {len(selected_langs)} languages for comparison:")
            print(", ".join(selected_langs))

            # Fetch words for each language
            lang_words = {}
            for lang in tqdm(selected_langs, desc="Fetching language data"):
                words = self.fetch_language_words(lang)
                if words:
                    lang_words[lang] = words

            # Build network
            G = self.build_network(burushaski_words, lang_words)

        # Calculate centrality measures
        centrality_df = self.calculate_centralities(G)

        # Aggregate centrality measures
        centrality_df = self.aggregate_centralities(centrality_df)

        # Report top nodes for each centrality measure
        self.report_top_nodes(centrality_df)

        # Visualize centrality distributions
        self.visualize_centrality_distributions(centrality_df)

        # Visualize network with aggregated centrality
        self.visualize_centralities(G, centrality_df, measure='aggregated')

        # Compare language centralities
        language_centralities = self.compare_language_centralities(G, centrality_df)

        # Save results
        self.save_results(centrality_df)

        return G, centrality_df


# Main function to run directly
def main():
    print("Burushaski Language Network Centrality Analysis")
    print("=" * 50)

    # Create analyzer
    centrality_analyzer = BurushaskiCentralityAnalyzer()

    # Either load an existing graph or build a new one
    print("\nInitiating network analysis...")
    G = centrality_analyzer.run_analysis(num_languages=15)

    # Run centrality analysis
    G, centrality_df = centrality_analyzer.run_centrality_analysis(G=G)

    # Optional: Run visualizations for specific centrality measures
    for measure in ['degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank']:
        centrality_analyzer.visualize_centralities(G, centrality_df, measure=measure)


# Execute if run as a script
if __name__ == "__main__":
    main()