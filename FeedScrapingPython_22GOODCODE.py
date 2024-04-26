import feedparser
import csv
import os
import requests
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

class RSSHelper:
    def __init__(self):
        pass

    def get_rss_entries(self, rss_feed):
        rss_feed = feedparser.parse(rss_feed)
        return rss_feed.entries

def generate_summaries(link):
    try:
        # Fetch the webpage content
        response = requests.get(link)
        if response.status_code == 200:
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')

            # Extract text from the webpage
            text = ' '.join([p.get_text() for p in soup.find_all('p')])

            # Generate a huge summary
            huge_summary = text

            # Generate a concise summary
            parser = HtmlParser.from_string(content, link, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            concise_summary = ""
            for sentence in summarizer(parser.document, 3):
                concise_summary += str(sentence) + " "

        return huge_summary, concise_summary
    except Exception as e:
        print(f"\033[91mError fetching content from {link}: {e}\033[0m")
        return '', ''

def compute_semantic_similarity(sentence1, sentence2):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Compute embeddings for both sentences
    embedding_1 = model.encode(sentence1, convert_to_tensor=True)
    embedding_2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embedding_1, embedding_2)
    
    return similarity_score.item()

def sentiment_analysis(text):
    if text is None:
        return 'Neutral', 0.0  # Return neutral sentiment if text is None

    sentiment_score = sia.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        return 'Positive', sentiment_score
    elif sentiment_score <= -0.05:
        return 'Negative', sentiment_score
    else:
        return 'Neutral', sentiment_score

def main():
    rss_feed_url = 'https://finance.yahoo.com/news/rssindex'

    rsh = RSSHelper()
    entries = rsh.get_rss_entries(rss_feed_url)

    # List of purposes
    purposes = [
        {'category': '1. Learning and growth Perspective', 'purpose': 'Manage our information as a strategic asset.'},
        {'category': '1. Learning and growth Perspective', 'purpose': 'Re-skill our staff to sell and support total financial solutions'},
        {'category': '1. Learning and growth Perspective', 'purpose': 'Develop a performance-focused culture'},
        {'category': '1. Learning and growth Perspective', 'purpose': 'Develop personal performance management processes that reward performance and risk taking'},
        {'category': '2. Internal process perspective', 'purpose': 'Build market awareness by segment'},
        {'category': '2. Internal process perspective', 'purpose': 'Drive execution of the sales process'},
        {'category': '3. Customer perspective', 'purpose': 'Provide me with good value and innovative financial solutions'},
        {'category': '4. Financial perspective', 'purpose': 'Grow income in key segments'},
        {'category': '5. Internal process perspective', 'purpose': 'Develop a 360 view of customers'},
        {'category': '5. Internal process perspective', 'purpose': 'Target existing customers with new offerings'},
        {'category': '6. Customer perspective', 'purpose': 'Be my valued, trusted financial provider'},
        {'category': '7. Financial perspective', 'purpose': 'Increase average share of wallet'},
        {'category': '7. Financial perspective', 'purpose': 'Increase shareholder value'},
        {'category': '8. Internal process perspective', 'purpose': 'Reduce cost by exploiting online channel and reducing branch network'},
        {'category': '8. Internal process perspective', 'purpose': 'Manage the balance of funding'},
        {'category': '9. Customer perspective', 'purpose': 'Provide me with low cost, convenient service'},
        {'category': '10. Financial perspective', 'purpose': 'Achieve the lowest cost of funds and cost to serve in the industry'},
    ]

    data = []
    for entry in entries:
        title = entry.get('title', '')
        link = entry.get('link', '')
        source = entry.get('source', {}).get('title', '')
        published = entry.get('published', '')
        huge_summary, concise_summary = generate_summaries(link)

        if huge_summary:
            for purpose_info in purposes:
                purpose = purpose_info['purpose']
                category = purpose_info['category']

                # Semantic similarity check
                similarity = compute_semantic_similarity(purpose, concise_summary)

                # Sentiment analysis
                sentiment, sentiment_score = sentiment_analysis(huge_summary)

                # Topic modeling for concise summary
                vectorizer = CountVectorizer(stop_words='english')
                X = vectorizer.fit_transform([concise_summary])
                lda = LatentDirichletAllocation(n_components=1, random_state=42)
                lda.fit(X)
                topics = [vectorizer.get_feature_names_out()[i] for i in lda.components_[0].argsort()[-10:][::-1]]

                # KPI, KRI, KCI
                if sentiment == 'Positive':
                    kpi_level = 'KPI' if similarity >= 0.5 else ''
                    kri_level = ''
                    kci_level = ''
                    kpi_low_level = ', '.join(topics) if 0.2 < similarity < 0.5 else ''
                    kri_low_level = ''
                    kci_low_level = ''
                elif sentiment == 'Negative':
                    kpi_level = ''
                    kri_level = 'KRI' if similarity >= 0.5 else ''
                    kci_level = ''
                    kpi_low_level = ''
                    kri_low_level = ', '.join(topics) if 0.2 < similarity < 0.5 else ''
                    kci_low_level = ''
                else:  # Neutral
                    kpi_level = ''
                    kri_level = ''
                    kci_level = 'KCI' if similarity >= 0.5 else ''
                    kpi_low_level = ''
                    kri_low_level = ''
                    kci_low_level = ', '.join(topics) if 0.2 < similarity < 0.5 else ''

                data.append({'Title': title, 'Link': link, 'Source': source, 'Published Date': published, 
                             'Huge Summary': huge_summary, 'Concise Summary': concise_summary, 'Category': category, 'Purpose': purpose, 
                             'Similarity Score': similarity, 'Indicators Trend': sentiment, 'Sentiment Score': sentiment_score,
                             'Subjects Similarity': 'Similar' if similarity >= 0.5 else 'Unsimilar',
                             'KPI Low Level': kpi_low_level, 'KRI Low Level': kri_low_level, 'KCI Low Level': kci_low_level})

    # Write data to CSV
    csv_file_path = os.path.join('C:\\Users\\USER\\Documents\\Consulting\\ADIT\\Cas D\'étude', 'rss_monitoring.csv')
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Link', 'Source', 'Published Date', 'Huge Summary', 'Concise Summary', 'Category', 'Purpose', 
                      'Similarity Score', 'Indicators Trend', 'Sentiment Score', 'Subjects Similarity', 
                      'KPI Low Level', 'KRI Low Level', 'KCI Low Level']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV file successfully created at: {csv_file_path}")

if __name__ == "__main__":
    main()


