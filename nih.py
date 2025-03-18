import requests
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import re

# Check and download required NLTK resources if missing
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt_tab'...")
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords'...")
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading 'wordnet'...")
    nltk.download('wordnet')

# Initialize stop words and lemmatizer for text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to sanitize text for Excel compatibility
def sanitize_text(text):
    if not isinstance(text, str):
        return text
    # Remove control characters that Excel can't handle
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

class NIHQuery:
    def __init__(self):
        # Define the NIH Reporter API endpoint
        self.url = 'https://api.reporter.nih.gov/v2/projects/search'
        self.grants = []  # Store fetched grant data

    def get_total_projects(self, search_query, fiscal_years):
        # Check total number of projects matching the query without fetching all data
        params = {
            'criteria': {
                'query': search_query,  # User-defined search terms
                'text_fields': ["project_title"],  # Search only in titles
                'fiscal_years': fiscal_years  # User-specified years
            },
            'offset': 0,  # Start at beginning
            'limit': 1  # Fetch minimal data to get total count
        }
        try:
            resp = requests.post(self.url, json=params)
            if resp.status_code == 200:
                json_data = resp.json()
                return json_data.get('meta', {}).get('total', 0)  # Extract total from metadata
            else:
                print(f"Error checking total projects: Status code {resp.status_code}")
                return 0
        except Exception as e:
            print("Error during total projects check:", e)
            return 0

    def fetch_data(self, search_query, fiscal_years, limit=500, total_limit=30000):
        offset = 0  # Pagination starting point
        self.grants = []  # Reset grants list
        while offset < total_limit:
            print(f"Fetching results... Offset: {offset}")
            params = {
                'criteria': {
                    'query': search_query,  # User-defined search terms
                    'text_fields': ["project_title"],  # Search only in titles
                    'fiscal_years': fiscal_years  # User-specified years
                },
                'offset': offset,  # Pagination offset
                'limit': min(limit, total_limit - offset)  # Batch size
            }
            try:
                resp = requests.post(self.url, json=params)  # API POST request
                if resp.status_code != 200:  # Check for successful response
                    break
                json_data = resp.json()
                if 'results' not in json_data:  # Verify data structure
                    break
                projects = json_data.get('results', [])
                self.grants.extend(projects)  # Append fetched projects
                if not projects:  # Stop if no more results
                    break
                offset += len(projects)  # Update offset for next batch
                time.sleep(1)  # Delay to respect API rate limits
            except Exception as e:
                print("Error during API request:", e)
                break

# Preprocess text for topic modeling
def preprocess_text(text):
    # Tokenize, lowercase, lemmatize, and remove stop words/nonalphabetic tokens
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

# Perform Latent Dirichlet Allocation (LDA) topic modeling
def perform_topic_modeling(titles, num_topics=5):
    processed_titles = [preprocess_text(title) for title in titles]  # Preprocess all titles
    dictionary = corpora.Dictionary(processed_titles)  # Create word-to-id mapping
    corpus = [dictionary.doc2bow(text) for text in processed_titles]  # Convert to bag-of-words
    
    # Train LDA model to identify topics
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    # Evaluate topic coherence (higher is better)
    coherence_model = CoherenceModel(model=lda_model, texts=processed_titles, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score for {num_topics} topics: {coherence_score}")
    
    return lda_model, dictionary, corpus

# Assign categories to titles based on dominant LDA topics
def assign_categories(lda_model, corpus, titles, topic_labels):
    categories = []
    for i, title in enumerate(titles):
        bow = lda_model.id2word.doc2bow(preprocess_text(title))  # Convert title to bag-of-words
        topic_probs = lda_model.get_document_topics(bow)  # Get topic probabilities
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]  # Select most likely topic
        categories.append(topic_labels[dominant_topic])  # Assign corresponding label
    return categories

def get_user_input():
    # Prompt user for search keywords
    keywords = input("Enter search keywords (separate multiple with commas): ").strip()
    search_query = " OR ".join(keyword.strip() for keyword in keywords.split(","))  # Join with OR for API query
    
    # Prompt user for fiscal years
    years_input = input("Enter fiscal years (e.g., 2023-2025 or 2023, 2024): ").strip()
    if "-" in years_input:  # Handle range input
        start, end = map(int, years_input.split("-"))
        fiscal_years = list(range(start, end + 1))
    else:  # Handle comma-separated list
        fiscal_years = [int(year.strip()) for year in years_input.split(",")]
    
    return search_query, fiscal_years

def main():
    # Get user-defined search query and fiscal years
    search_query, fiscal_years = get_user_input()
    print(f"Search Query: {search_query}")
    print(f"Fiscal Years: {fiscal_years}")
    
    # Initialize NIHQuery and check total projects
    rq = NIHQuery()
    total_projects = rq.get_total_projects(search_query, fiscal_years)
    print(f"Total projects identified: {total_projects}")
    
    # Prompt user to confirm fetching
    proceed = input(f"Would you like to fetch {total_projects} projects? (yes/no): ").strip().lower()
    if proceed != "yes":
        print("Operation cancelled by user.")
        return
    
    # Fetch data from NIH API if user agrees
    rq.fetch_data(search_query, fiscal_years)
    print(f"Total projects fetched: {len(rq.grants)}")
    
    # Extract project titles, filtering out missing ones
    titles = [grant.get("project_title", "") for grant in rq.grants if grant.get("project_title")]
    if not titles:
        print("No titles found.")
        return
    
    # Perform topic modeling on titles
    num_topics = 5  # Adjustable number of topics
    lda_model, dictionary, corpus = perform_topic_modeling(titles, num_topics)
    
    # Generate descriptive labels from top words in each topic
    topic_labels = []
    for topic_id in range(num_topics):
        top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=5)]
        label = " & ".join(top_words)  # Combine top words into a label
        topic_labels.append(label)
    
    # Assign categories to each project based on topics
    categories = assign_categories(lda_model, corpus, titles, topic_labels)
    
    # Build structured project data with hyperlinks
    project_data = []
    base_url = "https://reporter.nih.gov/project-details/"
    for grant, category in zip(rq.grants, categories):
        award_date = grant.get("award_notice_date", "Unknown")  # Extract dates with defaults
        start_date = grant.get("project_start_date", "Unknown")
        end_date = grant.get("project_end_date", "Unknown")
        
        project_num = sanitize_text(grant.get("project_num", "Unknown"))  # Get project number
        # Create Excel-compatible hyperlink if project number exists
        project_link = (f'=HYPERLINK("{base_url}{project_num}", "{project_num}")' 
                       if project_num != "Unknown" else "Unknown")

        project_data.append({
            "Project Number": project_link,
            "Project Title": sanitize_text(grant.get("project_title", "Unknown")),
            "Abstract": sanitize_text(grant.get("abstract_text", "No abstract available")),
            "Funding Category": sanitize_text(category),
            "Award Notice Date": award_date.split("T")[0] if isinstance(award_date, str) else award_date,
            "Project Start Date": start_date.split("T")[0] if isinstance(start_date, str) else start_date,
            "Project End Date": end_date.split("T")[0] if isinstance(end_date, str) else end_date,
            "Total Cost": grant.get("total_cost", 0),
            "Direct Cost": grant.get("direct_cost_amt", 0),
            "Indirect Cost": grant.get("indirect_cost_amt", 0),
            "Organization": sanitize_text(grant.get("organization", {}).get("org_name", "Unknown")),
            "State": sanitize_text(grant.get("organization", {}).get("org_state", "Unknown")),
            "Principal Investigator": sanitize_text(grant.get("principal_investigators", [{}])[0].get("full_name", "Unknown")),
            "PI Contact": sanitize_text(grant.get("contact_pi_name", "Unknown")),
            "Awarding Institute": sanitize_text(grant.get("agency_ic_admin", "Unknown")),
            "Funding Mechanism": sanitize_text(grant.get("funding_mechanism", "Unknown")),
            "CFDA Code": sanitize_text(grant.get("cfda_code", "Unknown")),
            "Fiscal Year": grant.get("fiscal_year", "Unknown")
        })
    
    # Convert to DataFrame for easy manipulation and export
    df = pd.DataFrame(project_data)
    if df.empty:
        print("No valid projects found.")
        return
    
    # Display distribution of funding categories
    print("\nCategory Distribution:")
    print(df['Funding Category'].value_counts())
    
    # Export to Excel with hyperlinks intact
    df.to_excel("NIH_Funding.xlsx", index=False)
    print("Data saved to NIH_Funding.xlsx")

if __name__ == "__main__":
    main()