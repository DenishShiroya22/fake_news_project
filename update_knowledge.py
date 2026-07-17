import pandas as pd
import os
import feedparser
def run_pipeline():
    print("🚀 Starting Hybrid Knowledge Base Pipeline...")
    
    # --- STEP 1: LOAD OR CREATE THE BASELINE ---
    csv_file = 'massive_knowledge_base.csv'
    
    if os.path.exists(csv_file):
        print(f"Found existing database: {csv_file}. Loading...")
        kb_df = pd.read_csv(csv_file)
    else:
        print("No existing database found. Initializing new repository from Tier 1 baseline...")
        
        # Checking if your downloaded dataset is available
        try:
            # Swap 'dataset/indian_news.xlsx' with your actual file path if different
            df = pd.read_excel('dataset/indian_news.xlsx')
            
            # Filter rows where the label is verified as TRUE
            df_true = df[df['Label'].astype(str).str.strip().str.lower() == 'true']
            kb_df = df_true[['Fact_Check_Source', 'Eng_Trans_Statement']].dropna()
            kb_df = kb_df.rename(columns={'Fact_Check_Source': 'source', 'Eng_Trans_Statement': 'fact'})
            print(f"✅ Extracted {len(kb_df)} baseline facts successfully.")
        except FileNotFoundError:
            print("⚠️ Baseline Excel file not found. Creating empty dataframe.")
            kb_df = pd.DataFrame(columns=['source', 'fact'])

    # --- STEP 2: THE AUTOMATION ENGINE (TIER 3) ---
    print("📡 Fetching live operational updates from network feeds...")
    
    # We ping Google News specifically for recent fact-checks in India
    rss_url = "https://news.google.com/rss/search?q=fact+check+india&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    
    new_sources = []
    new_facts = []
    
    # Extract the top 100 most recent verified articles from the live web
    for entry in feed.entries[:100]:
        # Clean up the source name if available, otherwise default to Live RSS
        source_name = entry.source.title if 'source' in entry else "Live News Network"
        new_sources.append(source_name)
        new_facts.append(entry.title)
    
    # Convert the live scraped data into a Pandas DataFrame
    df_new = pd.DataFrame({
        "source": new_sources,
        "fact": new_facts
    })
    
    print(f"✅ Successfully pulled {len(df_new)} live facts from the web.")
    
    # --- STEP 3: MERGING & DEDUPLICATION ---
    print("🔄 Merging datasets and executing deduplication algorithms...")
    combined_df = pd.concat([kb_df, df_new], ignore_index=True)
    
    # Drop exact semantic text duplicates to prevent vector bloating
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['fact'])
    final_count = len(combined_df)
    
    print(f"⚡ Deduplication complete. Removed {initial_count - final_count} duplicate rows.")
    
    # --- STEP 4: EXPORT TO MAIN STORAGE ---
    combined_df.to_csv(csv_file, index=False)
    print(f"🎉 Success! Total verified operational facts tracking: {final_count}")

if __name__ == "__main__":
    run_pipeline()