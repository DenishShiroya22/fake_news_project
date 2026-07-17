import pandas as pd
import os

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
    
    # For your live presentation, you can mock the daily incoming stream
    # or write an RSS parsing function here using standard XML feeds.
    new_data = {
        "source": ["PIB Fact Check India", "AltNews Verified"],
        "fact": [
            "The Ministry of Finance confirmed that no official notifications regarding emergency banking holidays have been distributed.",
            "The Department of Telecommunications issued a statement confirming that cellular networks will remain fully operational nationwide."
        ]
    }
    df_new = pd.DataFrame(new_data)
    
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