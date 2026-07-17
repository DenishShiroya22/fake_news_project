import pandas as pd

print("Extracting verified facts from Indian News dataset...")
# Load the raw dataset you downloaded earlier
df = pd.read_excel('dataset/indian_news.xlsx')

# Keep ONLY the rows where the label is TRUE (1)
df_true = df[df['Label'].astype(str).str.strip().str.lower() == 'true']

# Extract just the fact and the source
kb_df = df_true[['Fact_Check_Source', 'Eng_Trans_Statement']].dropna()
kb_df = kb_df.rename(columns={'Fact_Check_Source': 'source', 'Eng_Trans_Statement': 'fact'})

# Save it as a clean Knowledge Base CSV
kb_df.to_csv('massive_knowledge_base.csv', index=False)
print(f"Success! Extracted {len(kb_df)} verified facts into massive_knowledge_base.csv")