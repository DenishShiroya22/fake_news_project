import pandas as pd
import os

print("1. Processing Indian News Dataset...")
# Read the Excel file. We will use the English translation for consistency with the global models.
df_india = pd.read_excel('dataset/indian_news.xlsx')
df_india = df_india[['Eng_Trans_Statement', 'Label']].dropna()

# Map string 'true'/'false' to our standard 1/0 integers
df_india['Label'] = df_india['Label'].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0})
df_india = df_india.rename(columns={'Eng_Trans_Statement': 'text', 'Label': 'label'})


print("2. Processing Global News 1 (ISOT)...")
# Load both files and manually assign the binary labels
df_isot_true = pd.read_csv('dataset/global_news1/True.csv')
df_isot_true['label'] = 1

df_isot_fake = pd.read_csv('dataset/global_news1/Fake.csv')
df_isot_fake['label'] = 0

# Combine them, and merge the 'title' and 'text' columns so the model gets maximum context
df_isot = pd.concat([df_isot_true, df_isot_fake])
df_isot['text'] = df_isot['title'] + " " + df_isot['text']
df_isot = df_isot[['text', 'label']].dropna()


print("3. Processing Global News 2 (LIAR/Social Media)...")
# TSV means separated by tabs (\t). There is no header, so we assign column numbers based on the readme.
df_liar = pd.read_csv('dataset/global_news2/train.tsv', sep='\t', header=None)

# Column 2 is label, Column 3 is statement
df_liar = df_liar[[1, 2]].rename(columns={1: 'original_label', 2: 'text'})

# Convert the 6-way classification into a strict, clear binary
label_mapping = {
    'true': 1, 
    'mostly-true': 1, 
    # 'half-true' is removed from here
    'false': 0, 
    # 'barely-true' is removed from here
    'pants-fire': 0
}
df_liar['label'] = df_liar['original_label'].map(label_mapping)
df_liar = df_liar[['text', 'label']].dropna()

print("4. Merging into the Master Pipeline...")
# Stack all three sanitized DataFrames vertically
df_master = pd.concat([df_india, df_isot, df_liar], ignore_index=True)
print(f"Total unified rows: {len(df_master)}")

print("5. Deduplication and Cleanup...")
# Drop any rows where text is blank, and drop exact duplicates across datasets
df_master = df_master.dropna(subset=['text', 'label'])
df_master = df_master.drop_duplicates(subset=['text'])
print(f"Total rows after removing duplicates: {len(df_master)}")


print("6. Class Balancing (CRITICAL)...")
# Check the distribution
distribution = df_master['label'].value_counts()
print(f"Current Distribution:\n{distribution}")

fake_count = distribution.get(0, 0)
real_count = distribution.get(1, 0)
min_count = min(fake_count, real_count)

# Under-sample the majority class so we have a perfect 50/50 split
fake_balanced = df_master[df_master['label'] == 0].sample(n=min_count, random_state=42)
real_balanced = df_master[df_master['label'] == 1].sample(n=min_count, random_state=42)

# Re-combine and shuffle (frac=1)
df_final = pd.concat([fake_balanced, real_balanced]).sample(frac=1, random_state=42)
print(f"Final balanced dataset size: {len(df_final)} rows.")

print("7. Exporting to master_training_data.csv...")
df_final.to_csv('master_training_data.csv', index=False)
print("Success! Your highly robust, multi-source dataset is ready.")