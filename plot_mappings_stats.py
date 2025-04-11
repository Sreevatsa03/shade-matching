import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Clean up and load the latest file again
mapping_path = "mst_to_lab_hsl_mapping.csv"
mapping_df = pd.read_csv(mapping_path)

# Load the original metadata for sample counts
metadata_path = "/Users/sree/mst-e/mst-e_image_details.csv"
metadata = pd.read_csv(metadata_path)
metadata.columns = metadata.columns.str.strip().str.lower()

# Count how many samples per MST level
sample_counts = metadata['mst'].value_counts().sort_index()
mapping_df['sample_count'] = mapping_df['mst_level'].map(sample_counts)

# Plot: Sample count per MST level
plt.figure(figsize=(8, 4))
sns.barplot(x='mst_level', y='sample_count', data=mapping_df, palette='crest')
plt.title("Sample Count per MST Level")
plt.xlabel("MST Level")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

# Plot: LAB Lightness vs MST
plt.figure(figsize=(8, 4))
sns.lineplot(x='mst_level', y='L', data=mapping_df,
             marker='o', label='LAB Lightness')
sns.lineplot(x='mst_level', y='L_hsl', data=mapping_df,
             marker='s', label='HSL Lightness')
plt.title("Lightness vs MST Level (LAB and HSL)")
plt.xlabel("MST Level")
plt.ylabel("Lightness")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: a and b channels vs MST
plt.figure(figsize=(8, 4))
sns.lineplot(x='mst_level', y='a', data=mapping_df,
             marker='o', label='a (green-red)')
sns.lineplot(x='mst_level', y='b', data=mapping_df,
             marker='s', label='b (blue-yellow)')
plt.title("LAB a/b Channels vs MST Level")
plt.xlabel("MST Level")
plt.ylabel("Color Axis Value")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Saturation vs MST
plt.figure(figsize=(8, 4))
sns.lineplot(x='mst_level', y='S', data=mapping_df, marker='o', color='orange')
plt.title("HSL Saturation vs MST Level")
plt.xlabel("MST Level")
plt.ylabel("Saturation")
plt.tight_layout()
plt.show()
