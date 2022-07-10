import pandas as pd
from simcse import SimCSE
from tqdm import tqdm

df = pd.read_csv('./datasets/combined.csv', sep=",")

print(f"NROWS: {len(df)}")

model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

print("Build index with meddra embeddings")
model.build_index(df['meddra'].unique().tolist())

print()
print("Search the index for the meddra term closer to the ade")
# Since we didn't train the model we can use all the dataset as a test set
results = model.search(df['ade'].tolist(), top_k=5)

ades = df['ade'].unique().tolist()
real_meddras = df['meddra'].tolist()
predicted_meddras = []
for result_list in results:
    res = []
    for (meddra, _) in result_list:
        res.append(meddra)
    predicted_meddras.append(res)

to_classify = df.shape[0]
correctly_classified = 0
correct_meddra_in_top_5 = 0
wrongly_classified = 0

for (ade, predicted_meddra_options) in tqdm(zip(df['ade'].tolist(), predicted_meddras)):
    if predicted_meddra_options:
        if predicted_meddra_options[0] == ade:
            correctly_classified += 1
        elif ade in predicted_meddra_options:
            correct_meddra_in_top_5 += 1
        else:
            wrongly_classified += 1
    else:
        wrongly_classified += 1

print()
print(f"Examples: {to_classify}")
print(f"Correctly classified: {correctly_classified}")
print(f"Correct meddra is in the top 5: {correct_meddra_in_top_5}")