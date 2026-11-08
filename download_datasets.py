import os
from datasets import load_dataset

os.makedirs("data/mr", exist_ok=True)
os.makedirs("data/cr", exist_ok=True)
os.makedirs("data/sst", exist_ok=True)

print("1/3 Downloading and preparing MR dataset...")
mr = load_dataset("cornell-movie-review-data/rotten_tomatoes")

train_texts = [x["text"].replace("\n", " ").strip() for x in mr["train"]]
train_labels = [str(x["label"]) for x in mr["train"]]

test_texts = [x["text"].replace("\n", " ").strip() for x in mr["validation"]] + [x["text"].replace("\n", " ").strip() for x in mr["test"]]
test_labels = [str(x["label"]) for x in mr["validation"]] + [str(x["label"]) for x in mr["test"]]

with open("data/mr/text_train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_texts) + "\n")
with open("data/mr/label_train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_labels) + "\n")
with open("data/mr/text_test.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(test_texts) + "\n")
with open("data/mr/label_test.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(test_labels) + "\n")
print(f"MR saved: {len(train_texts)} train, {len(test_texts)} test (Total {len(train_texts) + len(test_texts)})")

print("\n2/3 Downloading and preparing CR dataset...")
cr = load_dataset("SetFit/cr")
cr_lines = []
for split in ["train", "test"]:
    for item in cr[split]:
        txt = item["text"].replace("\n", " ").replace("\t", " ").strip()
        lbl = item["label"]
        if txt:
            cr_lines.append(f"{lbl}\t{txt}")

with open("data/cr/custrev.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(cr_lines) + "\n")
print(f"CR saved: {len(cr_lines)} examples to data/cr/custrev.txt")

print("\n3/3 Downloading and preparing SST dataset...")
sst = load_dataset("SetFit/sst2")

for split_name, file_prefix in [("train", "train"), ("validation", "val"), ("test", "test")]:
    split_texts = [x["text"].replace("\n", " ").strip() for x in sst[split_name]]
    split_labels = [str(x["label"]) for x in sst[split_name]]
    
    with open(f"data/sst/text_{file_prefix}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(split_texts) + "\n")
    with open(f"data/sst/label_{file_prefix}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(split_labels) + "\n")
    print(f"SST {file_prefix} saved: {len(split_texts)} examples")

print("\nAll datasets generated successfully!")

