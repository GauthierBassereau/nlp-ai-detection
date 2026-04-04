from datasets import load_dataset

print(load_dataset("Hello-SimpleAI/HC3", revision="refs/convert/parquet", data_dir="all"))
