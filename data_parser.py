import json

# Function ----------------------------------------------------------
# Function from: https://www.altcademy.com/blog/how-to-split-a-list-in-python/#:~:text=We%20can%20use%20the%20zip,smaller%2C%20equal%2Dsized%20chunks.&text=In%20this%20example%2C%20we%20use%20the%20zip()%20function%20to,from%20each%20iterator%20into%20tuples.
def split_list(lst, chunk_size):
    chunks = [[] for _ in range((len(lst) + chunk_size - 1) // chunk_size)]
    for i, item in enumerate(lst):
        chunks[i // chunk_size].append(item)
    return chunks


# Function ----------------------------------------------------------
# NO NEED TO RUN NOW, JUST KEEPING CODE
def break_raw_train():
    
    # Open the file and load the json file in a python dictionary
    file = open("raw_data/train.json")
    data = json.load(file)

    # close the file
    file.close()

    # Specify the chunk size
    chunks = 1000
    data_split = split_list(data, chunks)

    # Index for json file saving
    json_file_num = 0

    # Save the data chunks into their own json files
    for data_part in data_split:

        file_path = f"raw_data/train{json_file_num}.json"

        # Open the file path and save the file
        with open(file_path, "w") as out_file:
            json.dump(data_part, out_file, indent=2)
        
        # Update the json file indexer
        json_file_num = json_file_num + 1


# Function ----------------------------------------------------------
def parse_data():
    print("1")


break_raw_train()
