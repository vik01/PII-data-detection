import json
import pandas as pd

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


# ONE-HOT ENCODE
# https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/
def one_hot(df, col, pre):
  encoded = pd.get_dummies(df[col], prefix=pre)
  for column in encoded:
    encoded = encoded.rename(columns={column: col + "_" + column})
  encoded['Id'] = df['Id']
  return encoded


# Function ----------------------------------------------------------
def parse_data():

    # Open the file and load the json file in a python dictionary
    file = open("raw_data/train6.json")
    documents = json.load(file)[:10]

    # New variable to hold data
    data_dict = {
        "Id": [],
        "Token": [],
        "PII": []
    }

    # Updating identifier
    id = 0

    # For each document.
    for document in documents:

        print(document)

        # Grab the tokens and labels
        doc_token = document["tokens"]
        doc_label = document["labels"]

        break

        # print(type(doc_token))
        # print(type(doc_label))

    #     if doc_token not in data_dict["Token"]:

    #         data_dict["Token"].append(doc_token)
    #         data_dict["Id"].append(id)
            
    #         if doc_label != "O":
    #             data_dict["PII"].append(False)
    #         else:
    #             data_dict["PII"].append(True)

    #         id = id + 1
    
    # all_data = pd.DataFrame(data_dict)

    # train_encoded = one_hot(all_data, "Token", 'is')
    # final_train_x = pd.merge(all_data, train_encoded, on=["Id"])
    # final_train_y = final_train_x["PII"].to_numpy()

    # final_train_x.to_csv("test.csv")

parse_data()



    

