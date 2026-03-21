import os
import json
# print(os.listdir("questions/"))

def convert_to_chunks(item):
    text_tags = ", ".join(item["tags"])
    
    text = f"""
        "Question": {item["question"]},
        "Subject": {item["subject"]},
        "type": {item["type"]},
       
        "tags":{text_tags}
        """
    return {
        "text" : text,
        "metadata" : item
    }
        

for file_name in os.listdir("questions/"):
    file_path = os.path.join("questions/", file_name)   

    with open(file_path , "r") as file:
        data = json.load(file)
        chunks = []
        for item in data["chunks"]:
            chunk = convert_to_chunks(item)
            chunk["text"] = chunk["text"].replace("\n", " ").replace('\"', ' ').strip()
            chunks.append(chunk)
            # data["chunks"].append(convert_to_chunks(item))
        # print(data["chunks"][0]["question"])
        
        
        
    os.makedirs("new_questions",exist_ok=True)
    with open(os.path.join("new_questions/", file_name), "w") as new_file:
        json.dump({"chunks": chunks} , new_file, indent=4)
    
        