import os
import json

def convert_to_chunks(item):
    topic_tags = ", ".join(item["topics"])
    text = f"""
    {item["subject"]} - {item["unit"]} : {item["title"]}  include topics : {topic_tags}
    
       """
    metadata = f"""
    
    """
    return {
        "text":text,
        "metadata": {
            "unit": item["unit"],
            "subject": item["subject"],
            "title" : item["title"],
            "type": "SYLLABUS",
            "topics": item["topics"]

       }
    }

for file_name in os.listdir("syllabus/"):
    file_path = os.path.join("syllabus/", file_name)
    
    with open(file_path, "r") as file:
        data = json.load(file)
        chunks = []
        for item in data["units"]:
            chunk = convert_to_chunks(item)
            chunk["text"] = chunk["text"].replace("\n", " ").replace('\"', ' ').strip()
            chunks.append(chunk)
            

    os.makedirs("new_syllabus",exist_ok=True)
    with open(os.path.join("new_syllabus/", file_name), "w") as new_file:
        
        json.dump({"units": chunks} , new_file, indent=4)              
        