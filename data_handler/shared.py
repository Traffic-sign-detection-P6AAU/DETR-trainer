import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_json(data, name):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(data, f)
