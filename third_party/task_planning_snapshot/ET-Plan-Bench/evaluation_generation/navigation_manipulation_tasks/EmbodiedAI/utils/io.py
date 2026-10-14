import json
import jsonlines


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in jsonlines.Reader(f):
            data.append(line)
    return data


def json_iter(json_file_path):
    with open(json_file_path, 'r') as json_file:
        json_list = list(json_file)

    for json_line in json_list:
        yield json.loads(json_line)