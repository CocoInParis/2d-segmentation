import json
from pathlib import Path

from keras.engine import Layer


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        print(str(obj))
        if isinstance(obj, Path):
            print(obj)
            return str(obj)
        if type(obj) == type:
            if isinstance(obj(), Layer):
                obj = str(type(obj()))
                obj = "layers." + obj.split('.')[-1].split("'")[0]
                return obj
        return json.JSONEncoder.default(self, obj)
