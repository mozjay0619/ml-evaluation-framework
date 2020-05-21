from collections import defaultdict


def dict_is_nested(d):
    
    return any(isinstance(i,dict) for i in d.values())

def get_merged_list_from_dict_list_values(d):
    
    s = set()
    
    for k, v in d.items():
        
        s = s.union(v)
    
    return list(s)

