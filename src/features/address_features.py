from rapidfuzz import fuzz

def jaccard(a,b):

    a=set(a)
    b=set(b)

    if len(a|b)==0:
        return 0

    return len(a&b)/len(a|b)

def address_features(row1,row2):

    return {

        "addr_ratio":
            fuzz.ratio(
                row1["clean_address"],
                row2["clean_address"]
            ),

        "addr_token_sort":
            fuzz.token_sort_ratio(
                row1["clean_address"],
                row2["clean_address"]
            ),

        "numeric_jaccard":
            jaccard(
                row1["numbers"],
                row2["numbers"]
            )
    }