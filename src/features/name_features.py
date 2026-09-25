from rapidfuzz import fuzz
import jellyfish

def name_features(name1, name2):

    return {

        "name_ratio":
            fuzz.ratio(name1, name2),

        "name_token_sort":
            fuzz.token_sort_ratio(name1, name2),

        "name_token_set":
            fuzz.token_set_ratio(name1, name2),

        "name_jaro":
            jellyfish.jaro_winkler_similarity(
                name1,
                name2
            )
    }