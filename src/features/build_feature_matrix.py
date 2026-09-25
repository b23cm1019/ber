from .name_features import name_features
from .address_features import address_features
from .structural_features import structural_features

def build_pair_features(r1,r2):

    feats={}

    feats.update(
        name_features(
            r1["core_name"],
            r2["core_name"]
        )
    )

    feats.update(
        address_features(
            r1,
            r2
        )
    )

    feats.update(
        structural_features(
            r1,
            r2
        )
    )

    return feats