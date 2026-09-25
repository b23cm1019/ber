def structural_features(r1,r2):

    return {

        "country_match":
            int(
                bool(r1["country"])
                and r1["country"] ==
                r2["country"]
            ),

        "legal_match":
            int(
                bool(r1["legal_form"])
                and r1["legal_form"] == r2["legal_form"]
            )
    }
