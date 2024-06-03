def eval(relevants: list, retrieved: list):

    metrics = {"mean_precision": None, "map": None}

    # Mean Precision and MAP Calculation
    mean_precision = 0
    map = 0
    for i, _ in enumerate(retrieved):
        mean_precision += calc_precision(relevants[i], retrieved[i])
        map += calc_average_precision(relevants[i], retrieved[i])
    metrics["mean_precision"] = mean_precision / len(retrieved)
    metrics["map"] = map / len(retrieved)

    return metrics


def eval_by_theme(relevants: list, retrieved: list, themes: list, queries_themes: list):

    metrics = {
        "mean_precision": None,
        "map": None,
        "theme": {theme: None for theme in themes},
    }

    # Mean Precision and MAP Calculation
    mean_precision = {"mean_precision": 0, "theme": {theme: 0 for theme in themes}}
    map = {"map": 0, "theme": {theme: 0 for theme in themes}}
    len_retrieved = {theme: 0 for theme in themes}

    for i, _ in enumerate(retrieved):
        myprecision = calc_precision(relevants[i], retrieved[i])
        mymap = calc_average_precision(relevants[i], retrieved[i])
        mean_precision["mean_precision"] += myprecision
        map["map"] += mymap

        mean_precision["theme"][queries_themes[i]] += myprecision
        map["theme"][queries_themes[i]] += mymap
        len_retrieved[queries_themes[i]] += 1

    metrics["mean_precision"] = mean_precision / len(retrieved)
    metrics["map"] = map / len(retrieved)

    for theme in themes:
        metrics["theme"][theme] = {
            "mean_precision": mean_precision["theme"][theme] / len_retrieved[theme],
            "map": map["theme"][theme] / len_retrieved[theme],
        }

    return metrics


def calc_precision(relevants: list, retrieved: list):
    qtd_relevant_docs = 0
    for doc in retrieved:
        if doc in relevants:
            qtd_relevant_docs += 1

    return qtd_relevant_docs / len(retrieved)


def calc_average_precision(relevants: list, retrieved: list):
    qtd_relevant_docs = 0
    precision = 0
    for k in range(len(retrieved)):
        if retrieved[k] in relevants:
            qtd_relevant_docs += 1
            precision += calc_precision(relevants, retrieved[: k + 1])
    return precision / qtd_relevant_docs if qtd_relevant_docs != 0 else 0
