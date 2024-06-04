import os
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

from evaluation import eval, eval_by_theme
from preprocessing import preprocess
from tfidf import Tfidf


def run_news(
    ks,
    calc_metrics_by_theme,
    corpus_theme_selection: str = None,
    queries_theme_selection: str = None,
    show_vocab_size: bool = False,
    preprocess_technique: str = "stemmer",
):
    tfidf = Tfidf()
    docs = []
    docs_path = []
    excluded_docs = []

    corpus_path = "data/20_newsgroups"
    queries_path = "data/mini_newsgroups"

    if corpus_theme_selection:
        corpus_path += "/" + corpus_theme_selection
    if queries_theme_selection:
        queries_path += "/" + queries_theme_selection

    # Loading Corpus

    pathlist = Path(corpus_path).glob("**/*")
    file_list = [path for path in pathlist if path.is_file()]
    themes = []

    """
    113
    266
    386
    """

    for path in tqdm(file_list, desc="Loading docs"):
        path_in_str = str(path)
        doc = preprocess(
            filename=path_in_str, has_header=True, technique=preprocess_technique
        )
        if doc != "":
            docs.append(doc)
            docs_path.append(path)

            theme = path.parent.name
            if theme not in themes:
                themes.append(theme)
        else:
            excluded_docs.append(path)

    pprint(f"Themes: {themes}")
    pprint(f"Excluded documents: {excluded_docs}")

    # Vectorize Corpus

    tfidf.vectorize(docs, docs_path)
    tfidf.compute_tfidf_matrix()

    if show_vocab_size:
        print("Size of Vocabulary: ", len(tfidf.vocab))

    # Loading Queries

    pathlist = Path(queries_path).glob("**/*")
    file_list = [path for path in pathlist if path.is_file()]

    queries = []
    for path in tqdm(file_list, desc="Loading queries"):
        queries.append(
            {
                "text": preprocess(
                    filename=str(path), has_header=True, technique=preprocess_technique
                ),
                "path": path,
            }
        )

    results = []

    for query in tqdm(queries, desc="Executing search"):

        retrieved_docs = tfidf.search(
            query=query["text"], k=max(ks), is_preprocessed=True
        )
        retrieved_docs_with_theme = []
        for doc in retrieved_docs:
            retrieved_docs_with_theme.append({"id": doc, "theme": doc.parent.name})

        results.append(
            {
                "query": str(query["path"]),
                "query_theme": query["path"].parent.name,
                "retrieved": retrieved_docs_with_theme,
            }
        )

    # Eval Results

    metrics = {}

    for k in tqdm(ks, desc="Evaluating"):
        # print(f"Evaluating for k={k}")
        # Calc MAP, and Precision for @k
        results_to_eval = {"relevants": [], "retrieved": [], "query_theme": []}

        for result in results:
            results_to_eval["relevants"].append([result["query_theme"]])
            results_to_eval["retrieved"].append(
                [doc["theme"] for doc in result["retrieved"][:k]]
            )
            results_to_eval["query_theme"].append(result["query_theme"])

        # pprint(results_to_eval)

        if calc_metrics_by_theme:
            # print("Evaluating with themes")
            metrics[f"@{k}"] = eval_by_theme(
                results_to_eval["relevants"],
                results_to_eval["retrieved"],
                themes,
                results_to_eval["query_theme"],
            )
        else:
            # print("Evaluating without themes")
            metrics[f"@{k}"] = eval(
                results_to_eval["relevants"], results_to_eval["retrieved"]
            )

    return metrics


ks = [10, 20, 50, 100]
corpus_theme_selection = "sci.med"
queries_theme_selection = "alt.atheism"

pprint(
    run_news(
        ks=ks,
        calc_metrics_by_theme=True,
        show_vocab_size=True,
        preprocess_technique="wordpiece",
    )
)
