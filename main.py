# from pathlib import Path
from preprocessing import preprocess
from tfidf import tfidf
from pprint import pprint

# pathlist = Path("data/20_newsgroups").glob('**/*')
# for path in pathlist:
#     path_in_str = str(path)
#     print(path_in_str)

path_in_str = "data/20_newsgroups/rec.sport.baseball/104501"
data = preprocess(path_in_str)
print(data)

pprint(tfidf([data]))
