"""
Political news filter and section EDA


Source for: 
- reports/nonpolisects_2021_scmp_valcnts.csv


Finding: 
- political_news_filter passes the eye test: 99% of sports articles get filtered out;
89% of lifestyle articles are filtered out too. 
85.6% of news articles are retained as "political".
"""
import polinews
import pandas as pd
import time 


def timeit(fn, *args, **kwargs):
    s = time.perf_counter()
    ret = fn(*args, **kwargs)
    e = time.perf_counter()
    print(f"{fn.__name__} took {e-s} secs")
    return ret

text_col = "Body"
index_col = "Index"
# %%
classifier = polinews.Classifier()

# %%
df = pd.read_csv(r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\2021.csv")

arr = df[text_col].astype(str).values
idx = df[index_col].values

estimations = timeit(classifier.estimate, arr)

df2 = pd.DataFrame({
    "Index":idx,
    "poliestimation":estimations,
})
merged = df2.merge(df, on="Index", how="inner")

poli =  merged.loc[lambda d: d.poliestimation >= 0.5]
nonpoli =  merged.loc[lambda d: d.poliestimation < 0.5]

year = "2021"
path =  fr"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\sections/sect_{year}.csv"
sections = pd.read_csv(path)

polisects = sections.merge(poli, on="Index", how="inner").section0.value_counts()
# %%
nonpolisects = sections.merge(nonpoli, on="Index", how="inner").section0.value_counts()
# %%
# reports to highlight: 
# lifestyle filtering
def get_perc(a, b):
    return round(a/ (a+b),3) *100
print(get_perc(polisects["lifestyle"], nonpolisects["lifestyle"]))
# only 11 percent of lifestyle articles end up in political news
print(get_perc(polisects["sport"], nonpolisects["sport"]))
# only .8 percent of sports articles end up as political news
print(get_perc(polisects["news"], nonpolisects["news"]))
# by contrast 85.6% of news gets classified as political

# %%

polisects.to_csv(r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\code\reports\polisects_2021_scmp_valcnts.csv")
nonpolisects.to_csv(r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\code\reports\nonpolisects_2021_scmp_valcnts.csv")

# %%
year = "2013"
path =  fr"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\sections/sect_{year}.csv"
section11 = pd.read_csv(path)
section11.section0.value_counts(dropna=False)
pd.DataFrame.value_counts()

