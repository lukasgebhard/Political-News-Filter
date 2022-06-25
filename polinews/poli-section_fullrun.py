"""
Full run on scmp to generate mask for whether news is political or not. 

Run using newssent environment. 

Warning: contains old file locations: you'll need to change to uniform 
"data/publication/publication_full.csv" format for file_loc
"""
from logging import root
import polinews
import pandas as pd
import time 
import os

def timeit(fn, *args, **kwargs):
    s = time.perf_counter()
    ret = fn(*args, **kwargs)
    e = time.perf_counter()
    print(f"{fn.__name__} took {e-s} secs")
    return ret


# %%
classifier = polinews.Classifier()

# %%

def generate_poli_mask(file_loc, save_loc, text_col, index_col):
    df = pd.read_csv(file_loc)

    arr = df[text_col].astype(str).values
    idx = df[index_col].values

    estimations = timeit(classifier.estimate, arr)

    df2 = pd.DataFrame({
        index_col:idx,
        "poliestimation":estimations,
    })
    merged = df2.merge(df, on=index_col, how="inner")
    merged[[index_col, "poliestimation"]].to_csv(save_loc)

    return merged
# %%
# scmp mask
text_col = "Body"
index_col = "Index"
for year in range(2011, 2021):
    file_loc = rf"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\{year}.csv"
    save_loc = fr"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\polimask\pmask_{year}.csv"
    generate_poli_mask(file_loc, save_loc, text_col, index_col)

# %% 
# NYT
# DON'T RUN YET
file_loc = r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\nyt\nyt_full.csv"
save_loc = r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\nyt\polimask\pmask_.csv"
text_col = "text"
index_col = "sourceurl"
generate_poli_mask(file_loc, save_loc, text_col, index_col)
import pandas as pd
df = pd.read_csv(save_loc)
# %% 
# HKFP
file_loc = r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\hkfp\allscrape_20211129_020006.csv"
save_loc = r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\hkfp\polimask\pmask_.csv"
text_col = "Body"
index_col = "Art_id"
generate_poli_mask(file_loc, save_loc, text_col, index_col)
# %%
# China daily
text_col = "plainText"
index_col = "id"
file_loc = rf"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\chinadaily\chinaddaily_full.csv"
save_loc = fr"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\chinadaily\polimask\pmask_.csv"
generate_poli_mask(file_loc, save_loc, text_col, index_col)
# %% 
#  Global times
file_loc = r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\globaltimes\ArticleItem\20211228_154428.csv"
save_loc = r"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\globaltimes\polimask\pmask_.csv"
text_col = "Body"
index_col = "Art_id"
generate_poli_mask(file_loc, save_loc, text_col, index_col)

# %% 

# %%
# SCMP polifilter consolidation
# dfls = []
# for year in range(2011, 2022):
#     df = pd.read_csv(fr"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\polimask\pmask_{year}.csv")
#     dfls.append(df)
# maindf = pd.concat(dfls)
# maindf.to_csv(fr"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\polimask\pmask_.csv")

# %% section consolidation
# dfls = []
# for year in range(2011, 2022):
#     df = pd.read_csv(fr"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\sections\sect_{year}.csv")
#     dfls.append(df)
# maindf = pd.concat(dfls)
# maindf.to_csv(fr"C:\Users\tlebr\OneDrive - pku.edu.cn\Thesis\data\scmp\sections\sections.csv")
