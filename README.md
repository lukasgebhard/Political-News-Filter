# Political News Filter

*Political News Filter* classifies English news articles based on whether they cover policy topics.

It uses a broad characterization of politics: Politics is about "who gets what, when, and how" [(Lasswell, 1936)](https://www.cambridge.org/core/journals/american-political-science-review/article/politics-who-gets-what-when-how-by-harold-d-lasswell-new-york-whittlesey-house-1936-pp-ix-264/90C407BEDE6963B3D2C84FF79C695E1E). As a result, *Political News Filter* may consider business news or tech news as political, depending on actual contents.

## Requirements

- Python 3.6+
- Pandas 0.24.1+
- NumPy 1.18.1+
- Keras 2.3.1+
- TensorFlow 2.1.0+

*Political News Filter* supports both CPU and GPU processing. The latter is faster but requires a CUDA-capable graphics card and the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

## Setup

1. Clone this repository:

    ```bash
    $ git clone https://github.com/lukasgebhard/Political-News-Filter.git
    $ cd Political-News-Filter
    ```

1. Download and extract [pon_classifier.zip](https://drive.google.com/open?id=1kmFr3WYOa7bSQELvpMcY77wH4gzLe9cJ) into the repository folder. Its inflated size is 1.2 GB.

1. Install Python dependencies. For example, create a virtual environment:

    ```bash
    $ virtualenv --python=python3.6 venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    ```

1. Verify the installation was successful:

    ```bash
    $ ./check_installation.sh
    Hooray! Political News Filter is properly installed and ready to use.
    ```

## Usage Demo

Start a Python session:

```bash
$ python3
```

Create exemplary articles:

```python
>>> political_article = '''White House declares war against terror. The US government officially announced a ''' \
                        '''large-scale military offensive against terrorism. Today, the Senate agreed to spend an ''' \
                        '''additional 300 billion dollars on the advancement of combat drones to be used against ''' \
                        '''global terrorism. Opposition members sharply criticize the government. ''' \
                        '''"War leads to fear and suffering. ''' \
                        '''Fear and suffering is the ideal breeding ground for terrorism. So talking about a ''' \
                        '''war against terror is cynical. It's actually a war supporting terror."'''
>>> nonpolitical_article = '''Table tennis world cup 2025 takes place in South Korea. ''' \
                           '''The 2025 world cup in table tennis will be hosted by South Korea, ''' \
                           '''the Table Tennis World Commitee announced yesterday. ''' \
                           '''Three-time world champion, Hu Ho Han, did not pass the qualification round, ''' \
                           '''to the advantage of underdog Bob Bobby who has been playing outstanding matches ''' \
                           '''in the National Table Tennis League this year.'''
```

To filter a list of news articles, call `filter_news`:

```python
>>> from political_news_filter import filter_news
>>> political_article == filter_news([political_article, nonpolitical_article])[0]
True
```

If you need more flexibility, you can directly call the underlying classifier:

```python
>>> from political_news_filter import Classifier
>>> classifier = Classifier()
>>> probabilities = classifier.estimate([political_article, nonpolitical_article])
>>> probabilities[0] > 0.99
True
>>> probabilities[1] < 0.01
True
```

Please read the docstrings for further information.

## Runtime Performance

Below are some benchmarks on a notebook with 6 CPU cores @ 2.6 GHz, a GPU with 4 GB GRAM and CUDA capability 7.5, 32 GB RAM, and a PCIe SSD drive:

Task | On CPU | On GPU
--- | --- | ---
One-time Initialization | 30 sec | 15 sec
Classification of 1,000 articles | 1.8 sec | 1.3 sec

## Architecture

The classifier is based on a [model by Heng Zheng](https://www.kaggle.com/hengzheng/news-category-classifier-val-acc-0-65) submitted to Kaggle under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) license. It is a convolutional neural network with a 100-dimensional [GloVe](https://www.aclweb.org/anthology/D14-1162/) embedding layer, three convolutional layers, each one followed by a ReLu layer and a pooling layer, and finally a softmax output layer. During training, a cross-entropy loss function is minimized using dropout regularization.

## Training & Evaluation

I created a labeled set of 0.57M news articles, selected from:

- [CC-News](https://commoncrawl.org/2016/10/news-dataset-available/) (extracted using [news-please](https://github.com/fhamborg/news-please))
- [The HuffPost dataset](https://www.kaggle.com/rmisra/news-category-dataset)
- [The BBC dataset](http://mlg.ucd.ie/datasets/bbc.html)

After fitting the classifier on 87.5 % of the articles, testing it on the remaining 12.5 % yields:

- F1 = 94.4
- Precision = 95.2
- Recall = 91.8 
