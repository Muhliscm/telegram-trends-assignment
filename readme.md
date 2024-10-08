# Data Science Task Telegram Trends

## 1. Methods for features

1. Total Number of characters
2. Total number of sentences
3. Total number of tokens
4. Warning Messages or not
5. Coin names
6. Token names
7. Tagged Accounts
8. User names
9. Restricted user details and status
10. Email Ids
11. Most Frequent words and word frequency
12. Popular Payment methods mentioned
13. Trading platform names Eg: trading view
14. stemming and lemmatization

## 2. Stemming and preprocessing Pipeline

[script link](preprocessing_script.py)

## 3. Descriptions for preprocessing pipeline

1. Take a data frame column
2. Convert to lower case
3. Tokenize the data
4. Check the data is alpha-numeric
5. Remove punctuations
6. Create stems for each token
7. Return the cleaned data column
```
input = df['Messages'] after removing emojis

For single input:

❗️New users are restricted until they ➡️ [CLICK HERE](https://telegram.me/CryptoGroupsBot?start=-1001059287525_1721158735_8c037afbbedabf97f1c7461f2857901e) ⬅️ and pass the captcha or be kicked.

Output:
user restrict click HTTP pass captcha kick

the same process will be applied to all rows

````

## 4. Data Analysis Notebook
[Notebook link](test.ipynb)
