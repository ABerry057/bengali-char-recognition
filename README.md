# Berry Bengali: Blog Post 1
### Alex Berry, Jason Chan, Hyunjoon Lee
Brown University Data Science Initiative  
DATA 2040: Deep Learning  
February 18th, 2020

### Bengali.AI Handwritten Grapheme Classification Kaggle Competition
This is the first blog post by our team, Berry Bengali, participating in the Bengali AI Kaggle competition, whose link can be found [here](https://www.kaggle.com/c/bengaliai-cv19/overview).

This project involves classifying handwritten characters of the Bengali alphabet, similar to classifying integers in the MNIST data set. In particular, the Bengali alphabet is broken down into three components for each grapheme, or character: 1) the root, 2) the vowel diacritic, 3) the consonant diacritic, where a diacritic is similar to an accent. The goal is the create a classification model that can classify each of these three components of a handwritten grapheme, and the final result is measured using the recall metrics, with double weight given to classification of the root.

Before we begin working on a baseline model, we decided to perform some exploratory data analysis in order to understand the data better. In particular, we are working with several types of data. Firstly, we have a training csv file which lists the correct root, vowel diacritic, and consonant diacritic, for each one of the training graphemes. These graphemes have an id which corresponds to image data, originally saved in parquet form, but we then parce them into png's to be more accessible in building a neural network and visualizing the data.

## Exploratory Data Analysis (EDA) in the Training Set
There are 2 types of training data: labels and images. For each id, we have an image data in **train_image_data.parquet** and label data in **train.csv**. There are total of 200840 samples, where each each image of Bengali consists of 32330 pixels (approx. 137 x 236), and 3 labels: **grapheme_root**, **vowel_diacritic**, and **consonant_diacritic**. We also downloaded **kalpurush** font to make sure the actual bengali graphemes are displayed rather than messy unicode.

```python
train_df = pd.read_csv('train.csv')
train_df.head()
```

![help](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/df_head.png)



### Unique Values
We explored the number of unique grapheme roots, vowel diacritic, and consonant diacritic.

```python
# number of unique values
print(f'Number of unique grapheme roots: {train_df["grapheme_root"].nunique()}')
print(f'Number of unique vowel diacritic: {train_df["vowel_diacritic"].nunique()}')
print(f'Number of unique consonant diacritic: {train_df["consonant_diacritic"].nunique()}')
```
Number of unique grapheme roots: 168  
Number of unique vowel diacritic: 11  
Number of unique consonant diacritic: 7

###  Most/Least Common Characters for each Component
#### 1. Top 10 Grapheme Roots
We looked at the graphemes with top 10 most common roots. Their grapheme root IDs were [72, 64, 13, 107, 23, 96, 113, 147, 133, 115].
```python
def get_n(df, field, n, ascend=False):
  return pd.DataFrame(df[field].value_counts(ascending=ascend))[:n].rename_axis('id').reset_index()

get_n(train_df, "grapheme_root", 10)
```
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/top10_roots.png)
```python
# top 10 most common grapheme roots
top_10_roots = get_n(train_df, 'grapheme_root', 10)
sorted_indices = top_10_roots.sort_values(by="grapheme_root", ascending=False)["id"].tolist()
# would have prefered to use "component" instead of "index", but unicode isn't supported in the plot
sns.barplot(x="id", y="grapheme_root", data=top_10_roots, order = sorted_indices).set_title("10 Most Common Roots by ID")
```
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/top10_roots_bar.png)
The following figure is 5 samples of grapheme for each IDs of 10 most commont roots.

```python
def make_contact_sheet(images, labels, ex_labels, num_samples):
    '''
      Prints a grid of images with labels of type ex_labels.
      Each column corresponds to a different ex_label and there are nrows total.
      Inputs:
        - images: A list of images to sample from
        - labels: A list where the ith element corresponds to the label of image i in images
        - ex_labels: An array specifying the labels we wish to sample
        - num_samples: A nonnegtive integer specifying how many samples we want for each label
    '''
    indices = [[0] * len(ex_labels) for i in range(num_samples)]
    samples = [[0] * len(ex_labels) for i in range(num_samples)]

    for i in range(len(ex_labels)):
        for j in range(num_samples):
            indices[j][i] = np.where(labels == ex_labels[i])[0][j]

    for i in range(len(ex_labels)):
        for j in range(num_samples):
            samples[j][i] = images.iloc[indices[j][i],1:].to_numpy().astype(int).reshape(137,236)

    f, axs = plt.subplots(num_samples, len(ex_labels), sharey = True, figsize = (20, 10))

    for i in range(len(samples)):
        for j in range(len(samples[0])):
            axs[i, j].imshow(samples[i][j])
            axs[i, j].axis("off")

top10_root_labels = [72, 64, 13, 107, 23, 96, 113, 147, 133, 115]
make_contact_sheet(train_images_0, train_label['grapheme_root'], top10_root_labels, 5)
```
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/top_grapheme_roots.png)
#### 2. Bottom 10 Grapheme Roots
Then, we looked at the graphemes with bottom 10 least common roots. Their grapheme root IDs were [63, 0, 12, 1, 130, 45, 158, 102, 33, 73].
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/bottom10_roots_bar.png)
The following figure is 5 samples of grapheme for each IDs of 10 least common roots.
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/bottom_grapheme_roots.png)
#### 3. Top 5 Vowels
After the grapheme roots, we looked at top 5 most common vowels. Their vowel diacritic IDs were [0, 1, 7, 2, 4].
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/top5_vowels_bar.png)
The following figure is 5 samples of grapheme for each IDs of 5 most common vowels.
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/vowels.png)
#### 4. All 7 Consonants
After the grapheme roots, we looked at all 7 consonants. Their vowel diacritic IDs were [0, 2, 5, 4, 1, 6, 3].
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/top5_consonants_bar.png)
The following figure is 5 samples of grapheme for each IDs of 7 consonants.
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/consonant.png)
#### 5. Top 10 Combinations
Lastly, we looked at the graphemes in which are the 10 most common combinations of **grapheme_root**, **vowel_diacritic**, and **consonant_diacritic**.

**Quantifying the Most Common Combinations**

```python
combo_tally = coll.Counter()
for _, row in tqdm(train_df.iterrows()):
  combo = (row['grapheme_root'], row['vowel_diacritic'], row['consonant_diacritic']) #change strings to tuples
  combo_tally[combo] += 1
combo_tally.most_common(10)

output:
[((64, 7, 2), 303),
 ((72, 0, 2), 297),
 ((64, 3, 2), 289),
 ((167, 7, 0), 283),
 ((74, 1, 0), 178),
 ((29, 0, 0), 178),
 ((48, 4, 0), 177),
 ((107, 7, 0), 177),
 ((103, 1, 0), 177),
 ((96, 9, 5), 177)]
```
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/top10_combos.png)
The following figure is a sample for each most common combination of componenets.
![alt text](https://github.com/csjasonchan357/bengali-char-recognition/raw/master/figures/combinations.png)

### Conclusion of Component EDA

In the training set, the most common root graphemes are 72, 64, and 13 and the last common are 63, 0, and 12. The most common vowel diacritics are 0, 1, and 7. The most common consonant diacritics are 0, 2, and 5. The most common combinations of components are mostly just combinations of the most common components, which is expected. The most common combinations are 64-7-2, 72-0-2, and 64-3-2.

## Baseline Model Identification
Before we begin training a neural network, we're first interested in looking at the balance of the data set, and analyzing a baseline model. Here we define a baseline model to be a basic model which chooses the most populous class (the majority class) and makes that classification for every possible data point.

To specify, in this competition, our goal is the break down each grapheme into three components. The final evaluation metric is a weighted average between the recall scores based on the classification for the root, vowel diacritic, and consonant diacritic. In other words, for each grapheme, for each of these three components, a predicition will be made, and the recall score will be measured based on the true labels. From there, a weighted average is taken where the root recall score has two times the weight of the vowel and consonant diacritic scores, resulting in one final metric.

```python
most_common_root_freq = train_df['grapheme_root'].value_counts(ascending=False).max()
most_common_root = train_df['grapheme_root'].value_counts(ascending=False).idxmax()
most_common_vowel_freq = train_df['vowel_diacritic'].value_counts(ascending=False).max()
most_common_vowel = train_df['vowel_diacritic'].value_counts(ascending=False).idxmax()
most_common_consonant_freq = train_df['consonant_diacritic'].value_counts(ascending=False).max()
most_common_consonant = train_df['consonant_diacritic'].value_counts(ascending=False).idxmax()
```

With the above code, we are able to find the majority class and balance of the entire data set for each of the three grapheme components. The id of the majority class, an example of a grapheme that contains that component, and the balance of that majority class is shown in the table below.

|Most Common Root|Most Common Vowel Diacritic | Most Common Consonant Diacritic
|---|---|---|
|id = 72|id = 0|id = 0|
|e.g. র্দি| e.g. র্ব্য| e.g. ষী|
|balance = 2.9%| balance = 20.7%| balance = 62.4%|

## Baseline Model Analysis

Therefore, the baseline model would predict root = 72, vowel diacritic = 0, and consonant = 0, for every data point. Just by observing the balance of the classes, we can see that this baseline model is unlikely to provide good predictions at all. In fact, due to the extra weight placed on the root classification, and the low balance for both root and vowel diacritic, and the nature that each of these target labels are multi-class labels, our baseline model will most certainly produce only a mediocre result. This is good, as it gives us plenty of space to improve our model and make use of our convolutional neural nets.  

Following the Kaggle competition's official instructions on evaluation, we want to see the performance of our baseline model if it were to provide a single majority class classification for each of the three grapheme components. To do this, we wrote the following code below, which calculates the recall score for each components, given that we predict the majority class for each component, and determine the weighted average final score

```python
y_true_root = list(train_df['grapheme_root'])
y_pred_root = np.full(shape=train_df.shape[0], fill_value=most_common_root)
root_score = sklearn.metrics.recall_score(y_true_root, y_pred_root, average='macro')

y_true_vowel = list(train_df['vowel_diacritic'])
y_pred_vowel = np.full(shape=train_df.shape[0], fill_value=most_common_vowel)
vowel_score = sklearn.metrics.recall_score(y_true_vowel, y_pred_vowel, average='macro')

y_true_consonant = list(train_df['consonant_diacritic'])
y_pred_consonant = np.full(shape=train_df.shape[0], fill_value=most_common_consonant)
consonant_score = sklearn.metrics.recall_score(y_true_consonant, y_pred_consonant, average='macro')

final_score = np.average([root_score, vowel_score, consonant_score], weights = [2,1,1])
```

In the table below, we see the recall scores for each of the components, as well as the final weighted average score.

|Root Recall |Vowel Diacritic Recall | Consonant Diacritic Recall | Weighted Average Recall|
|---|---|---|---|
|0.01 |0.09 | 0.14| 0.06|

As we can see, each individual recall score is very low, and our final weighted average recall score is abysmal. Again, this is largely due to the fact that each of these component classifications are multi-class labels, and in general there are much less overlap in combinations of characters, given there are so many possibilities, and we are looking at our data in the context of a real life language, used to convey the entirety of humanity in one particular language. Thus, this baseline analysis does not provide a strong baseline comparison, due to the fact that even a simple neural network model would be able to outperform this baseline.


## Next Steps
In fact, creating a simple neural network model and iterating it is in the next steps in our competition project process. First we will preprocess the image data further. This includes de-noising it, adding a threshhold filter to it (to turn it from greyscale to black and white, essentially define the outlines of the written parts more clearly), contour it (make a bounding box and use only parts of the image that contain important information, essentially get rid of white space), and rescale it. From there, we will build a model with both convolutional layers, in order to extract feature information about the images, and then split the model into three dense layers which serve as producing the output classifications for the three components. We are looking forward to implementing all of this!
