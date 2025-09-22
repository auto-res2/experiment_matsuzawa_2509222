
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: archived
    dtype: string
  - name: author
    dtype: string
  - name: author_fullname
    dtype: string
  - name: body
    dtype: string
  - name: comment_type
    dtype: string
  - name: controversiality
    dtype: string
  - name: created_utc
    dtype: string
  - name: edited
    dtype: string
  - name: gilded
    dtype: string
  - name: id
    dtype: string
  - name: link_id
    dtype: string
  - name: locked
    dtype: string
  - name: name
    dtype: string
  - name: parent_id
    dtype: string
  - name: permalink
    dtype: string
  - name: retrieved_on
    dtype: string
  - name: score
    dtype: string
  - name: subreddit_id
    dtype: string
  - name: subreddit_name_prefixed
    dtype: string
  - name: subreddit_type
    dtype: string
  - name: total_awards_received
    dtype: string
  splits:
  - name: programming
    num_bytes: 3466623746
    num_examples: 7503347
  - name: tifu
    num_bytes: 4761338653
    num_examples: 12738669
  - name: explainlikeimfive
    num_bytes: 8451732573
    num_examples: 16392814
  - name: WritingPrompts
    num_bytes: 4651591771
    num_examples: 4436210
  - name: changemyview
    num_bytes: 8603031915
    num_examples: 11600073
  - name: LifeProTips
    num_bytes: 5272994396
    num_examples: 12829459
  - name: todayilearned
    num_bytes: 22655655241
    num_examples: 60199778
  - name: science
    num_bytes: 7069809765
    num_examples: 18112884
  - name: askscience
    num_bytes: 3144754665
    num_examples: 6286702
  - name: ifyoulikeblank
    num_bytes: 547200329
    num_examples: 1332211
  - name: Foodforthought
    num_bytes: 308377128
    num_examples: 567900
  - name: IWantToLearn
    num_bytes: 408331672
    num_examples: 745543
  - name: bestof
    num_bytes: 2003718831
    num_examples: 4347522
  - name: IAmA
    num_bytes: 9380094090
    num_examples: 25778822
  - name: socialskills
    num_bytes: 1000014402
    num_examples: 1842733
  - name: relationship_advice
    num_bytes: 22298879735
    num_examples: 38937398
  - name: philosophy
    num_bytes: 1494947876
    num_examples: 2391695
  - name: YouShouldKnow
    num_bytes: 1165617658
    num_examples: 2639265
  - name: history
    num_bytes: 1457852402
    num_examples: 2962043
  - name: books
    num_bytes: 4562689426
    num_examples: 10187495
  - name: Showerthoughts
    num_bytes: 13259109532
    num_examples: 34123213
  - name: personalfinance
    num_bytes: 9484869588
    num_examples: 18361314
  - name: buildapc
    num_bytes: 9801044390
    num_examples: 21761801
  - name: EatCheapAndHealthy
    num_bytes: 853462012
    num_examples: 1821897
  - name: boardgames
    num_bytes: 3131627378
    num_examples: 6328926
  - name: malefashionadvice
    num_bytes: 2928017882
    num_examples: 7712258
  - name: femalefashionadvice
    num_bytes: 1619784736
    num_examples: 3262969
  - name: scifi
    num_bytes: 888152056
    num_examples: 2193741
  - name: Fantasy
    num_bytes: 2285934538
    num_examples: 4566639
  - name: Games
    num_bytes: 10396813188
    num_examples: 23373965
  - name: bodyweightfitness
    num_bytes: 794549854
    num_examples: 1613634
  - name: SkincareAddiction
    num_bytes: 3421122597
    num_examples: 5660550
  - name: podcasts
    num_bytes: 464773126
    num_examples: 943266
  - name: suggestmeabook
    num_bytes: 1842944304
    num_examples: 3492937
  - name: AskHistorians
    num_bytes: 2244587909
    num_examples: 2714353
  - name: gaming
    num_bytes: 28374513722
    num_examples: 85729253
  - name: DIY
    num_bytes: 2113533684
    num_examples: 4489265
  - name: sports
    num_bytes: 2230129132
    num_examples: 6470079
  - name: space
    num_bytes: 3081499208
    num_examples: 7896182
  - name: gadgets
    num_bytes: 1683252868
    num_examples: 4104833
  - name: Documentaries
    num_bytes: 1852644771
    num_examples: 4051474
  - name: GetMotivated
    num_bytes: 1211761267
    num_examples: 3221980
  - name: UpliftingNews
    num_bytes: 2003149025
    num_examples: 4741948
  - name: technology
    num_bytes: 10826871436
    num_examples: 25404699
  - name: Fitness
    num_bytes: 6191132755
    num_examples: 14319856
  - name: travel
    num_bytes: 1740556350
    num_examples: 3806755
  - name: lifehacks
    num_bytes: 626791812
    num_examples: 1799437
  - name: Damnthatsinteresting
    num_bytes: 6376694618
    num_examples: 15643554
  - name: gardening
    num_bytes: 1825313940
    num_examples: 4568468
  - name: mildlyinteresting
    num_bytes: 9079894206
    num_examples: 26436769
  download_size: 109177016105
  dataset_size: 255339788158
annotations_creators:
- no-annotation
language:
- en
language_creators:
- found
license: []
multilinguality:
- monolingual
pretty_name: Reddit comments
size_categories:
- 10B<n<100B
source_datasets: []
tags:
- reddit
- social-media
task_categories:
- text-generation
task_ids:
- dialogue-modeling
- language-modeling
---

# Dataset Card for "REDDIT_comments"
## Dataset Description

- **Homepage:**
- **Paper: https://arxiv.org/abs/2001.08435**

### Dataset Summary
Comments of 50 high-quality subreddits, extracted from the REDDIT PushShift data dumps (from 2006 to Jan 2023).

### Supported Tasks
These comments can be used for text generation and language modeling, as well as dialogue modeling.

## Dataset Structure
### Data Splits
Each split corresponds to a specific subreddit in the following list: "tifu", "explainlikeimfive", "WritingPrompts", "changemyview", "LifeProTips", "todayilearned", "science", "askscience", "ifyoulikeblank", "Foodforthought", "IWantToLearn", "bestof", "IAmA", "socialskills", "relationship_advice", "philosophy", "YouShouldKnow", "history", "books", "Showerthoughts", "personalfinance", "buildapc", "EatCheapAndHealthy", "boardgames", "malefashionadvice", "femalefashionadvice", "scifi", "Fantasy", "Games", "bodyweightfitness", "SkincareAddiction", "podcasts", "suggestmeabook", "AskHistorians", "gaming", "DIY", "mildlyinteresting", "sports", "space", "gadgets", "Documentaries", "GetMotivated", "UpliftingNews", "technology", "Fitness", "travel", "lifehacks", "Damnthatsinteresting", "gardening", "programming"

## Dataset Creation

### Curation Rationale
All the information fields have been cast to string, as their format change through time from one dump to the following. A reduced number of keys have been kept: "archived", "author", "author_fullname", "body", "comment_type", "controversiality", "created_utc", "edited", "gilded", "id", "link_id", "locked", "name", "parent_id", "permalink", "retrieved_on", "score", "subreddit", "subreddit_id", "subreddit_name_prefixed", "subreddit_type", "total_awards_received".

### Source Data
The [Reddit PushShift data dumps](https://files.pushshift.io/reddit/) are part of a data collection effort which crawls Reddit at regular intervals, to extract and keep all its data.

#### Initial Data Collection and Normalization
See the paper.

#### Who are the source language producers?
Redditors are mostly young (65% below 30), male (70%), and American (50% of the site).

### Personal and Sensitive Information
The data contains Redditor's usernames associated to their content.

## Considerations for Using the Data

This dataset should be anonymized before any processing.
Though the subreddits selected are considered as being of higher quality, they can still reflect what you can find on the internet in terms of expressions of biases and toxicity.

### Contributions

Thanks to [@clefourrier](https://github.com/clefourrier) for adding this dataset.
Output:
{
    "extracted_code": ""
}
