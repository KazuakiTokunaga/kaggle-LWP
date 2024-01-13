First of all, I would like to express my gratitude to the staff and the Kagglers who shared great information and knowledge. My result is not so good one, but I am posting this writeup in the hope that it will be helpful to someone else and as a record for myself.

## Overview

I created my own LightGBM model and blended it with a Public Notebook model. The best score was achieved with a submission that ensemble the Public Notebook with 0.7 weight and my LightGBM model with 0.3 weight. My model alone had scores of Public 0.5828 / Private 0.5718, which is almost the same as the public single model. But it seems that my model have added a bit of diversity to the ensemble, which leads to the improved score.

| model | CV | Public Score | Private Score | sub |
| --- | --- | --- | --- | --- |
| My model 1 | 0.6001 | 0.582878 | 0.571849 | [x] |
| My model 2 | 0.6018 | 0.584491 | 0.572845 | [] |
| My model 2 * 0.5 + [Public1](https://www.kaggle.com/code/cody11null/lgbm-x2-nn) * 0.5 | - | 0.580021 | 0.568756 | [x] |
| My model 1 * 0.3 + [Public2](https://www.kaggle.com/code/kononenko/lgbm-x2-nn-fusion) * 0.7 | - | 0.578381 | 0.568015 | [x] |

My model 1 and 2 are almost same models with a little different features.

## My Model

I built my LightGBM model as follows. First, I used Stratified Kfold with 5 folds. In each fold, I trained the LightGBM model three times with different seeds. I repeated this process five times, changing the seed for Stratified Kfold. In other words, I trained LightGBM models with 5 folds * 3 model seeds * 5 fold seeds. This was made to  minimize the impact of the random seed as much as possible. For inference, I used the average of these 75 models as the predicted score.

With Regard to feature engineering, I adopted [the Silver Bullet Notebook](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features) as the baseline and made several changes. The number of features amounts to 192.

- Deleted features that were almost identical in content.
- Corrected anomalies in the data that appeared to be bugs. This allowed me to correct data for several IDs that had logs significantly over 30 minutes.
  - Logs where the down_time was reversed by more than one second were corrected so there was no reversal.
  - Logs where the difference in down_time was more than 20 minutes were corrected to a 10-second difference.
  - Logs where more than 20 minutes had passed at the first log were corrected to a 2-minute lapse.
  - Delete some events which are down_event==Process and have too long action_time.
- Added and modified some features about the typing process.
  - Created another P-bursts feature, using bursts of consecutive inputs within 0.3 seconds.
  - The number of characters and the maximum cursor position at the 20 and 25-minute mark.
  - The number of keyboard shortcut.
  - Some gap features of word_count with event_id shift 200.
- Added many features about the essay.
  - Reconstructed the essay at the 15-minute mark and compared it to the final essay, editing distance in the first 100 characters.
  - Add aggreated features of words and sentences.
  - The number of sentences with the first word being over six characters and containing a comma.
  - The proportion of sentences with a one-letter first word.
  - The number of sentences ending with a comma within the first three or four words.
  - The average number of characters in sentences up to the first eight words.
  - The number of sentences ending with a question mark.
  - The number of sentences ending with an exclamation mark.
  - The number of long sentences containing hyphens.
  - The number of long sentences with a comma around the middle.
  - The median total number of characters in 10 consecutive words.
  - The number of combinations of 5 consecutive words where the total number of characters exceeds 35.
  - And so on.

I added and removed features, checking whether it improved CV and LB. However, while CV improved, LB hardly did. Looking at the decimal scores after deadline, it seemed that LB's score was moving in the right direction, but the improvement was quite small. The hyperparameters for LightGBM were decided by optimizing the basic parameters with optuna. However, I didn't spend much time on it, as it seemed to be overfitting to the training data.

## Some Reflections
It was good to know that the silver medal score was achieved with blending of public Notebook and my model that has the almost same score as public single model itself. However, looking at the top solutions, it seemed difficult to achieve a gold medal without using natural language processing techniques. Since I was using only LightGBM with manually created features, it was difficult to achieve a gold medal with this approach. I didn't come up with those approaches, so I would like to learn them from solutions. 

My code can be found in [this github repository](https://github.com/KazuakiTokunaga/kaggle-LWP).