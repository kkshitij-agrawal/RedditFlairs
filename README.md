# RedditFlairs
A system which inputs a link and detects the flair (category) of a Reddit post.

# Dependencies Used
<img src="/Images/lib.JPG" width="400">

# Running the code
use the command **_python ml.py_** to run the python.

# Output received
- You'll get a list of all test data titles that were used to testes on the trained model and in addition to the flair they belong to according to the trained model. 
- The accuracy of the trained model is also printed.
- Two arrays containing the encoded flair values are also printed. 1st array gives the encoded value of the flair for the post in test data based on the number of comments. 2nd array gives the encoded value of the flair for the post in test data based on the score.

# Code Explanation

First, I used the **Reddit API** to get access to the reddit posts
<img src="/Images/reddit.JPG" width="400">

I created the variables to store the data. Test and Train variable were separate, I stored the following things- 
**Title, comment, score(Upvotes) and the flair**.

For the**Train dataset**, I took manually created the database to get a variety of data with each having enough number of test cases.
I first searched the subreddit with flair name using the command: _'subreddit.search('flair_name:"Photography"')'_
I stored the data in a database and added 100 posts of each category. Thus, a total of 1199 train data set.

For the **Test dataset**, I took the top 100 NEW posts from the subreddit and used them as the test data set. There is surely no overlapping as the train data set was created a few days ago from submission data and test data is obviously the most recent with 100th post being created only at max 10 hours ago.

But why did I not use 'train_test_split'? 
The data I was taking from reddit didn't have enough number of posts from each category and then dividing it into train-test wouldn't ensure variety of test cases.

<img src="/Images/data.JPG" width="400">

After having collected the data I used **MultinomialNB** to train my data set as first I decided to use the TITLE to determine the flairs. I counted the number of words in each title and calcualed their weightage based upon their frequency (more freq, less weightage). The following commands (image) were used to fit(train) with the dataset and then predict(test). In addition I also printed the Accuracy of the model using '_accuracy_score_' command.

<img src="/Images/ml.JPG" width="400">

Before I tested with other collected data suchas comments and score visually represented the **graphs using matplotlib**.
The first graph shows the graph between the assigned values of each flair in x-axis and upvotes. (A better method for the graph provided below)
The second graph shows the graph between each flair and their upvotes(red) and comments(blue).
<img src="/Images/graph.JPG" width="400">
<img src="/Images/g2.JPG" width="400"><img src="/Images/g1.JPG" width="800">

Now, I used other features (comments and score) of the reddit post to determine the flair. I first encoded the flair names into dummy variables using 'LabelEncoder' and then created two **LogisticRegression** models trained them according to each feature mentioned before respectively.

<img src="/Images/train2.JPG" width="400">

Finally, I used while loop to continuously take the reddit post **link input from user** and find the features of the post. And then using the title, used the already trained model to predict the flair of the post.

<img src="/Images/input.JPG" width="400">

Note:
I have deployed the app on Heroku but I keep an error in the logs _'at=error code=h10 status=503'_ which i haven't been able to debug yet. In addition, the user input code is yet to be written in flask so that it works with the web-app.

# Resources

USING THE REDDIT API FOR DATA
http://www.storybench.org/how-to-scrape-reddit-with-python/

LEARNING ML AND PANDAS
https://www.youtube.com/channel/UCh9nVJoWXmFb7sLApWGcLPQ

HEROKU WITH PYTHON
https://devcenter.heroku.com/articles/getting-started-with-python
