# RedditFlairs
A system which inputs a link and detects the flair (category) of a Reddit post.

# Dependencies Used
<img src="/Images/lib.JPG" width="400">

# Code Explanation

First, I used the reddit API to get access to the reddit posts
<img src="/Images/reddit.JPG" width="400">

I created the variables to store the data. Test and Train variable were separate, I stored the following things- 
Title, comment, score(Upvotes) and the flair.

For the Train dataset, I took manually created the database to get a variety of data with each having enough number of test cases.
I first searched the subreddit with flair name using the command: 'subreddit.search('flair_name:"Photography"')'
I stored the data in a database and added 100 posts of each category. Thus, a total of 1199 train data set.

For the Test dataset, I took the top 100 NEW posts from the subreddit and used them as the test data set. There is surely no overlapping as the train data set was created a few days ago from submission data and test data is obviously the most recent with 100th post being created only at max 10 hours ago.

But why did I not use 'train_test_split'? The data I was taking from reddit didn't have enough number of posts from each category and then dividing it into train-test wouldn't ensure variety of test cases.
<img src="/Images/data.JPG" width="400">

After having collected the data I used Logistic Expression to train my ML model as first I decided to use the TITLE to determine the flairs. The following commands were used to fit(train) with the dataset and then predict(test). In addition I also printed the Accuracy of the model using 'accuracy_score' command.
<img src="/Images/ml.JPG" width="400">

Before I tested with other collected data suchas comments and score visually represented the graphs using matplotlib.
The first graph shows the graph between the assigned values of each flair in x-axis and upvotes. (A better method for the graph provided below)
The second graph shows the graph between each flair and their upvotes(red) and comments(blue).
<img src="/Images/graph.JPG" width="400">
<img src="/Images/g2.JPG" width="400"><img src="/Images/g1.JPG" width="400">


