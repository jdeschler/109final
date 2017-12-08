---
title: Baseball Win Probability, CS109a Final Project for Group 34
---

#### Problem Statement and Motivation
We were tasked with building an **MLB win probability model**, using historical play-by-play data to predict the outcome of a given game.  While win probability, more specifically **win probability added (WPA)**, is a popular metric to evaluate players’ performances and managers’ decisions, the actual methodology used to calculate WPA is often either overlooked or entirely ignored.  With our project, we hope to bridge this gap, fitting a transparent model that offers accurate predictions.

#### Introduction and Description of Data
The data used in our project was obtained from Baseball-Reference, a popular site that houses MLB game data going back to 1876.  For the purposes of this assignment, we required detailed **play-by-play data**, so we scraped 6 seasons’ worth of games between **2012 and 2017**, including all regular season and postseason contests.  

For every event, this data included included a number of features, including:

* the inning
* score
* base-out state
* pitcher
* batter

Our **exploratory data analysis** revealed an intuitive but exciting trend: there appeared to be a strong relationship between score differential and inning, as leads became better predictors of victory later in games.  While perhaps obvious, since opponents have fewer opportunities to mount a comeback as the game progresses, this revelation was very important for the development of our project, as it strongly suggested a need for **higher-order interaction terms**.  We also decided to include **teams’ records as predictors** in our model after our exploratory analysis, as by definition, teams with a stronger track record should be expected to win more frequently.

#### Literature Review/Related Work
In <a href='https://www.baseball-reference.com/boxes/ARI/ARI201704020.shtml'> Baseball-Reference</a>’s play-by-play data, the site provides **their own predicted win probabilities at the start of each event**, along with the change in win probability since last event, although no information is given surrounding how these values were produced.  

When considering different potential models for our project, we found <a href='http://homepage.divms.uiowa.edu/~dzimmer/sports-statistics/nettletonandlock.pdf'>Lock</a>’s paper (cited in the Sports project description PDF) to be very helpful.  Although his research focused on NFL data, it highlighted the power of the random forest model and identified significant predictors, inspiring us to pursue a similar approach.

Work done by <a href='https://www.fangraphs.com/library/misc/wpa/'>Fangraphs</a> in explaining their calculations of **Win Probability Added and Leverage Index** were also crucial as we developed a strategy for determining the best players and managers as measured by these metrics.

#### Modeling Approach and Project Trajectory
Since a baseball game can result in **only two distinct outcomes** (a win or a loss), our project can be considered a **binary classification problem**.  Thus, we felt that a **logistic regression model** could serve as an appropriate baseline.  Before fitting our model, we carried out the following procedures to engineer the data:

* Created dummy variables: one for every batter and pitcher in the dataset.  
* One-hot encoded innings:  although innings do happen in sequential order, they <a href='https://jdeschler.github.io/109final/EDA.html#inning-x-score-differential'>may not be linearly related</a> to win probability as their numeric labels suggest, and we wanted to allow our model the flexibility to account for these relationships. 
* Created third-order interaction terms: these could represent the relationships between inning, base-out state, and score differential, among other things.  

After including these features, we fit our logistic model on the dataset. 

This baseline model performed well, registering with a classification accuracy score of 0.705.

After fitting this logistic regression baseline model, we hoped to improve our performance by fitting more **complex ensemble models**, specifically **Random Forests** and **AdaBoost** classifiers.  While noted for their **superior accuracy**, these models were much more difficult from a **computational perspective**, as they require hyperparameter tuning to determine the optimal maximum decision tree depth in each.  

To obtain these values, we performed cross-validation on a smaller sample of our data, using 100,000 data points rather in place of the nearly 1.5 million contained in the entire dataset because of time constraints.  After completing this cross-validation process, we then fit these ensemble models on the entire dataset.  Each showed small improvements, which were encouraging, although they did not exhibit the significant increase that we had hoped for.  Our final random forest model recorded a classification accuracy score of 0.718, and the AdaBoost model posted a classification accuracy score of 0.721.

