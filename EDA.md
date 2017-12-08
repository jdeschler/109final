---
nav_include: 2
title: EDA
---

## Some notes from our EDA

#### Wins
Initially, we only scraped the number of wins that the team at bat had, and were planning on using that as a predictor for who would win the game. However, when we scattered the number of wins against the percentage of times that team won a game, we saw the following:
![Wins vs Win %](./images/eda_wins.png)
We believe that the extreme values at the far right are due to the fact that very few teams reach that many wins. For example, if one team wins 100 games, then the last game of their season is 101 games, but no other teams reach 100 wins, then this predictor would only have one data point. We believe that the clustering of data arond 50% in the middle area is due to the fact that many teams reach a decent number of wins throughout the season, so there are many games played when teams have 60-80 wins. This means that we would expect around half of those games to be wins and half losses. Thus, we decided that we should include both the batting team's wins _and_ the fielding team's wins when we rescraped our data the EDA. 

#### Inning x Score Differential
As any avid baseball fan will tell you, a 2 run deficit feels much larger in the later innings than in the 1st or 2nd. So we decided to see if our data bore out this conventional wisdom.
![Wins vs Inning x Score Differential][./images/eda_inning.png)
As we can see, the score differential matters more and more as the game goes on. We included all innings after the 9th with the 9th inning, as the situation in the game is the same. For the ninth inning (and onwards), we see a very steep curve of win probability against score differential. This makes sense, as many teams bring their closer (best relief pitcher) into the game when winning in the ninth. These graphs confirmed our suspicions that we should include interaction terms for these variables in our analysis.



