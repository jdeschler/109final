---
nav_include: 1
title: Data Cleaning
notebook: final_data_cleaning.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import pandas as pd
import numpy as np
import copy
```




```python
data2012 = pd.read_csv('Data/2012season.csv')
data2013 = pd.read_csv('Data/2013season.csv')
data2014 = pd.read_csv('Data/2014season.csv')
data2015 = pd.read_csv('Data/2015season.csv')
data2016 = pd.read_csv('Data/2016season.csv')
data2017 = pd.read_csv('Data/2017season.csv')
```




```python
data = pd.concat([data2012, data2013, data2014, data2015, data2016, data2017])
data.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Inn</th>
      <th>Score</th>
      <th>Out</th>
      <th>RoB</th>
      <th>Pit(cnt)</th>
      <th>R/O</th>
      <th>@Bat</th>
      <th>Batter</th>
      <th>Pitcher</th>
      <th>wWPA</th>
      <th>wWE</th>
      <th>Play Description</th>
      <th>play_number</th>
      <th>ytd_wins</th>
      <th>ytd_win_opp</th>
      <th>win</th>
      <th>game_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>t1</td>
      <td>0-0</td>
      <td>0</td>
      <td>---</td>
      <td>2,(0-1) CX</td>
      <td>O</td>
      <td>SEA</td>
      <td>Chone Figgins</td>
      <td>Brandon McCarthy</td>
      <td>-2%</td>
      <td>48%</td>
      <td>Groundout: SS-1B (Weak SS)</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>OAK201203280</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>t1</td>
      <td>0-0</td>
      <td>1</td>
      <td>---</td>
      <td>2,(1-0) BX</td>
      <td>O</td>
      <td>SEA</td>
      <td>Dustin Ackley</td>
      <td>Brandon McCarthy</td>
      <td>-2%</td>
      <td>46%</td>
      <td>Groundout: 2B-1B</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>OAK201203280</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>t1</td>
      <td>0-0</td>
      <td>2</td>
      <td>---</td>
      <td>6,(3-2) CBFBBX</td>
      <td>NaN</td>
      <td>SEA</td>
      <td>Ichiro Suzuki</td>
      <td>Brandon McCarthy</td>
      <td>1%</td>
      <td>48%</td>
      <td>Single to P (Ground Ball to Weak SS-2B)</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>OAK201203280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>t1</td>
      <td>0-0</td>
      <td>2</td>
      <td>1--</td>
      <td>5,(2-2) B&gt;F1BF&gt;S</td>
      <td>O</td>
      <td>SEA</td>
      <td>Justin Smoak</td>
      <td>Brandon McCarthy</td>
      <td>-2%</td>
      <td>45%</td>
      <td>Strikeout Swinging</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>OAK201203280</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>b1</td>
      <td>0-0</td>
      <td>0</td>
      <td>---</td>
      <td>6,(2-2) BCCFBX</td>
      <td>NaN</td>
      <td>OAK</td>
      <td>Jemile Weeks</td>
      <td>Felix Hernandez</td>
      <td>-4%</td>
      <td>42%</td>
      <td>Single to CF (Line Drive to Short CF)</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>OAK201203280</td>
    </tr>
  </tbody>
</table>
</div>





```python
data2 = copy.deepcopy(data)
def score_diff(s):
    a,b = s.split('-')
    return int(a) - int(b)

data2 = data2.dropna(axis = 0, subset = ['wWPA', 'wWE'])

data2['Ro1'] = data2.apply(lambda row: 0 if row['RoB'][0] is '-' else 1, axis = 1)
data2['Ro2'] = data2.apply(lambda row: 0 if row['RoB'][1] is '-' else 1, axis = 1)
data2['Ro3'] = data2.apply(lambda row: 0 if row['RoB'][2] is '-' else 1, axis = 1)
data2['Home'] = data2.apply(lambda row: 0 if row['Inn'][0] is 't' else 1, axis = 1)
data2['score_diff'] = data2.apply(lambda row: score_diff(row['Score']), axis = 1)
data2['Inn'] = data2.apply(lambda row: int(row['Inn'][1:]), axis = 1)
data2['win'] = data2.apply(lambda row: int(row['win']), axis = 1)
data2['wins_to_date'] = data2.apply(lambda row: int(row['ytd_wins']), axis = 1)
data2['wWPA'] = data2.apply(lambda row: int(row['wWPA'][:-1]), axis = 1)
data2['wWE'] = data2.apply(lambda row: int(row['wWE'][:-1]), axis = 1)
data2['opp_wins_to_date'] = data2.apply(lambda row: int(row['ytd_win_opp']), axis = 1)
data2 = data2.drop(['Unnamed: 0', 'Pit(cnt)', 'RoB', 'Play Description', 'R/O', 'Score', 'ytd_wins', 'ytd_win_opp'], axis = 1)
data2.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Inn</th>
      <th>Out</th>
      <th>@Bat</th>
      <th>Batter</th>
      <th>Pitcher</th>
      <th>wWPA</th>
      <th>wWE</th>
      <th>play_number</th>
      <th>win</th>
      <th>game_id</th>
      <th>Ro1</th>
      <th>Ro2</th>
      <th>Ro3</th>
      <th>Home</th>
      <th>score_diff</th>
      <th>wins_to_date</th>
      <th>opp_wins_to_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>SEA</td>
      <td>Chone Figgins</td>
      <td>Brandon McCarthy</td>
      <td>-2</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>OAK201203280</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>SEA</td>
      <td>Dustin Ackley</td>
      <td>Brandon McCarthy</td>
      <td>-2</td>
      <td>46</td>
      <td>2</td>
      <td>1</td>
      <td>OAK201203280</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>SEA</td>
      <td>Ichiro Suzuki</td>
      <td>Brandon McCarthy</td>
      <td>1</td>
      <td>48</td>
      <td>3</td>
      <td>1</td>
      <td>OAK201203280</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>SEA</td>
      <td>Justin Smoak</td>
      <td>Brandon McCarthy</td>
      <td>-2</td>
      <td>45</td>
      <td>4</td>
      <td>1</td>
      <td>OAK201203280</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>OAK</td>
      <td>Jemile Weeks</td>
      <td>Felix Hernandez</td>
      <td>-4</td>
      <td>42</td>
      <td>5</td>
      <td>0</td>
      <td>OAK201203280</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```python
data2.to_csv("cleaned.csv")
```


#### Data should be cleaned, but not one-hot encoded yet

