# [RealtorBuddy](https://github.com/JOSMANC/RealtorBuddy)
***


## Prototype Project for Data Science Fellowship

The current value of homes in the United States is 27.5 trillion dollars.

Reports indicate that 2015 will be the year that many Millennials will be buying their first homes and the home pricing index has risen every Month for the past seven Months.  

Despite this rebound in the housing market, there still does not exist a tool that would allow for a robust up-to-date assessment of a home price's using all the features that an owner knows about their home.  Many internal features of home are out-of-date in the public domain which can result in poor home evaluations.  

In order to respond to this need in the market I have prototyped a model which has been designed to give the most accurate valuation of a home possible. The approach considered is highly scalable and leaves room for future improvements.  

Supplemental to this work is the creation of a pipeline framework for rapid feature selection, feature representation and model parameterization which can be easily applied to problems outside the housing market.

### [Data Collection](https://github.com/JOSMANC/RealtorBuddy/blob/master/preliminary_work/) 
The focus for the work was a single city spanning two counties but the underlying approach would work for anywhere in the United States. Data sets were collected from 2009 to 2015.  The data characterizing these homes were:

1. Home listings documents consisting of 3 million postings characterizing 1.5 million unique homes. Each Home was characterized by over 300 columns that contained information about the interior features of the home [beds, bath, taxes, size, etc.], dates when events took place [listing date, contract date, etc.] and specifics qualities of the home [Pools, Mountain Views, Horse Stables, etc.].
2. Average test scores for every High School in the two counties.  The address for every high school were identified from listings on U.S. News & World Reports.  Their scores were obtained from local reports spanning from 2010 – 2014.
3. A square grid over the latitude and longitude of the two counties containing the total number of business, and explicit counts of the number of bars, restaurants and coffee shops.  The data was obtained from a latitude and longitude grid search computed with the Yelp API.
4. Census tract and block level information for the two counties were obtained from the American Community Survey 5 Year Data reports from 2011 – 2013.  These include median income, population totals, number of renters, number of people taking public transport and the central points for each census block group.  

### [Defining a Home which Represented the Market](https://github.com/JOSMANC/RealtorBuddy/blob/master/preliminary_work/ExampleDataAccesQuery.sql) 
The production model was designed to compute the fair market price for Single-Family homes.  It is therefore important to define explicitly what homes are truly representative of the market.  A market home was defined predominately as one where:

1. Realtor remarks did not name the home as a short-sale or HUD auction.
2. The home sold within $100,000 of the asking price.
2. It existed on less than an Acre of land.
3. It was a single family house with one home on the lot.
3. No missing features were observed.

#### [Data access](https://github.com/JOSMANC/RealtorBuddy/blob/master/preliminary_work/) 
The data sources were organized and connected primarily through the census tracts and homes latitudes and longitudes.  Values associated with each home were connected on one single pass of the housing dataset:

1. The listing data (originally in the form of a ~10 GB csv file) was transferred into a PostgreSQL database. During the transfer each home's nearest census block and track was appended to the listing data using a K-Nearest-Neighbors fit to the each census block location.  This approach proved faster than querying census API for each address.
2. The High School scores were pickled into a K-Nearest-Neighbors Regressor with the neighbors set to 1 this allowed each home's distance from the school and that schools score to be efficiently computed and added for each home.
3. The commercial business information was associated to the nearest census group block. Additionally, a K-Nearest-Neighbors Regressor was considered to compute the distance weighted scores for the total number of business near each home.  
4. The census tract and block group information used the respective identifiers.

### [Model Train and Validation](test_train_validate_model/)
The majority of the work was spend on accessing the importance of each of the over 600 available features.  In order to find the optimal number and representation of the features the a training/testing  [class](test_train_validate_model/TrainTestValidateSearch.py) was created was created which allowed for direct pipelining and metric testing for several different models and feature representations.  

The model contained functions to quickly select different features to test, grid search parameters and score the model. Cross-validation was utilized to compare across different models and feature frameworks with the average percent error and median error for the total data sets and the middle 50% of home prices [$100,000 - $230,000]. This work was primarily conducted using [iPython Notebooks](test_train_validate_model/ModelTestTemplate.ipynb) for fast EDA. 

The average feature importance over the fold was reported from a random forest to determine where signal was being observed.  Prior to ruling out linear models due to the inherent non-linearity of the problem, coefficients were reported with their deviations across the fold to asses their predictive power.  

### Final Model Results
The final model and feature space was quite small compared to the 600 available features.  In the end it was found that select statistics about home's prices in their nearest census block group and high level tract group were extremely powerful predictors of home prices, more so that explicit consideration of a direct range of neighbors.  Following these features, a set of unique features directly related to the homes contained high predictive power.  Finally, the first principle component of the boolean features, which only a owner would know, were found to contribute to a more accurate model.  After considering all the available data it is clear that the feature space is highly co-linear, homes which contain 2 bedrooms with not contain 9 bathrooms.  Further more the feature space is highly non-linear with the number of bedrooms correlating with lower prices when they exceed a threshold.  

The optimal predictor of the price was consistently a random forest.  Scores were within 2-5% using boosting approaches and a multivariate-adaptive 

#### Models results in optimal feature space

Hold-out test set score containing the most recently priced homes.

On all market homes:

One 95 percent of market homes:

| Model        |  Median Absolute Error  |Percent Error    |
| ------------- |:---------------------:|:---------------:|
| Tuned Random Forest  |  13,750   | 9.6 |

On inter-quartile range of market homes:

| Model                |  Median Absolute Error  |Percent Error    | 
| ---------------------|:--------------:|:----------------:|
| Tuned Random Forest  |  13,000         |             9.1 |  

Cross-Validation set containing over 200,000 homes:
On all market homes:

| Model        |  Median Absolute Error  |Percent Error    |
| ------------- |:---------------------:|:---------------------------:|
| Tuned Random Forest  |  13,400  | 15.1 |

One on inter-quartile range of market homes:

| Model        |  Median Absolute Error  |Percent Error    |
| ------------- |:---------------------:|:---------------------------:|
| Tuned Random Forest  |  14,400   | 11.0 |   

Cross-validated models in feature spaces which did not explicitly consider the price of neighbors never were lower than $30,000 Median Absolute Error.

#### Comparing to Realtors
The realtor's initial listing price in the hold-out set has a 5% error relative to closing and $7,500 Median Absolute errors. The realtor's listing price however remains a highly biased estimator for the price of a home.  The home is unlikely to sell for any price that is much higher or lower than what is listed.  

Comparing to homes which do not sell the power of the prototype model is clear.  The predicted price of the model relative to the realtor's listing price is characterized by twice the Median Absolute difference from the realtors listing price of homes that do sell. 

#### Comparing Across Time
It is important to state that the nature of the census block statistics and an explicitly include time variable allow for the model to perform well across time. Not just in a static matter.  This was validated using a time forward cross-validation approach where sections of time were trained on and then subsequent time points in the future were used for testing.  The errors reported above were consistent across these time chunks.  

### [Production Implementation](https://github.com/JOSMANC/RealtorBuddy/tree/master/production_code)
The production prototype would behave as follows.

1. A web form would intake a users address and all the information which they have available for their home.  The form would mirror that of official realtor listing documents. 
2. Relevant information which consumers are not immediately aware of, namely census tract, latitude, longitude are computed by querying against availible APIS 
3. A query for the relevant statistics for the area would be computed and aggregated for the input home
4. The value predicted by the model is rounded to the nearest 1000, in line with how most prices are routinely given in for homes.
5. In addition the inputed information would be added to a form which would be required when the home is listed.
 
### Future Considerations 
1. Further gains in the model could be achieved by creating locale nearest neighbor models which are associated home features.  This would require substantially larger compute time relative to the currently proposed model.  
2. Realtor have the option of testing the market by proposing prices which they know exceeds the true worth of a home.  This bias could be considered explicitly in how the model is fit by considering a bias cost function for the model.  scikit-learn, the machine learning library used for this prototype does not explicitly allow for such a function. It is my intention to create routines which would allow myself and others to apply biases to the cost functions. 

### Final Notes on Predicting Time to Sell

The prototype was originally intended to allow for both price and time to contract prediction based on the market price which the model defines, the relative difference of a users listing price relative to the market price.  Further improvements in the pricing model are required before such an prototype is possible. 

----
##If you like this be sure to check out your other buddy, [WoodCutterBuddy](https://github.com/JOSMANC/WoodCutterBuddy), your new friend for saving on wood costs for your next project.
