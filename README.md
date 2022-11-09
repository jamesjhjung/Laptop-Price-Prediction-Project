# Predicting Price of Laptop Computers

## Project/Goals
The goal of this project is to create a prediction model for the price of laptop computers given a laptop's specifications and components. This model will serve as a form of a benchmarking tool to determine adequate pricing of the laptop model of interest.

## Process
### EDA
First, exploratory data analysis was performed, where the raw dataset was examined along with its variables.
### Feature Engineering
Most of my data consisted of categorical variables and many of them thus required transformations in order to use them for modeling. Many of the string values stored in each variables were transformed and manipulated, and encoded for modeling.
### Modeling
After running many different regression models on default parameters, random forest regressor was found to be the best performing one out of the preselected group. The hyperparameters for random forest were tuned and the model was optimized for maximum performance.

## Results
The model can successfully predict a price of a laptop, with a mean absolute error of around 196 euros, and an R2 value of around 0.83.

## Challenges
Because the values stored in the raw dataset was relatively messy, it took some time transforming them to be adeqaute for fitting onto the model. Additionally, I initialy aimed to deploy the model on AWS, and the problems I encountered during the process took up a great portion of the allotted time, and I had no choice but to discard the step entirely.

## Future Goals
I would like to try a different form of encoding for my categorical variables, because many of the components of a laptop are directly correlated to the price of the laptop in whole, and thus can be ranked based on each component's price accordingly. I would also like to update the dataset to be able to represent newer releases of laptops and their components, as the dataset was last updated two years ago, and most of the newer components were excluded from the model. Lastly, I would like to revisit the deployment process, and possibly even deploy the model onto a website for ease of use.