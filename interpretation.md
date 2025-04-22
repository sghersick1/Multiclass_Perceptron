# Interpretations

## Reflection

What features did you use in your training?
Alcohol,Malic acid,Ash,Alcalinity of ash,Total phenols,Flavanoids,Nonflavanoid phenols,Proanthocyanins,Color intensity,Hue,OD280/OD315 of diluted wines (all columns except 5 and 13)

we used the classes for our target variables

## Errors

How many errors occured on each epoch for each learned model? Did the models differ? Summarize what you saw in the output graphs.
model 1 vs rest: converged after just one iter
model 2 vs rest: started with 14 errors then quickly converged to zero in around 35ish epochs  
model 3 vs rest: converged after just one iter

## Test data results

### What were your test data results?

How well did the multi-class perceptron seem to work? Did each model seem to work similarly well, or were some classes easier to predict correctly? Did you try other hyperparameters before settling on these? Discuss the process and success.

In this second revision of my program I saw much better results, most because I correctly calculated for the prediction with the net_input like stated in the book. Class 3 had 100% accuracy while the other two classes had 95+% accuracy which I was happy with. I tried a few different hyper parameters, one thing I noticed was that the models converged pretty quickly no matter the eta value. I ended up decidin with a relatively low eta and # of epochs because that's where I saw the best results. I didn't run a formal GridSearch but over all I believe the results were positive.

## Reflection

Answer both of the following questions:

1. What was the hardest part about getting this code working?
   something that I lacked at first and came to bite me in the butt was planning. I tried to get right into solving the problems and I was getting
   constantly stuck as a result. I then deleted most of my code and started again, this time planning much more thoroughly. Also, It was tricky working with the book's Perceptron code at first because I kept trying to use Pandas DataFrame
   instead of arrays. Eventually once I restarted I was able to figure everything out and I am pretty happy with how my code turned out.

   Here are the steps that I ended up planning out before retackling the code:

   # Planned steps

   1. read in data from .csv file into a pandas DataFrame
   2. Observe data using a box plot and .describe()
   3. remove columns 5 and 13 which were on a drastically bigger scale
   4. split data into X and y, remove row 0 (labels), and then put X and y into nparrays
   5. split into training, validation, and testing data
   6. Standard scale my X data
   7. Train each of the three one vs. rest models
   8. print results testing each model with validation data (weights, bias, confusion matrix, and errors)
   9. Create method that puts observations through each model, records predictions, and assigns observatin based on times predicted
   10. test data with Testing data and print out confusion matrix and classification report
