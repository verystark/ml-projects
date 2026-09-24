# Course exercises
Exercises and projects completed in my university ML/AI course, organized by week.

### Week 1 - Stock market plotting
Loaded the OMXH25 companies one by one, plotted their stock prices, and calculated how an investment would have performed between two dates. Identified the company that provided the best return.

### Week 2 - Linear model and stock market trends
Derived the analytical solution for the parameters of a linear model and implemented it in Python. Applied the model to OMXH25 stock data to compare companies against their linear price trends and identify potentially underrated companies.

### Week 3 - Image classification with 1-NN
Implemented a 1-nearest-neighbor (1-NN) classifier for a dataset containing 60,000 training images and 10,000 test images. Used NumPy vectorization, linear algebra, matrix multiplication, and batching to improve the performance of the classifier while maintaining its accuracy.

The optimized implementation achieved 84.75% classification accuracy with a runtime of approximately 3 seconds.

### Week 4 - Stock market prediction with neural regression
Implemented a neural network to predict future stock prices for the 25 largest OMX Helsinki companies. The model was trained using historical adjusted closing prices up to the end of 2025.

For each company, sliding windows was used to create training data where the previous 100 days (M=100) were used as input and the following 10 days (K=10) as the target output. The training data from all companies was combined to train the shared model. Separate test data was created from 2026 data to evaluate the models on previously unseen data.

Different neural network configurations were tested by varying the number of layers, activation functions, number of epochs, and learning rate. The models were evaluated using mean squared error (MSE). Finally, the trained model was used to predict the adjusted closing price of KEMIRA for Monday, September 28.