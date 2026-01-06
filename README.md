# Project MidTerm Report: Stock Price Prediction Using LSTM Model

Learning Python and applying it to stocks has been an exciting journey. Unlike many other programming languages, Python feels more like writing in English, making it fun. 
Below is a detailed breakdown of everything I learned and implemented to make this project a reality.


##  1. Python Basics

Building a strong foundation was the first step.
### Data Types & Structures
Python allows for dynamic typing, meaning a variable's type can change during runtime. I explored the common data types:
* **Integers & Floats:** For handling prices and volumes.
* **Strings:** Text like ticker symbols ("AAPL").
* **Booleans:** True/False flags for logic.

I also mastered the core data structures:
* **Lists:** Versatile and mutable; I used them to store sequences of data.
* **Dictionaries:** These store data as key-value pairs, which is perfect for organizing configuration parameters.

### Operators, Loops, & Functions
* **Operators:** Python supports basic arithmetic like addition and exponentiation (`**`).
* **Loops:** I used `For Loops` to iterate over my datasets and `While Loops` to run training epochs.
* **Functions:** Defining functions using `def` helped me keep my code clean and reusable.
* **Classes:** Understanding Object-Oriented Programming was crucial. I learned to define classes with `__init__`, which became essential later when building custom models in PyTorch.


##  2. Libraries in Python

The availability of frameworks made working with data and AI models so much easier.

### NumPy
Numpy is fundamental for numerical computing.
* **Arrays:** I learned that Numpy arrays are far more efficient than Python lists for math operations.
* **Indexing and Slicing:** Essential for preparing time-series data windows.
* **Numpy Maths:** Functions like `np.mean()` helped me normalize my data.

### Pandas
Pandas is a powerhouse for data manipulation.
* **Series & DataFrames:** I used DataFrames (2D) to structure the OHLC (Open, High, Low, Close) stock data.
* **Read CSV/JSON:** Loading datasets was simple using `pd.read_csv()`.
* **Locate Function:** I used `.loc[]` and `.iloc[]` to filter specific dates and prices.

### Matplotlib
Visualization is key to understanding data.
* **Plotting:** I used `plt.plot()` to visualize price trends vs. my predictions.
* **Subplots:** Great for stacking price charts and volume bars.
* **Histograms & Pie Charts:** Helped in analyzing the distribution of returns and portfolio allocation.

### Sklearn
I used Scikit-Learn for preprocessing and baseline models.
* **Linear Regression:** I used this to establish a simple baseline trendline before moving to complex deep learning models.


##  3. Introduction to Stock Market

To understand *what* I was predicting, I dove into resources from **Zerodha Varsity**.
* I learned that the stock market is essentially a mechanism for price discovery driven by supply and demand.
* **Volatility:** Prices don't move in straight lines; understanding market noise was a big realization.
* **Indices:** Learned how Nifty and Sensex act as barometers for the overall market health.


##  4. Machine Learning

I explored the broader landscape of ML before narrowing down my approach.
* **Types of ML:**
    * **Supervised:** Learning from labeled data (this is what I used for price prediction).
    * **Unsupervised:** Finding hidden patterns (like clustering similar stocks).
    * **Reinforcement:** Learning through trial and error (often used in trading bots).
    
* **Linear vs. Logistic Regression:**
    * **Linear Regression:** Predicts a continuous value (like tomorrow's price).
    * **Logistic Regression:** Predicts a probability (0 or 1), useful for predicting "Buy" vs "Sell" signals.
    
* **The Overfitting Issue:**
    * I ran into a problem where my model memorized the training data but failed on new data.
    * **Solution:** I learned to use regularization and simpler architectures to fix this.


##  5. Technical Analysis

Using Zerodha Varsity, I learned that "History tends to repeat itself."
* **Candlesticks:** I learned to read OHLC candles to gauge market sentiment.
* **Indicators:** I studied Moving Averages (SMA/EMA) and RSI. These aren't just lines on a chart; they became extra **features** (inputs) for my neural network to help it learn momentum.


##  6. YFinance Library

Data is the fuel for Deep Learning.
* Instead of downloading manual CSVs, I used the `yfinance` library.
* **Fetching Data:** With `yf.download()`, I could pull years of historical data for any ticker symbol in seconds. This allowed me to test my model on multiple stocks easily.


##  7. Neural Networks

Neural networks are powerful models inspired by the human brain.
* **Basics:** They consist of layers of neurons—an input layer, hidden processing layers, and an output layer.
* **Mechanism:** Each neuron takes an input, multiplies it by a "weight" (importance), adds a "bias," and passes it through an activation function.


##  8. Gradient Descent & Backpropagation

To understand how the network learns, I had to get into the math.
* **Gradient Descent:** This is the optimization algorithm. It iteratively tweaks the weights to minimize the error (loss).
* **Backpropagation:** I used the example of Logistic Regression to grasp this. It’s essentially the chain rule of calculus, calculating how much each specific weight contributed to the error and adjusting it backward.


##  9. PyTorch

I chose PyTorch for its flexibility and dynamic computation graphs.
* **Tensors:** The fundamental building block (like Numpy arrays, but GPU-ready).
* **Autograd:** This feature automatically calculated the gradients for me, saving me from writing complex calculus code from scratch.
* **Dataset & DataLoader:** I implemented custom classes to batch my time-series data efficiently.
* **Implementation:** I built Linear and Logistic regression models from scratch in PyTorch to understand the syntax before building the LSTM.
* **Saving/Loading:** Learned to use `torch.save()` so I wouldn't lose my trained model after closing the notebook.


##  10. LSTM (Long Short-Term Memory)

Standard Neural Networks struggle with time-series data because they have no "memory" of what happened 100 days ago. Enter LSTM.
* **Basics:** LSTM is a type of Recurrent Neural Network (RNN) designed to learn long-term dependencies.
* **How it Works:** It uses a gate mechanism to control the flow of information:
    1.  **Forget Gate:** Decides what history is irrelevant and should be thrown away.
    2.  **Input Gate:** Decides what new information is important to store.
    3.  **Output Gate:** Decides what the next hidden state should be.
    
**Key Takeaway:** This architecture allows the model to capture trends (like a long-term bull market) while ignoring temporary noise, making it perfect for stock price prediction.


### Acknowledgement
A huge shoutout to the mentors who guided me. Their constant support, no matter how small or big the issue, made learning so much easier and less intimidating. Thank you for taking the time and effort to help me grow; I am incredibly grateful for your encouragement and expertise.
