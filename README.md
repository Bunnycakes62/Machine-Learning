# Machine-Learning
This is a collection of machine learning projects I have been exploring over the years. Projects range from feature engineering, to Monte Carlo techniques, to reinforcement learning, and natural language processing. Here's a break down of what I'm interested in and what I've been up to.

### Computer Vision -- MY CURRENT RESEARCH INTEREST
I love love love computer vision models since they have so many applications and so many platforms to explore new techniques. This is my current field of research, so for my latest in greatest in computer vision please check out my [Siren repo](https://github.com/Bunnycakes62/Siren), a project which utilizes sinusoidal activation functions to generate high quality implicit image reconstruction.

For other projects in the field of vision, check out `SSBU Stage Identifier`, a simple CNN that maps pixel inputs to nine categorical outputs, namely the main competition stages featured in Super Smash Bros Ultimate.

### Natural Language Processing
`ToxicCommentParser.py` is a project that explores different sequential architectures to categorize comments extracted from Reddit as toxic, severe toxic, obscene, threat, insult, and identity hate. This project utilized Long-Short Term Memory networks (LSTMs), Bidirectional RNNs, and Gated Recurrent Networks (GRUs). Ultimately it was optimized using LSTMs. This was also an exploration in natural language processing techniques: from preprocessing the data, to word tokenizing (splitting up the data into individual words known as tokens), to word embedding (a learned representation for text where tokens that have the same meaning have similar representations).

### Generative Adversarial Networks
`MapsGAN` is my first attempt at building a generative adversarial network that intakes satellite images of maps and generates a google style map. Writen using keras, this is one of the more challenging concepts I've explored. A discriminator, encoder, and decoder network were all built and optimized separately then brought together to train the generator network.

### Transfer Learning
Remember that internet challenge the Getty hosted in the beginning of the covid lockdown where people staged photos of their favorite pieces of artwork and posted it on reddit? No? Well [here it is!](https://www.reddit.com/r/funny/comments/ftdalk/the_getty_is_challenging_quarantined_people_to/). Inspired by this, I made `Stylizer` which was a fun project that trained a network to stylize my family photos in the likening of whichever artist I chose (from the short lists of artists that I scraped from the MET open source dataset, see `DeepArtist`). Though this effect could be achieved using GANs, this project used a deep CNN architecture and a specialized loss metric, namely a gram matrix which minimized both the content loss (the picture of my family) and the style loss with respect to the overall output image.

### Reinforcement Learning
This section contains classic textbook projects such as an exploration of the `OneArmBandit.py`, a problem that explores the exploration-exploitation tradeoff common in recinforcement learning problems in the form of a virtual gambler pulling the arms of slot machines. The goal is to maximize the sum of the rewards through a series of lever pulls, which sample from some unknown probablity distribution. 

Two other projects related to this field are `Roomba.py` a robot that is optimized to visit and 'clean' every square on some board and `TicTacToe.py` in which the goal is to train an agent to beat an optimized opponent in a game of chess. Both projects are an exploration of Q-learning.

### More on the Data Science Side
A submission to the classic Kaggle feature engineering competition, `TitanicCompetition` is an exploration in data cleaning, data handling, feature engineering, and visualization. 

At the onset of the Coronavirus outbreak, a dataset was created and hosted on kaggle which contained self-reported novel corona virus infections categorized by state/province, country, date updated, and other metrics such as confirmed, suspected, recovered, and deaths. Using a Monte Carlo Markov Chain, `MCMCCorona` is a program that attempted to predict the spread of confirmed cases at n-day intervals with some varying number of forecasts put forth.

# More Projects on the WAY!!!!
I am currently learning about recommender systems in the hopes to build a more accurate netflix recommender. More accurate than Netflix you scoff? Well, not really. I have the unique challenge of sharing a netflix account with both my husband and my son and we have the continual problem of not logging into our own profiles when we watch our shows. Because of that I am constantly being bombarded with recommendations for cartoons or true-crime, both of which are not my cup of tea. So why not approach this problem in the context of machine learning? Set a clustering algorithm to separate my husband, my son, and my preferences and see if we can't get a self-curated recommender system going? We'll see where it lands. Unfortunately, I've also recently discovered a reddit scraping API that is very distracting, but oh so fun.
