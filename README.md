# LT2212 V20 Assignment 1
Catherine Viloria

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

Part 1 - Convert the data into a dataframe
    For tokenizing, I decided to filter out integers and punctuation through .isalpha(). I also used .lower() to make each word into a lowercase. Although there are many words that are proper nouns and require an uppercase letter every time it is used, I decided that it was more important to standardise all the words that appear at the beginning of each sentence. This occurrence happens much more frequently and must be addressed. 


Part 4 - Visualize again
    After tf-idf was performed, words that appear in more articles have a much smaller score compared to words that appear in few articles. Through tf-idf, common words are bypassed and highlights words that appear less frequently. Through this process, it can be said that a lot of function words are filtered out and we are mostly left with content words. 


Part Bonus - Classify
    Classification accuracy on training data
        raw count: 0.9396551724137931
        with tf-idf: 0.9439655172413793
    
   The accuracy score with tf-idf is 0.004310344828 more than the accuracy score using the raw count. As said in Part 4, tf-idf is able to reduce the weight of more common words that appear in more documents within the corpus. Through this, we are able to highlight more interesting words and can use that to predict which class an article belongs to. Since the classes are based on topics of the article, it is more important to look at content words and words that will only really appear in specific classes. Function words are likely to be in articles for both classes. Tf-idf is also able to filter out function words and words that both classes might share. In the case of crude and grain, the word 'stocks' may appear just as frequently in both classes so it is important that tf-idf is able to identify this.